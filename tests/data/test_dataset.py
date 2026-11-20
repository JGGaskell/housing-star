from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from pandera.errors import SchemaError

from housing_star.config import AdjustmentConfig, CleaningConfig, GraphConfig
from housing_star.data.cache import TransactionCache
from housing_star.data.dataset import PriceSeriesDataset, estimate_growth
from housing_star.spatial.graph import Neighbourhood
from tests.conftest import synthetic_transactions


def test_y_removes_the_training_window_log_mean(ds):
    np.testing.assert_allclose(ds.y_train().mean(axis=1), 0.0, atol=1e-12)
    assert ds.y_eval().shape == (ds.N, ds.n_eval)


def test_to_abs_inverts_the_transform(ds):
    np.testing.assert_allclose(ds.to_abs(ds.y), ds.prices)


def test_shape_properties(ds):
    assert (ds.N, ds.T, ds.n_eval, ds.train_end) == (6, 24, 3, 21)


@pytest.mark.parametrize(
    "change",
    [
        {"train_end": 0},
        {"train_end": 99},
        {"postcodes": ["only-one"]},
        {"times": ["2000-01-01"]},
        {"prices": np.full((6, 24), -1.0)},
        {"prices": np.full((6, 24), np.nan)},
    ],
)
def test_invalid_construction_is_rejected(ds, change):
    fields = {k: getattr(ds, k) for k in ("prices", "W", "postcodes", "times", "train_end")}
    with pytest.raises(ValueError):
        PriceSeriesDataset(**{**fields, **change})


def test_window_recomputes_demeaning_on_its_own_training_columns(ds):
    view = ds.window(10, 12)
    assert view.T == 12 and view.n_eval == 2
    np.testing.assert_allclose(view.y[:, :10].mean(axis=1), 0.0, atol=1e-12)
    assert not np.allclose(view.log_node_means, ds.log_node_means)
    assert view.target == ds.target


@pytest.mark.parametrize("args", [(0, 5), (6, 5), (5, 99)])
def test_window_rejects_invalid_bounds(ds, args):
    with pytest.raises(ValueError):
        ds.window(*args)


def test_with_holdout_and_train_only(ds):
    assert ds.with_holdout(5).n_eval == 5
    train = ds.train_only()
    assert (train.T, train.n_eval) == (ds.train_end, 0)


def test_index_of_matches_merged_labels(make_dataset):
    ds = make_dataset(n_nodes=3)
    merged = PriceSeriesDataset(
        prices=ds.prices,
        W=ds.W,
        postcodes=["A", "B+C", "D"],
        times=ds.times,
        train_end=ds.train_end,
    )
    assert [merged.index_of(x) for x in ("A", "C", "B+C", "Z")] == [0, 1, 1, None]


def test_target_idx(ds):
    assert ds.target_idx == 0


def test_target_idx_requires_a_target(ds):
    anonymous = PriceSeriesDataset(ds.prices, ds.W, ds.postcodes, ds.times, ds.train_end)
    with pytest.raises(ValueError, match="no target"):
        _ = anonymous.target_idx


def test_target_idx_requires_the_target_to_be_a_node(ds):
    lost = PriceSeriesDataset(ds.prices, ds.W, ds.postcodes, ds.times, ds.train_end, target="ZZ")
    with pytest.raises(ValueError, match="not among"):
        _ = lost.target_idx


def test_neighbour_order_is_nearest_first_and_excludes_self(ds):
    assert ds.neighbour_order(2).tolist() == [1, 3, 0, 4, 5]


def test_neighbour_order_needs_coordinates(ds):
    bare = PriceSeriesDataset(ds.prices, ds.W, ds.postcodes, ds.times, ds.train_end)
    with pytest.raises(ValueError, match="coordinates"):
        bare.neighbour_order(0)


def long_prices(nodes, years, price=lambda i, t: 100_000.0 * (1.05**t) * (1 + 0.1 * i)):
    from datetime import date

    rows = [
        {
            "postcode": node,
            "bucket_start": date(2000 + t, 1, 1),
            "price": price(i, t),
            "n": 10,
        }
        for i, node in enumerate(nodes)
        for t in range(years)
    ]
    return pl.DataFrame(rows)


def hood(n):
    return Neighbourhood([f"N{i}" for i in range(n)], np.arange(n) * 500.0, np.zeros(n))


def test_from_panel_builds_weights_and_split():
    ds = PriceSeriesDataset.from_panel(
        long_prices(hood(4).keys, 10), hood(4), target="N0", holdout=2
    )
    assert (ds.N, ds.T, ds.train_end) == (4, 10, 8)
    np.testing.assert_allclose(ds.W.sum(axis=1), 1.0)
    assert ds.times[0] == "2000-01-01"


def test_from_panel_needs_at_least_two_surviving_nodes():
    sparse = long_prices(hood(4).keys, 10).filter(
        (pl.col("postcode") == "N0") | (pl.col("bucket_start").dt.year() < 2002)
    )
    with pytest.raises(ValueError, match="at least 2 nodes"):
        PriceSeriesDataset.from_panel(sparse, hood(4), target="N0", holdout=1)


def test_from_panel_rejects_a_year_missing_for_every_node():
    prices = long_prices(hood(2).keys, 4).filter(pl.col("bucket_start").dt.year() != 2001)
    with pytest.raises(ValueError, match="not evenly spaced"):
        PriceSeriesDataset.from_panel(prices, hood(2), target="N0", holdout=1)


def test_from_panel_rejects_a_median_the_schema_forbids():
    prices = long_prices(hood(4).keys, 6).with_columns(
        pl.when(pl.col("postcode") == "N0").then(0.0).otherwise(pl.col("price"))
    )
    with pytest.raises(SchemaError, match="price"):
        PriceSeriesDataset.from_panel(prices, hood(4), target="N0", holdout=1)


def test_from_panel_requires_the_target_to_survive_cleaning():
    prices = long_prices(hood(4).keys, 10)
    with pytest.raises(ValueError, match="dropped during cleaning"):
        PriceSeriesDataset.from_panel(prices, hood(4), target="ELSEWHERE", holdout=1)


def test_from_postcode_reads_through_the_cache(tmp_path):
    cache = TransactionCache(root=tmp_path, fetcher=synthetic_transactions)
    ds = PriceSeriesDataset.from_postcode(
        "GL5",
        start="2000-01-01",
        end="2012-01-01",
        holdout=2,
        graph=GraphConfig(radius_m=15_000, max_nodes=4),
        cleaning=CleaningConfig(min_nodes=2),
        cache=cache,
    )
    assert ds.target == "GL5" and ds.n_eval == 2 and ds.postcodes[0] == "GL5"
    assert ds.T == 12


def test_from_postcode_can_build_adjusted_prices(tmp_path):
    cache = TransactionCache(root=tmp_path, fetcher=synthetic_transactions)
    ds = PriceSeriesDataset.from_postcode(
        "GL5",
        start="2000-01-01",
        end="2012-01-01",
        holdout=2,
        graph=GraphConfig(radius_m=15_000, max_nodes=4),
        cleaning=CleaningConfig(min_nodes=2),
        cache=cache,
        prices="adjusted",
        adjustment=AdjustmentConfig(trim=0.5),
    )
    assert ds.T == 12 and ds.n_eval == 2
    # Synthetic prices rise 6% a year with the same mix each year, so the index does too,
    # in the hold-out as well as the training years.
    np.testing.assert_allclose(np.diff(np.log(ds.prices), axis=1), np.log(1.06), atol=1e-4)


def test_from_postcode_uses_default_configs(tmp_path):
    cache = TransactionCache(root=tmp_path, fetcher=synthetic_transactions)
    ds = PriceSeriesDataset.from_postcode("GL5", start="2000-01-01", end="2012-01-01", cache=cache)
    assert ds.n_eval == 1


def test_growth_means_are_the_average_training_change_in_log_price(ds):
    log = np.log(ds.prices[:, : ds.train_end])
    np.testing.assert_allclose(ds.growth_means, np.diff(log, axis=1).mean(axis=1))


def test_the_slope_method_fits_a_line_through_every_training_year(ds):
    line = np.log(ds.prices[:, : ds.train_end])
    slope = replace(ds, growth_method="slope").growth_means
    t = np.arange(ds.train_end)
    expected = [np.polyfit(t, row, 1)[0] for row in line]
    np.testing.assert_allclose(slope, expected)
    assert not np.allclose(slope, ds.growth_means)


def test_growth_means_ignore_the_holdout_and_follow_the_window(ds):
    tampered = replace(ds, prices=np.where(np.arange(ds.T) >= ds.train_end, 1e9, ds.prices))
    np.testing.assert_array_equal(tampered.growth_means, ds.growth_means)
    view = ds.window(10, 12)
    np.testing.assert_allclose(view.growth_means, estimate_growth(np.log(ds.prices), 10, "mean"))
    assert not np.allclose(view.growth_means, ds.growth_means)


@pytest.mark.parametrize("method", ["mean", "slope"])
def test_growth_needs_two_training_periods(ds, method):
    np.testing.assert_array_equal(ds.window(1, 1).growth_means, 0.0)
    assert (replace(ds, growth_method=method).window(1, 3).growth_means == 0).all()
