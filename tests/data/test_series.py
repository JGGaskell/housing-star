from datetime import date

import polars as pl
import pytest

from housing_star.data.cache import TransactionCache
from housing_star.data.schema import TRANSACTION_SCHEMA
from housing_star.data.series import panel_series
from tests.conftest import fake_transaction_id, synthetic_transactions


def make_cache(tmp_path, rows):
    frame = pl.DataFrame(rows, schema=TRANSACTION_SCHEMA, orient="row")
    return TransactionCache(root=tmp_path, fetcher=lambda spec: frame)


def sale(price, month, postcode, ident=None):
    from datetime import date

    ident = ident or fake_transaction_id(price, month, postcode)
    return (
        price,
        date(2020, month, 1),
        postcode,
        "FLAT-MAISONETTE",
        ident,
        "ADD",
    )


def test_district_series_never_picks_up_longer_districts(tmp_path):
    cache = make_cache(tmp_path, [sale(100, 1, "E1 6AN"), sale(900, 2, "E14 9XY")])
    out = panel_series(["E1"], start="2020-01-01", end="2021-01-01", freq="1y", cache=cache)
    assert out.to_dicts() == [
        {"postcode": "E1", "bucket_start": out["bucket_start"][0], "price": 100.0, "n": 1}
    ]


def test_two_flats_sold_on_the_same_day_both_count(tmp_path):
    cache = make_cache(
        tmp_path,
        [
            sale(100, 1, "GL5 2AD", fake_transaction_id("flat", 1)),
            sale(300, 1, "GL5 2AD", fake_transaction_id("flat", 2)),
        ],
    )
    out = panel_series(["GL5"], start="2020-01-01", end="2021-01-01", freq="1y", cache=cache)
    assert (out["price"][0], out["n"][0]) == (200.0, 2)


def test_buckets_report_median_and_count(tmp_path):
    cache = make_cache(
        tmp_path,
        [sale(100, 1, "GL5 2AD"), sale(300, 2, "GL5 2AD"), sale(900, 3, "GL5 5XX")],
    )
    out = panel_series(["GL5"], start="2020-01-01", end="2021-01-01", freq="1y", cache=cache)
    assert (out["price"][0], out["n"][0]) == (300.0, 3)


def test_sector_nodes_filter_on_the_sector_digit(tmp_path):
    cache = make_cache(tmp_path, [sale(100, 1, "GL6 8AU"), sale(900, 1, "GL6 5AU")])
    out = panel_series(["GL6 8"], start="2020-01-01", end="2021-01-01", freq="1y", cache=cache)
    assert out["price"].to_list() == [100.0]


def test_panel_stacks_nodes_and_sorts_buckets(tmp_path):
    cache = make_cache(tmp_path, [sale(100, 1, "GL5 2AD"), sale(200, 7, "GL5 2AD")])
    out = panel_series(
        ["GL5", "GL6"], start="2020-01-01", end="2021-01-01", freq="6mo", cache=cache
    )
    gl5 = out.filter(pl.col("postcode") == "GL5")
    assert gl5["price"].to_list() == [100.0, 200.0]
    assert out.filter(pl.col("postcode") == "GL6").is_empty()


def test_default_cache_location_comes_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("HOUSING_STAR_CACHE", str(tmp_path))
    warm = make_cache(tmp_path, [sale(5, 1, "GL5 2AD")])
    warm.transactions("GL5", "2020-01-01", "2021-01-01")  # populate; later calls need no fetch
    out = panel_series(["GL5"], start="2020-01-01", end="2021-01-01", freq="1y")
    assert out["price"].to_list() == [5.0]


def test_download_plan_lists_nodes_that_still_need_downloading(tmp_path):
    from housing_star.config import GraphConfig
    from housing_star.data.series import download_plan

    cache = TransactionCache(root=tmp_path, fetcher=synthetic_transactions, verbose=False)
    graph = GraphConfig(radius_m=15_000, max_nodes=3)
    cold = download_plan("GL5", start="2000-01-01", end="2005-01-01", graph=graph, cache=cache)
    assert cold.columns == ["node", "fully_cached", "missing_from", "missing_to"]
    assert cold["node"][0] == "GL5" and cold.height == 3
    assert not cold["fully_cached"].any()
    assert cold["missing_from"].unique().to_list() == [date(2000, 1, 1)]

    panel_series(
        cold["node"].to_list(), start="2000-01-01", end="2005-01-01", freq="1y", cache=cache
    )
    warm = download_plan("GL5", start="2000-01-01", end="2005-01-01", graph=graph, cache=cache)
    assert warm["fully_cached"].all() and warm["missing_from"].null_count() == 3


def test_download_plan_defaults_to_the_default_cache(monkeypatch, tmp_path):
    from housing_star.data.series import download_plan

    monkeypatch.setenv("HOUSING_STAR_CACHE", str(tmp_path))
    plan = download_plan("GL5", start="2000-01-01", end="2001-01-01")
    assert not plan["fully_cached"].any()


def sold(year, unit, ptype, price):
    return (
        price,
        date(year, 6, 1),
        unit,
        ptype,
        fake_transaction_id(year, unit, ptype, price),
        "ADD",
    )


# Detached homes on GL5 2AD and terraces on GL5 3BB both rise 10% a year, but the share of sales
# that are detached falls from 2/3 to 1/3 to 0: the median falls while every price rises.
MIX_SHIFT = [
    sold(2018, "GL5 2AD", "DETACHED", 400_000),
    sold(2018, "GL5 2AD", "DETACHED", 400_001),
    sold(2018, "GL5 3BB", "TERRACED", 200_000),
    sold(2019, "GL5 2AD", "DETACHED", 440_000),
    sold(2019, "GL5 3BB", "TERRACED", 220_000),
    sold(2019, "GL5 3BB", "TERRACED", 220_001),
    sold(2020, "GL5 3BB", "TERRACED", 242_000),
    sold(2020, "GL5 3BB", "TERRACED", 242_001),
]


def test_adjusted_prices_follow_prices_when_the_median_follows_the_mix(tmp_path):
    cache = make_cache(tmp_path, MIX_SHIFT)
    window = dict(start="2018-01-01", end="2021-01-01", freq="1y", cache=cache)
    median = panel_series(["GL5"], **window)["price"].to_list()
    adjusted = panel_series(["GL5"], **window, prices="adjusted")["price"].to_list()
    assert median[1] < median[0]
    assert adjusted[1] / adjusted[0] == pytest.approx(1.1, rel=1e-5)
    assert adjusted[2] / adjusted[1] == pytest.approx(1.1, rel=1e-5)


def test_adjusted_holdout_periods_do_not_inform_the_training_prices(tmp_path):
    cache = make_cache(tmp_path, [*MIX_SHIFT, sold(2020, "GL5 9ZZ", "FLAT-MAISONETTE", 90_000)])
    window = dict(start="2018-01-01", end="2021-01-01", freq="1y", cache=cache)
    with_holdout = panel_series(["GL5"], **window, prices="adjusted", holdout=1)
    clean = panel_series(
        ["GL5"], start="2018-01-01", end="2020-01-01", freq="1y", cache=cache, prices="adjusted"
    )
    assert with_holdout["price"][:2].to_list() == pytest.approx(clean["price"].to_list())
    # The 2020 flat is in a postcode and type training never saw, so it counts as an average
    # home; at 90,000 that is about 1 log point below the period's others, and it is trimmed.
    assert with_holdout["n"].to_list() == [3, 3, 2]


def test_a_holdout_longer_than_the_data_leaves_no_training_effects(tmp_path):
    cache = make_cache(tmp_path, MIX_SHIFT)
    out = panel_series(
        ["GL5"],
        start="2018-01-01",
        end="2021-01-01",
        freq="1y",
        cache=cache,
        prices="adjusted",
        holdout=5,
    )
    assert out.height == 3  # every period levelled as a plain trimmed mean of log prices


def test_adjusted_prices_with_no_sales_are_empty(tmp_path):
    cache = make_cache(tmp_path, [])
    out = panel_series(
        ["GL5"],
        start="2018-01-01",
        end="2021-01-01",
        freq="1y",
        cache=cache,
        prices="adjusted",
        holdout=1,
    )
    assert out.is_empty() and out.columns == ["postcode", "bucket_start", "price", "n"]


@pytest.mark.parametrize(
    ("kwargs", "message"), [({"prices": "mean"}, "prices must be"), ({"holdout": -1}, "holdout")]
)
def test_panel_series_rejects_bad_arguments(tmp_path, kwargs, message):
    cache = make_cache(tmp_path, MIX_SHIFT)
    with pytest.raises(ValueError, match=message):
        panel_series(
            ["GL5"], start="2018-01-01", end="2021-01-01", freq="1y", cache=cache, **kwargs
        )
