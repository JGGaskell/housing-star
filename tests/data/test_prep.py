from datetime import date

import numpy as np
import polars as pl
import pytest

from housing_star.config import CleaningConfig
from housing_star.data.prep import (
    Panel,
    _fill_row,
    _merge,
    drop_thin_nodes,
    fill_training_gaps,
    merge_sparse_nodes,
    panel_from_long,
    prepare_panel,
)

NAN = np.nan


def make_panel(prices, counts=None, eastings=None):
    prices = np.asarray(prices, dtype=float)
    N, T = prices.shape
    counts = (
        np.where(np.isfinite(prices), 10.0, NAN) if counts is None else np.asarray(counts, float)
    )
    east = np.arange(N) * 1_000.0 if eastings is None else np.asarray(eastings, float)
    return Panel(
        [f"PC{i}" for i in range(N)], east, np.zeros(N), prices, counts, [str(t) for t in range(T)]
    )


CLEAN = CleaningConfig(min_coverage=0.6, min_mean_n=5.0, min_nodes=2, max_gap=1)


def test_panel_from_long_pivots_onto_the_node_list():
    long = pl.DataFrame(
        {
            "postcode": ["B", "A", "A"],
            "bucket_start": [date(2020, 1, 1), date(2021, 1, 1), date(2020, 1, 1)],
            "price": [5.0, 6.0, 7.0],
            "n": [1, 2, 3],
        }
    )
    panel = panel_from_long(long, ["A", "B", "C"], np.zeros(3), np.zeros(3))
    np.testing.assert_array_equal(panel.prices, [[7, 6], [5, NAN], [NAN, NAN]])
    np.testing.assert_array_equal(panel.counts[0], [3, 2])
    assert panel.times == ["2020-01-01", "2021-01-01"]


def test_a_regular_calendar_keeps_a_gap_that_only_one_node_misses():
    days = [date(2020, 10, 1), date(2021, 1, 1), date(2021, 4, 1)]
    long = pl.DataFrame(
        {
            "postcode": ["A", "A", "A", "B", "B"],
            "bucket_start": [*days, days[0], days[2]],
            "price": [1.0, 2.0, 3.0, 4.0, 5.0],
            "n": [1, 1, 1, 1, 1],
        }
    )
    panel = panel_from_long(long, ["A", "B"], np.zeros(2), np.zeros(2))
    assert panel.times == ["2020-10-01", "2021-01-01", "2021-04-01"]
    np.testing.assert_array_equal(panel.prices[1], [4.0, NAN, 5.0])


def test_a_year_missing_for_every_node_is_rejected():
    long = pl.DataFrame(
        {
            "postcode": ["A", "A", "A", "B"],
            "bucket_start": [
                date(2019, 1, 1),
                date(2020, 1, 1),
                date(2022, 1, 1),
                date(2019, 1, 1),
            ],
            "price": [1.0, 1.0, 1.0, 1.0],
            "n": [1, 1, 1, 1],
        }
    )
    with pytest.raises(
        ValueError,
        match=(
            r"2020-01-01 to 2022-01-01 is 24 months, and the shortest step is 12 months"
            r".*at least one node"
        ),
    ):
        panel_from_long(long, ["A", "B"], np.zeros(2), np.zeros(2))


def test_the_reported_gap_is_the_longest_when_the_hole_comes_first():
    long = pl.DataFrame(
        {
            "postcode": ["A", "A", "A"],
            "bucket_start": [date(2018, 1, 1), date(2020, 1, 1), date(2021, 1, 1)],
            "price": [1.0, 1.0, 1.0],
            "n": [1, 1, 1],
        }
    )
    with pytest.raises(
        ValueError,
        match=r"2018-01-01 to 2020-01-01 is 24 months, and the shortest step is 12 months",
    ):
        panel_from_long(long, ["A"], np.zeros(1), np.zeros(1))


def test_dates_that_do_not_share_a_day_of_the_month_are_rejected():
    long = pl.DataFrame(
        {
            "postcode": ["A", "A", "A"],
            "bucket_start": [date(2020, 1, 1), date(2020, 4, 1), date(2020, 7, 15)],
            "price": [1.0, 1.0, 1.0],
            "n": [1, 1, 1],
        }
    )
    with pytest.raises(
        ValueError, match="2020-07-15 is not a whole number of months after 2020-04-01"
    ):
        panel_from_long(long, ["A"], np.zeros(1), np.zeros(1))


def test_two_dates_on_different_days_of_the_month_are_rejected():
    long = pl.DataFrame(
        {
            "postcode": ["A", "A"],
            "bucket_start": [date(2020, 1, 1), date(2020, 2, 15)],
            "price": [1.0, 1.0],
            "n": [1, 1],
        }
    )
    with pytest.raises(
        ValueError, match="2020-02-15 is not a whole number of months after 2020-01-01"
    ):
        panel_from_long(long, ["A"], np.zeros(1), np.zeros(1))


def test_two_dates_are_one_step_so_a_missing_year_between_them_is_invisible():
    """A hole shows up only when another gap reveals the step. Two dates have one step."""
    long = pl.DataFrame(
        {
            "postcode": ["A", "A"],
            "bucket_start": [date(2018, 1, 1), date(2020, 1, 1)],
            "price": [1.0, 1.0],
            "n": [1, 1],
        }
    )
    panel = panel_from_long(long, ["A"], np.zeros(1), np.zeros(1))
    assert panel.times == ["2018-01-01", "2020-01-01"]


def test_a_single_period_is_a_regular_calendar():
    long = pl.DataFrame(
        {
            "postcode": ["A"],
            "bucket_start": [date(2020, 6, 15)],
            "price": [1.0],
            "n": [1],
        }
    )
    panel = panel_from_long(long, ["A"], np.zeros(1), np.zeros(1))
    assert panel.times == ["2020-06-15"]


def test_panel_from_long_rejects_empty_input():
    empty = pl.DataFrame(
        schema={
            "postcode": pl.Utf8,
            "bucket_start": pl.Date,
            "price": pl.Float64,
            "n": pl.UInt32,
        }
    )
    with pytest.raises(ValueError, match="No transactions"):
        panel_from_long(empty, ["A"], np.zeros(1), np.zeros(1))


def test_thin_node_is_merged_into_its_nearest_neighbour():
    good = [100.0, 110, 120, 130]
    panel = make_panel(
        [good, [NAN, NAN, NAN, 50.0], [200.0, 210, 220, 230]], eastings=[0, 900, 5_000]
    )
    merged = merge_sparse_nodes(panel, CLEAN, train_end=4)
    assert merged.postcodes == ["PC2", "PC1+PC0"]


def test_merging_stops_at_min_nodes():
    panel = make_panel([[NAN, NAN, 1.0, 1.0], [NAN, NAN, NAN, 2.0]])
    assert merge_sparse_nodes(panel, CLEAN, train_end=4).postcodes == panel.postcodes


def test_merge_uses_count_weighted_prices_and_training_weighted_centroid():
    counts = [[1.0, 1.0, 9.0], [3.0, 3.0, 1.0], [1.0, 1.0, 1.0]]
    panel = make_panel(
        [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0], [1.0, 1.0, 1.0]],
        counts,
        eastings=[0, 1_000, 9_000],
    )
    out = _merge(panel, 0, 1, train_end=2)
    assert out.postcodes == ["PC2", "PC0+PC1"]
    np.testing.assert_allclose(out.prices[1], [17.5, 17.5, 11.0])  # (n0*p0 + n1*p1) / (n0 + n1)
    assert out.eastings[1] == pytest.approx(750.0)  # training counts 2 vs 6 -> 3/4 towards node 1


def test_merge_falls_back_to_equal_weights_without_training_counts():
    panel = make_panel([[NAN, 1.0], [NAN, 3.0]], [[NAN, 5.0], [NAN, 5.0]], eastings=[0, 100])
    assert _merge(panel, 0, 1, train_end=1).eastings[0] == pytest.approx(50.0)


def test_merged_cells_without_any_sales_stay_missing():
    panel = make_panel([[NAN, 1.0], [NAN, 3.0]], [[NAN, 5.0], [NAN, 5.0]])
    assert np.isnan(_merge(panel, 0, 1, train_end=1).prices[0, 0])


@pytest.mark.parametrize(
    ("row", "n_sales", "kept"),
    [
        ([1.0, 1, 1, 1], 10.0, True),
        ([1.0, NAN, NAN, NAN], 10.0, False),  # coverage 0.25
        ([1.0, 1, 1, 1], 1.0, False),  # mean sale count 1 < 5
    ],
)
def test_drop_thin_nodes_thresholds(row, n_sales, kept):
    base = [10.0, 10, 10, 10]
    counts = [[10.0] * 4, np.where(np.isfinite(row), n_sales, NAN)]
    out = drop_thin_nodes(make_panel([base, row], counts), CleaningConfig(min_nodes=2), train_end=4)
    assert (len(out.postcodes) == 2) is kept


def test_drop_thin_nodes_enforces_the_longest_gap():
    panel = make_panel([[1.0, NAN, NAN, NAN, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    config = CleaningConfig(min_coverage=0.5, min_nodes=2, max_gap=2)
    with pytest.raises(ValueError, match="All nodes were dropped"):
        drop_thin_nodes(panel, config, train_end=10)


def test_fill_row_interpolates_interior_gaps():
    filled, extrapolated = _fill_row(np.array([1.0, NAN, 3.0, 4.0]), 4)
    np.testing.assert_allclose(filled, [1, 2, 3, 4])
    assert not extrapolated


def test_fill_row_extrapolates_edges_linearly():
    filled, extrapolated = _fill_row(np.array([NAN, 2.0, 3.0, NAN]), 4)
    np.testing.assert_allclose(filled, [1, 2, 3, 4])
    assert extrapolated


def test_fill_row_with_one_observation_is_flat():
    filled, extrapolated = _fill_row(np.array([NAN, 5.0, NAN]), 3)
    np.testing.assert_allclose(filled, [5, 5, 5])
    assert extrapolated


@pytest.mark.parametrize("row", [[NAN, NAN, NAN], [1.0, 2.0, 3.0]])
def test_fill_row_leaves_empty_and_complete_rows_alone(row):
    filled, extrapolated = _fill_row(np.array(row), 3)
    np.testing.assert_array_equal(filled, row)
    assert not extrapolated


def test_fill_only_touches_the_training_window():
    panel = make_panel([[1.0, NAN, 3.0, NAN, 5.0]])
    out = fill_training_gaps(panel, CLEAN, train_end=3)
    np.testing.assert_array_equal(out.prices[0], [1, 2, 3, NAN, 5])


@pytest.mark.parametrize(("keep", "expect_nan"), [(False, True), (True, False)])
def test_extrapolated_rows_are_blanked_unless_kept(keep, expect_nan):
    panel = make_panel([[NAN, 2.0, 3.0, 4.0]])
    config = CleaningConfig(min_nodes=2, keep_extrapolated=keep)
    blanked = np.isnan(fill_training_gaps(panel, config, train_end=4).prices).all()
    assert bool(blanked) == expect_nan


def test_prepare_panel_returns_a_complete_matrix():
    panel = make_panel(
        [[100.0, 110, 120, 130, 140], [200.0, NAN, 220, 230, 240], [300.0, 310, 320, 330, 340]]
    )
    out = prepare_panel(panel, CLEAN, holdout=1)
    assert np.isfinite(out.prices).all()
    assert out.prices[1, 1] == pytest.approx(210.0)


def test_nodes_missing_in_the_holdout_are_dropped():
    panel = make_panel([[100.0, 110, 120, 130], [200.0, 210, 220, NAN], [300.0, 310, 320, 330]])
    assert prepare_panel(panel, CLEAN, holdout=1).postcodes == ["PC0", "PC2"]


def test_prepare_panel_fails_if_nothing_survives():
    panel = make_panel([[100.0, 110, 120, NAN], [200.0, 210, 220, NAN]])
    with pytest.raises(ValueError, match="missing observations"):
        prepare_panel(panel, CLEAN, holdout=1)


@pytest.mark.parametrize("holdout", [-1, 4])
def test_prepare_panel_rejects_invalid_holdout(holdout):
    with pytest.raises(ValueError, match="holdout"):
        prepare_panel(make_panel([[1.0, 2, 3, 4]]), CLEAN, holdout=holdout)


def test_prepare_panel_rejects_non_positive_training_prices():
    panel = make_panel([[100.0, 0.0, 120, 130], [200.0, 210, 220, 230]])
    with pytest.raises(ValueError, match="Non-positive"):
        prepare_panel(panel, CLEAN, holdout=1)


def test_holdout_values_cannot_change_the_training_data():
    """Leakage guard: two panels that differ only in hold-out values prepare identically."""
    train = [
        [100.0, NAN, 120, 130],
        [NAN, 210, 220, 230],
        [300.0, 310, 320, 330],
        [1.0, NAN, NAN, 5.0],
    ]
    counts = [[10.0, NAN, 10, 10], [NAN, 10, 10, 10], [10.0, 10, 10, 10], [1.0, NAN, NAN, 1.0]]
    a = make_panel([[*row, 140.0] for row in train], [[*row, 10.0] for row in counts])
    b = make_panel([[*row, 99_999.0] for row in train], [[*row, 10.0] for row in counts])
    out_a, out_b = prepare_panel(a, CLEAN, holdout=1), prepare_panel(b, CLEAN, holdout=1)
    assert out_a.postcodes == out_b.postcodes
    np.testing.assert_array_equal(out_a.prices[:, :4], out_b.prices[:, :4])
    np.testing.assert_array_equal(out_a.eastings, out_b.eastings)
