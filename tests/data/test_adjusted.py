"""The composition-adjusted price index (housing_star.data.adjusted)."""

from __future__ import annotations

from datetime import date
from math import exp, log

import numpy as np
import polars as pl
import pytest

import housing_star as hs
from housing_star.config import AdjustmentConfig
from housing_star.data.adjusted import SALE_COLUMNS, adjusted_levels

# True effects used to generate exact (noise-free) sales.
A = {2000: 12.0, 2001: 12.1, 2002: 12.25, 2003: 12.2, 2004: 12.4}
B = {"U1": 0.5, "U2": 0.0, "U3": -0.2}
C = {"DETACHED": 0.3, "TERRACED": 0.0}

# The mix changes every year, and the periods are linked only through a chain of postcodes
# (2000-2001 share U1 and U2, 2002 joins through U3 and U1, 2003 through U2 and U3).
MIX = [
    (2000, "U1", "TERRACED"),
    (2000, "U2", "TERRACED"),
    (2000, "U2", "DETACHED"),
    (2001, "U1", "DETACHED"),
    (2001, "U2", "TERRACED"),
    (2001, "U3", "TERRACED"),
    (2002, "U3", "TERRACED"),
    (2002, "U3", "DETACHED"),
    (2002, "U1", "TERRACED"),
    (2002, "U2", "TERRACED"),
    (2003, "U2", "DETACHED"),
    (2003, "U3", "TERRACED"),
]


def frame(rows):
    """Sales from ``(node, year, unit, ptype, log_price)`` rows."""
    return pl.DataFrame(
        [(node, date(year, 1, 1), unit, ptype, lp) for node, year, unit, ptype, lp in rows],
        schema=dict(
            zip(SALE_COLUMNS, [pl.Utf8, pl.Date, pl.Utf8, pl.Utf8, pl.Float64], strict=True)
        ),
        orient="row",
    )


def exact(node="N", mix=MIX, a=A, b=B, c=C):
    return [(node, year, unit, ptype, a[year] + b[unit] + c[ptype]) for year, unit, ptype in mix]


def log_levels(out, node="N"):
    rows = out.filter(pl.col("postcode") == node).sort("bucket_start")
    return {d.year: log(p) for d, p in zip(rows["bucket_start"], rows["price"], strict=True)}


def test_the_index_recovers_price_changes_that_the_median_misses():
    out = adjusted_levels(frame(exact()))
    levels = log_levels(out)
    for year in (2001, 2002, 2003):
        assert levels[year] - levels[year - 1] == pytest.approx(A[year] - A[year - 1], abs=1e-9)
    medians = (
        frame(exact())
        .group_by("bucket_start")
        .agg(pl.col("log_price").median())
        .sort("bucket_start")["log_price"]
    )
    # 2000 to 2001: the median falls 0.2 while prices rose 0.1.
    assert medians[1] - medians[0] == pytest.approx(-0.2)
    assert A[2001] - A[2000] == pytest.approx(0.1)


def test_the_level_is_the_log_price_of_the_average_training_sale():
    out = adjusted_levels(frame(exact()))
    shift = np.mean([B[unit] + C[ptype] for _, unit, ptype in MIX])
    assert log_levels(out) == pytest.approx({year: A[year] + shift for year in range(2000, 2004)})


def test_output_columns_and_counts():
    out = adjusted_levels(frame(exact()))
    assert out.columns == ["postcode", "bucket_start", "price", "n"]
    assert out["n"].to_list() == [3, 3, 4, 2]
    assert out["n"].dtype == pl.UInt32


def test_holdout_sales_never_change_a_training_level():
    base = adjusted_levels(frame(exact()), holdout_from=date(2003, 1, 1))
    wild = exact() + [
        ("N", 2003, "NEW", "TERRACED", 20.0),
        ("N", 2004, "U1", "DETACHED", 9.0),
        ("N", 2004, "U2", "TERRACED", 14.0),
    ]
    later = adjusted_levels(frame(wild), holdout_from=date(2003, 1, 1))
    training = pl.col("bucket_start") < date(2003, 1, 1)
    assert later.filter(training).equals(base.filter(training))


def test_holdout_levels_hold_the_training_effects_fixed():
    mix = [*MIX, (2004, "U1", "DETACHED"), (2004, "U3", "TERRACED")]
    out = adjusted_levels(frame(exact(mix=mix)), holdout_from=date(2003, 1, 1))
    levels = log_levels(out)
    assert levels[2004] - levels[2002] == pytest.approx(A[2004] - A[2002], abs=1e-9)
    assert levels[2003] - levels[2002] == pytest.approx(A[2003] - A[2002], abs=1e-9)


def test_an_unseen_postcode_and_type_in_the_holdout_count_as_average():
    rows = [*exact(), ("N", 2004, "NEW", "FLAT-MAISONETTE", 13.0)]
    out = adjusted_levels(frame(rows), holdout_from=date(2004, 1, 1))
    assert log_levels(out)[2004] == pytest.approx(13.0)


def test_an_unseen_type_in_a_known_postcode_keeps_the_postcode_effect():
    training = adjusted_levels(frame(exact()))
    rows = [*exact(), ("N", 2004, "U1", "FLAT-MAISONETTE", A[2004] + B["U1"])]
    out = adjusted_levels(frame(rows), holdout_from=date(2004, 1, 1))
    # U1 sits B["U1"] above the average training postcode, and the flat counts as average type.
    shift_b = B["U1"] - np.mean([B[u] for _, u, _ in MIX])
    expected = A[2004] + B["U1"] - shift_b
    assert log_levels(out)[2004] == pytest.approx(expected)
    assert log_levels(training)[2003] == pytest.approx(log_levels(out)[2003])


def test_a_training_outlier_is_trimmed_and_does_not_move_the_level():
    # Each sale five times over, so one outlier has the leverage it would in a real period.
    many = exact() * 5
    clean = adjusted_levels(frame(many))
    out = adjusted_levels(frame([*many, ("N", 2001, "U2", "TERRACED", A[2001] + 2.0)]))
    assert log_levels(out) == pytest.approx(log_levels(clean))
    assert out["n"].to_list() == clean["n"].to_list() == [15, 15, 20, 10]


def test_a_holdout_outlier_is_trimmed_before_the_mean():
    rows = [
        *exact(),
        ("N", 2004, "U1", "TERRACED", A[2004] + B["U1"]),
        ("N", 2004, "U2", "TERRACED", A[2004]),
        ("N", 2004, "U3", "TERRACED", A[2004] + B["U3"] + 3.0),
    ]
    out = adjusted_levels(frame(rows), holdout_from=date(2004, 1, 1))
    holdout = out.filter(pl.col("bucket_start") == date(2004, 1, 1))
    assert holdout["n"].item() == 2
    clean = adjusted_levels(frame(rows[:-1]), holdout_from=date(2004, 1, 1))
    assert log_levels(out)[2004] == pytest.approx(log_levels(clean)[2004])


def test_a_trim_wide_enough_keeps_every_sale():
    rows = [*exact(), ("N", 2001, "U2", "TERRACED", A[2001] + 2.0)]
    out = adjusted_levels(frame(rows), config=AdjustmentConfig(trim=10.0))
    assert out.filter(pl.col("bucket_start") == date(2001, 1, 1))["n"].item() == 4


def test_a_period_linked_by_no_postcode_has_no_level():
    rows = [
        *exact(),
        ("N", 2004, "LONE1", "TERRACED", 12.0),
        ("N", 2004, "LONE2", "DETACHED", 12.5),
    ]
    out = adjusted_levels(frame(rows))
    assert sorted(log_levels(out)) == [2000, 2001, 2002, 2003]


def test_only_the_largest_linked_group_of_periods_is_kept():
    side = [("N", year, "SIDE", "TERRACED", 11.0 + 0.1 * (year - 2005)) for year in (2005, 2006)]
    out = adjusted_levels(frame(exact() + side))
    assert sorted(log_levels(out)) == [2000, 2001, 2002, 2003]


def test_a_node_whose_sales_are_all_outliers_has_no_levels():
    # Two postcodes moving in opposite directions by a factor of e**4: every residual is +-1.
    clash = [
        ("BAD", 2000, "X1", "TERRACED", 12.0),
        ("BAD", 2001, "X1", "TERRACED", 16.0),
        ("BAD", 2000, "X2", "TERRACED", 12.0),
        ("BAD", 2001, "X2", "TERRACED", 8.0),
    ]
    out = adjusted_levels(frame(exact() + clash))
    assert out["postcode"].unique().to_list() == ["N"]


def test_nodes_are_fitted_independently():
    alone = adjusted_levels(frame(exact()))
    other = exact(node="M", b={"U1": 9.0, "U2": 0.0, "U3": -4.0})
    other = [(n, y, "M-" + u, t, lp) for n, y, u, t, lp in other]
    both = adjusted_levels(frame(exact() + other))
    assert both.filter(pl.col("postcode") == "N").equals(alone)
    assert sorted(both["postcode"].unique().to_list()) == ["M", "N"]


def test_every_sale_in_the_holdout_gives_a_trimmed_mean_of_log_prices():
    rows = [
        ("N", 2000, "U1", "TERRACED", 12.0),
        ("N", 2000, "U2", "TERRACED", 12.2),
        ("N", 2000, "U3", "TERRACED", 15.0),
    ]
    out = adjusted_levels(frame(rows), holdout_from=date(1999, 1, 1))
    assert out["price"].item() == pytest.approx(exp(12.1))
    assert out["n"].item() == 2


def test_no_sales_give_an_empty_frame():
    out = adjusted_levels(frame([]), holdout_from=date(2000, 1, 1))
    assert out.is_empty()
    assert out.columns == ["postcode", "bucket_start", "price", "n"]


def test_a_missing_column_is_rejected():
    with pytest.raises(ValueError, match="missing column"):
        adjusted_levels(frame(exact()).drop("ptype"))


def test_trim_must_be_positive():
    with pytest.raises(ValueError, match="trim"):
        AdjustmentConfig(trim=0.0)


def test_adjustment_config_is_exported():
    assert hs.AdjustmentConfig is AdjustmentConfig
