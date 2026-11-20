"""A composition-adjusted price index for each node and period.

A median moves when the mix of homes sold changes as well as when prices change. The adjusted
index removes the mix. Within each node, every training sale's log price is modelled as

    log P = a[period] + b[unit postcode] + c[property type] + e

and fitted by least squares. A unit postcode (``"GL5 2AD"``) covers about fifteen addresses, so
``b`` absorbs which streets a period's sales came from and ``c`` which kinds of home. ``a`` is then
the price level of a fixed mix. Each node is fitted on its own sales.

Only training periods inform ``b`` and ``c``. Hold-out periods are levelled with those effects
held fixed, so data from the hold-out never changes a training value. A hold-out sale in a postcode
or property type the training window never saw is treated as the node's average one (effect 0).

How it is solved
    The postcode effects are eliminated exactly: subtracting each postcode's own mean from the
    sale prices and from the period and type indicators leaves a small least-squares problem in
    ``a`` and ``c`` alone (one unknown per period and per property type). It is solved directly,
    and ``b`` is each postcode's mean residual. There is no iteration and no tolerance.

Normalisation
    The effects are only defined relative to each other, so within each node they are shifted to
    average zero over its training sales. ``a`` is then the log price of the node's average
    training sale, and ``exp(a)`` is reported as the price.

Which periods are identified
    One period's level is compared with another's only through postcodes that sold in both,
    directly or through a chain of other periods. Within each node, training periods are split into
    groups that share no postcode, and only the group with the most sales is kept. A period whose
    sales all came from postcodes that sold in no other training period, for example, has no level
    and is left out, like a period with no sales.

Outliers
    Sales more than ``trim`` log points from their fitted value are dropped and the fit repeated
    once. In a hold-out period, sales more than ``trim`` from the period's median adjusted log price
    are dropped before the mean is taken.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from housing_star._types import FloatArray, IntArray
from housing_star.config import AdjustmentConfig

SALE_COLUMNS = ("node", "bucket_start", "unit", "ptype", "log_price")
"""Columns :func:`adjusted_levels` needs: node key, bucket date, unit postcode, property type and
the natural log of the price."""

_LEVELS = {"node": pl.Utf8, "bucket_start": pl.Date, "a": pl.Float64, "n": pl.UInt32}
_UNITS = {"unit": pl.Utf8, "b": pl.Float64}
_TYPES = {"node": pl.Utf8, "ptype": pl.Utf8, "c": pl.Float64}


def adjusted_levels(
    sales: pl.DataFrame,
    *,
    holdout_from: date | None = None,
    config: AdjustmentConfig | None = None,
) -> pl.DataFrame:
    """Composition-adjusted price for every node and period: ``postcode, bucket_start, price, n``.

    Parameters
    ----------
    sales : one row per sale with the columns in :data:`SALE_COLUMNS`.
    holdout_from : first hold-out bucket. Sales from it onwards do not inform the postcode and
        type effects. ``None`` treats every sale as training.
    config : trimming settings.

    ``n`` is the number of sales behind each price after trimming.

    Raises
    ------
    ValueError
        If a column in :data:`SALE_COLUMNS` is missing.
    """
    missing = [c for c in SALE_COLUMNS if c not in sales.columns]
    if missing:
        raise ValueError(f"sales is missing column(s) {missing}; expected {list(SALE_COLUMNS)}.")
    config = config or AdjustmentConfig()
    sales = sales.select(SALE_COLUMNS)
    if holdout_from is None:
        training, later = sales, sales.clear()
    else:
        training = sales.filter(pl.col("bucket_start") < holdout_from)
        later = sales.filter(pl.col("bucket_start") >= holdout_from)

    fits = [_fit_node(frame, config) for frame in training.partition_by("node")]
    periods = pl.concat([f[0] for f in fits]) if fits else pl.DataFrame(schema=_LEVELS)
    units = pl.concat([f[1] for f in fits]) if fits else pl.DataFrame(schema=_UNITS)
    types = pl.concat([f[2] for f in fits]) if fits else pl.DataFrame(schema=_TYPES)
    levels = pl.concat([periods, _holdout_levels(later, units, types, config)])
    return levels.select(
        pl.col("node").alias("postcode"),
        "bucket_start",
        pl.col("a").exp().alias("price"),
        "n",
    ).sort("postcode", "bucket_start")


def _fit_node(
    sales: pl.DataFrame, config: AdjustmentConfig
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """One node's training levels ``(node, bucket_start, a, n)`` and its unit and type effects."""
    kept = _largest_group(sales)
    _, _, _, resid = _solve(kept)
    kept = kept.filter(pl.Series(np.abs(resid) <= config.trim))
    if kept.is_empty():  # every sale was an outlier against the others
        return (
            pl.DataFrame(schema=_LEVELS),
            pl.DataFrame(schema=_UNITS),
            pl.DataFrame(schema=_TYPES),
        )
    kept = _largest_group(kept)
    a, b, c, _ = _solve(kept)
    # Normalise: unit and type effects average zero over the node's training sales.
    b_bar, c_bar = float(b.mean()), float(c.mean())
    fitted = kept.with_columns(
        a=pl.Series(a + b_bar + c_bar), b=pl.Series(b - b_bar), c=pl.Series(c - c_bar)
    )
    periods = fitted.group_by("node", "bucket_start").agg(
        pl.col("a").first(), pl.len().cast(pl.UInt32).alias("n")
    )
    units = fitted.group_by("unit").agg(pl.col("b").first())
    types = fitted.group_by("node", "ptype").agg(pl.col("c").first())
    return periods, units, types


def _codes(column: pl.Series) -> tuple[IntArray, int]:
    """Integer codes 0..k-1 for the distinct values of ``column``, and k."""
    codes = column.rank("dense").cast(pl.Int64).to_numpy() - 1
    return codes, int(codes.max()) + 1


def _largest_group(sales: pl.DataFrame) -> pl.DataFrame:
    """The sales of the largest group of periods linked to each other by shared postcodes."""
    period, n_periods = _codes(sales["bucket_start"])
    unit, n_units = _codes(sales["unit"])
    label = np.arange(n_periods)
    while True:  # each period takes the smallest label it can reach through a shared postcode
        by_unit = np.full(n_units, n_periods)
        np.minimum.at(by_unit, unit, label[period])
        spread = label.copy()
        np.minimum.at(spread, period, by_unit[unit])
        if np.array_equal(spread, label):
            break
        label = spread
    groups = label[period]
    sizes = np.bincount(groups, minlength=n_periods)
    return sales.filter(pl.Series(groups == int(np.argmax(sizes))))


def _solve(sales: pl.DataFrame) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Per-sale period, unit and type effects and residuals, by least squares.

    The unit effects are absorbed exactly: with ``D`` the period and type indicators, the normal
    equations for ``theta = (a, c)`` after demeaning within each unit are
    ``(D'D - sum_u C_u C_u' / n_u) theta = D'y - sum_u C_u S_u / n_u``, where ``C_u`` counts unit
    ``u``'s sales in each column, ``n_u`` is its number of sales and ``S_u`` the sum of their log
    prices. The minimum-norm solution is taken; normalisation fixes the remaining constants.
    """
    y = sales["log_price"].to_numpy()
    period, n_periods = _codes(sales["bucket_start"])
    ptype, n_types = _codes(sales["ptype"])
    unit, n_units = _codes(sales["unit"])
    k = n_periods + n_types
    counts = np.zeros((n_units, k))
    np.add.at(counts, (unit, period), 1.0)
    np.add.at(counts, (unit, n_periods + ptype), 1.0)
    n_u = np.bincount(unit, minlength=n_units).astype(float)
    sum_u = np.bincount(unit, weights=y, minlength=n_units)
    gram = np.zeros((k, k))
    np.add.at(gram, (period, period), 1.0)
    np.add.at(gram, (n_periods + ptype, n_periods + ptype), 1.0)
    np.add.at(gram, (period, n_periods + ptype), 1.0)
    np.add.at(gram, (n_periods + ptype, period), 1.0)
    rhs = np.bincount(period, weights=y, minlength=n_periods)
    rhs = np.concatenate([rhs, np.bincount(ptype, weights=y, minlength=n_types)])
    gram -= counts.T @ (counts / n_u[:, None])
    rhs -= counts.T @ (sum_u / n_u)
    theta = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    b_unit = (sum_u - counts @ theta) / n_u
    a = np.asarray(theta[period], dtype=np.float64)
    c = np.asarray(theta[n_periods + ptype], dtype=np.float64)
    b = np.asarray(b_unit[unit], dtype=np.float64)
    return a, b, c, np.asarray(y - a - b - c, dtype=np.float64)


def _holdout_levels(
    later: pl.DataFrame, units: pl.DataFrame, types: pl.DataFrame, config: AdjustmentConfig
) -> pl.DataFrame:
    """Hold-out levels with the training effects held fixed; unseen postcodes and types count 0."""
    adjusted = (
        later.join(units, on="unit", how="left")
        .join(types, on=["node", "ptype"], how="left")
        .with_columns(
            r=pl.col("log_price") - pl.col("b").fill_null(0.0) - pl.col("c").fill_null(0.0)
        )
        .with_columns(centre=pl.col("r").median().over("node", "bucket_start"))
        .filter((pl.col("r") - pl.col("centre")).abs() <= config.trim)
    )
    return adjusted.group_by("node", "bucket_start").agg(
        pl.col("r").mean().alias("a"), pl.len().cast(pl.UInt32).alias("n")
    )
