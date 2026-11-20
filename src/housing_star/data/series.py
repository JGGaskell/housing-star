"""Aggregate transactions into per-node price time series: medians, or a composition-adjusted index."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any, Literal, overload

import polars as pl

from housing_star._output import export
from housing_star.config import AdjustmentConfig, GraphConfig
from housing_star.data.adjusted import adjusted_levels
from housing_star.data.cache import TransactionCache
from housing_star.data.postcodes import infer_level, node_key_expr
from housing_star.data.schema import validate_panel
from housing_star.spatial.graph import select_neighbourhood

PriceMeasure = Literal["median", "adjusted"]
"""How a node's price for a period is measured; see :func:`panel_series`."""

PRICE_COLUMNS = {
    "postcode": pl.Utf8,
    "bucket_start": pl.Date,
    "price": pl.Float64,
    "n": pl.UInt32,
}


def node_series(
    node: str,
    *,
    start: str | date,
    end: str | date,
    freq: str,
    cache: TransactionCache,
) -> pl.DataFrame:
    """Median price and sale count per time bucket for one node.

    Only transactions whose postcode belongs to ``node`` *exactly* are kept, so ``"E1"`` never
    picks up ``"E14"`` sales even though the former is a string prefix of the latter.

    Parameters
    ----------
    node : district (``"GL5"``) or sector (``"GL6 8"``).
    freq : Polars truncation interval, e.g. ``"1y"``, ``"6mo"`` or ``"1q"``.
    """
    return (
        _node_transactions(node, start=start, end=end, freq=freq, cache=cache)
        .group_by("bucket_start")
        .agg(
            pl.col("pricePaid").median().alias("price"),
            pl.len().alias("n"),
        )
        .with_columns(pl.lit(node).alias("postcode"))
        .select(list(PRICE_COLUMNS))
        .sort("bucket_start")
    )


def _node_transactions(
    node: str, *, start: str | date, end: str | date, freq: str, cache: TransactionCache
) -> pl.DataFrame:
    """The validated sales of exactly ``node``, with the bucket each falls in."""
    key = node_key_expr("propertyAddress_postcode", infer_level(node))
    return (
        cache.transactions(node, start, end)
        .filter(key == node)
        .with_columns(pl.col("transactionDate").dt.truncate(freq).alias("bucket_start"))
    )


def _adjusted_series(
    nodes: Sequence[str],
    *,
    start: str | date,
    end: str | date,
    freq: str,
    cache: TransactionCache,
    holdout: int,
    adjustment: AdjustmentConfig | None,
) -> pl.DataFrame:
    """Composition-adjusted prices; the last ``holdout`` buckets do not inform the effects."""
    sales = pl.concat(
        [
            _node_transactions(node, start=start, end=end, freq=freq, cache=cache).select(
                pl.lit(node).alias("node"),
                "bucket_start",
                pl.col("propertyAddress_postcode").alias("unit"),
                pl.col("propertyType").alias("ptype"),
                pl.col("pricePaid").log().alias("log_price"),
            )
            for node in nodes
        ]
    )
    buckets = sorted(sales["bucket_start"].unique().to_list())
    holdout_from = buckets[max(len(buckets) - holdout, 0)] if holdout and buckets else None
    return adjusted_levels(sales, holdout_from=holdout_from, config=adjustment)


def panel_series(
    nodes: Sequence[str],
    *,
    start: str | date,
    end: str | date,
    freq: str,
    cache: TransactionCache | None = None,
    prices: PriceMeasure = "median",
    holdout: int = 0,
    adjustment: AdjustmentConfig | None = None,
) -> pl.DataFrame:
    """Long-format prices for ``nodes``: columns ``postcode, bucket_start, price, n``.

    Parameters
    ----------
    prices : ``"median"`` (the median sale price of each period) or ``"adjusted"`` (the
        composition-adjusted index of :mod:`housing_star.data.adjusted`, which holds the mix of
        streets and property types fixed).
    holdout : trailing periods reserved for evaluation. For ``"adjusted"`` their sales do not
        inform the postcode and type effects, so the hold-out never changes a training price.
        Medians are computed period by period and do not use it.
    adjustment : trimming and solver settings for ``"adjusted"``.

    The frame is checked with :func:`housing_star.data.schema.validate_panel` before it is
    returned. Those rows were aggregated from sales that already passed
    :class:`~housing_star.data.schema.TransactionSchema`. A bucket with no sales is absent.
    Pivoting the frame (:func:`~housing_star.data.prep.panel_from_long`) rejects a calendar
    that is not evenly spaced, so a missing period cannot become a one-column lag.
    """
    if prices not in ("median", "adjusted"):
        raise ValueError(f"prices must be 'median' or 'adjusted', not {prices!r}.")
    if holdout < 0:
        raise ValueError("holdout must be non-negative.")
    cache = cache or TransactionCache()
    # One planning pass for every node: missing sales are fetched in a few batched requests.
    cache.prefetch(nodes, start, end)
    if prices == "adjusted":
        frame = _adjusted_series(
            nodes,
            start=start,
            end=end,
            freq=freq,
            cache=cache,
            holdout=holdout,
            adjustment=adjustment,
        )
    else:
        frame = pl.concat(
            [node_series(n, start=start, end=end, freq=freq, cache=cache) for n in nodes]
        )
    return validate_panel(frame)


@overload
def download_plan(
    postcode: str,
    *,
    start: str | date,
    end: str | date,
    graph: GraphConfig | None = ...,
    cache: TransactionCache | None = ...,
    as_pandas: Literal[False] = ...,
) -> pl.DataFrame: ...
@overload
def download_plan(
    postcode: str,
    *,
    start: str | date,
    end: str | date,
    graph: GraphConfig | None = ...,
    cache: TransactionCache | None = ...,
    as_pandas: bool,
) -> Any: ...
def download_plan(
    postcode: str,
    *,
    start: str | date,
    end: str | date,
    graph: GraphConfig | None = None,
    cache: TransactionCache | None = None,
    as_pandas: bool = False,
) -> Any:
    """What building a dataset would download, without downloading anything.

    Returns one row per node of ``postcode``'s neighbourhood: ``node``, ``fully_cached`` and, when
    not cached, the ``missing_from`` / ``missing_to`` dates that still have to be fetched. If every
    node is fully cached, building the dataset is fast and needs no network access.
    """
    graph = graph or GraphConfig()
    cache = cache or TransactionCache()
    hood = select_neighbourhood(postcode, radius_m=graph.radius_m, max_nodes=graph.max_nodes)
    rows = []
    for node in hood.keys:
        gaps = cache.missing(node, start, end)
        rows.append(
            {
                "node": node,
                "fully_cached": not gaps,
                "missing_from": gaps[0][0] if gaps else None,
                "missing_to": gaps[-1][1] if gaps else None,
            }
        )
    schema = {
        "node": pl.Utf8,
        "fully_cached": pl.Boolean,
        "missing_from": pl.Date,
        "missing_to": pl.Date,
    }
    return export(pl.DataFrame(rows, schema=schema), as_pandas)
