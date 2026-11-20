"""Leak-free cleaning of a node-by-period price panel.

Every step that looks across time (coverage, gap filling, merging weights) uses only the first
``train_end`` periods, so the hold-out can never influence which nodes survive or how the training
data are filled. The one exception is documented on :func:`prepare_panel`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

from housing_star._types import FloatArray, IntArray
from housing_star.config import CleaningConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Panel:
    """Node-by-period prices and sale counts (NaN marks a missing observation)."""

    postcodes: list[str]
    eastings: FloatArray
    northings: FloatArray
    prices: FloatArray
    counts: FloatArray
    times: list[str]

    def take(self, rows: list[int] | IntArray) -> Panel:
        """The sub-panel containing ``rows``."""
        idx = np.asarray(rows, dtype=int)
        return Panel(
            [self.postcodes[i] for i in idx],
            self.eastings[idx],
            self.northings[idx],
            self.prices[idx],
            self.counts[idx],
            self.times,
        )


def _month_step(earlier: date, later: date) -> int:
    """Whole months from ``earlier`` to ``later`` when both fall on the same day, else 0."""
    if earlier.day != later.day:
        return 0
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


def _require_regular_calendar(times: list[date]) -> None:
    """Reject dates that are not one fixed, positive number of months apart.

    The panel's columns are these dates, and a lag of one column is one period. A bucket that
    no node observed is absent. Three or more dates make that hole a longer gap, which is
    rejected. Two dates have a single step, so a hole between them cannot be told from a
    coarser frequency. ``times`` is sorted and unique.
    """
    if len(times) < 2:
        return
    steps = [_month_step(times[i], times[i + 1]) for i in range(len(times) - 1)]
    shortest, longest = min(steps), max(steps)
    if shortest <= 0:
        broken = steps.index(shortest)
        left, right = times[broken], times[broken + 1]
        raise ValueError(
            f"The price periods are not a regular calendar: {right.isoformat()} is not a whole "
            f"number of months after {left.isoformat()} on the same day of the month. Growth and "
            "lags need equal steps on one grid, such as every 1 January or every quarter start."
        )
    if longest != shortest:
        skipped = steps.index(longest)
        left, right = times[skipped], times[skipped + 1]
        raise ValueError(
            f"The price periods are not evenly spaced: {left.isoformat()} to {right.isoformat()} "
            f"is {longest} months, and the shortest step is {shortest} months. A period with no "
            "sales in any node is omitted, so growth and lags would treat that gap as one period. "
            "Use a window or frequency in which every period has a sale in at least one node."
        )


def panel_from_long(
    prices: pl.DataFrame, nodes: list[str], eastings: FloatArray, northings: FloatArray
) -> Panel:
    """Pivot long prices (``postcode, bucket_start, price, n``) onto the node list.

    The columns are the observed bucket dates. Those dates have to lie on one grid: the same
    day of the month, a fixed positive number of months apart. A date no node observed is not
    a column. That is a missing period, not a gap in a node, and it is rejected so growth and
    lags cannot treat it as one step.

    Raises
    ------
    ValueError
        If ``prices`` is empty, or the bucket dates are not a regular calendar.
    """
    if prices.is_empty():
        raise ValueError(
            "No transactions found for any node in the requested window. Land Registry data covers "
            "England and Wales only; check the postcode and the date range."
        )
    times = sorted(prices["bucket_start"].unique().to_list())
    _require_regular_calendar(times)
    row = {node: i for i, node in enumerate(nodes)}
    col = {t: j for j, t in enumerate(times)}
    values = np.full((len(nodes), len(times)), np.nan)
    counts = np.full_like(values, np.nan)
    for node, bucket, price, n in prices.select("postcode", "bucket_start", "price", "n").rows():
        values[row[node], col[bucket]] = price
        counts[row[node], col[bucket]] = n
    return Panel(list(nodes), eastings, northings, values, counts, [t.isoformat() for t in times])


def _training_stats(panel: Panel, train_end: int) -> tuple[FloatArray, FloatArray]:
    """Per-node coverage and mean sale count over the training window."""
    seen = np.isfinite(panel.prices[:, :train_end]) & np.isfinite(panel.counts[:, :train_end])
    n_seen = seen.sum(axis=1)
    totals = np.where(seen, panel.counts[:, :train_end], 0.0).sum(axis=1)
    return n_seen / train_end, totals / np.maximum(n_seen, 1)


def _merge(panel: Panel, a: int, b: int, train_end: int) -> Panel:
    """Merge nodes ``a`` and ``b``: count-weighted prices, training-count-weighted centroid."""
    na = np.nan_to_num(panel.counts[[a, b]])
    pa = np.nan_to_num(panel.prices[[a, b]])
    total = na.sum(axis=0)
    safe = np.where(total > 0, total, 1.0)
    price = np.where(total > 0, (na * pa).sum(axis=0) / safe, np.nan)
    count = np.where(total > 0, total, np.nan)

    weights = np.nan_to_num(panel.counts[[a, b], :train_end]).sum(axis=1)
    weights = weights if weights.sum() > 0 else np.ones(2)
    easting = float(np.average(panel.eastings[[a, b]], weights=weights))
    northing = float(np.average(panel.northings[[a, b]], weights=weights))

    keep = [i for i in range(len(panel.postcodes)) if i not in (a, b)]
    rest = panel.take(keep)
    return Panel(
        [*rest.postcodes, f"{panel.postcodes[a]}+{panel.postcodes[b]}"],
        np.append(rest.eastings, easting),
        np.append(rest.northings, northing),
        np.vstack([rest.prices, price]),
        np.vstack([rest.counts, count]),
        panel.times,
    )


def merge_sparse_nodes(panel: Panel, cleaning: CleaningConfig, train_end: int) -> Panel:
    """Merge the thinnest node into its nearest neighbour until all pass or ``min_nodes`` remain."""
    while len(panel.postcodes) > cleaning.min_nodes:
        coverage, mean_n = _training_stats(panel, train_end)
        thin = np.flatnonzero((coverage < cleaning.min_coverage) | (mean_n < cleaning.min_mean_n))
        if thin.size == 0:
            break
        worst = int(thin[np.lexsort((mean_n[thin], coverage[thin]))[0]])
        dist = np.hypot(
            panel.eastings - panel.eastings[worst], panel.northings - panel.northings[worst]
        )
        dist[worst] = np.inf
        nearest = int(np.argmin(dist))
        logger.debug("Merging %s into %s", panel.postcodes[worst], panel.postcodes[nearest])
        panel = _merge(panel, worst, nearest, train_end)
    return panel


def _longest_run(mask: FloatArray) -> int:
    run = best = 0
    for flag in mask:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def drop_thin_nodes(panel: Panel, cleaning: CleaningConfig, train_end: int) -> Panel:
    """Drop nodes failing coverage, mean-count or maximum-gap thresholds (training window only)."""
    coverage, mean_n = _training_stats(panel, train_end)
    missing = ~np.isfinite(panel.prices[:, :train_end])
    keep = [
        i
        for i in range(len(panel.postcodes))
        if coverage[i] >= cleaning.min_coverage
        and mean_n[i] >= cleaning.min_mean_n
        and _longest_run(missing[i]) <= cleaning.max_gap
    ]
    if not keep:
        raise ValueError(
            "All nodes were dropped by the quality thresholds; relax min_coverage, min_mean_n or max_gap."
        )
    return panel.take(keep)


def _fill_row(row: FloatArray, train_end: int) -> tuple[FloatArray, bool]:
    """Linearly interpolate interior gaps and extrapolate edges of the training window.

    Returns the filled training window and whether any edge was extrapolated. Rows with no
    training observation are returned unchanged.
    """
    head = row[:train_end]
    seen = np.flatnonzero(np.isfinite(head))
    if seen.size == 0 or seen.size == train_end:
        return head, False
    first, last = int(seen[0]), int(seen[-1])
    x = np.arange(train_end, dtype=float)
    left = right = 0.0
    if seen.size > 1:
        left = (head[seen[1]] - head[seen[0]]) / (seen[1] - seen[0])
        right = (head[seen[-1]] - head[seen[-2]]) / (seen[-1] - seen[-2])
    filled = head.copy()
    filled[first : last + 1] = np.interp(x[first : last + 1], seen, head[seen])
    filled[:first] = head[first] + left * (x[:first] - first)
    filled[last + 1 :] = head[last] + right * (x[last + 1 :] - last)
    return filled, first > 0 or last < train_end - 1


def fill_training_gaps(panel: Panel, cleaning: CleaningConfig, train_end: int) -> Panel:
    """Fill gaps in the training window; hold-out columns are never touched.

    Nodes that needed edge extrapolation are blanked (and later dropped) unless
    ``cleaning.keep_extrapolated`` is set.
    """
    prices = panel.prices.copy()
    for i in range(prices.shape[0]):
        filled, extrapolated = _fill_row(prices[i], train_end)
        prices[i, :train_end] = filled
        if extrapolated and not cleaning.keep_extrapolated:
            prices[i, :] = np.nan
    return Panel(
        panel.postcodes, panel.eastings, panel.northings, prices, panel.counts, panel.times
    )


def prepare_panel(panel: Panel, cleaning: CleaningConfig, holdout: int) -> Panel:
    """Merge, filter and fill ``panel`` using the first ``T - holdout`` periods only.

    Finally, nodes with any missing value (including in the hold-out) are dropped so the result is
    a complete matrix. Hold-out *values* never affect cleaning; hold-out *availability* can decide
    whether a node survives this last step.

    Raises
    ------
    ValueError
        If ``holdout`` leaves no training data, every node is dropped, or a training price is
        non-positive.
    """
    T = panel.prices.shape[1]
    train_end = T - holdout
    if holdout < 0 or train_end <= 0:
        raise ValueError(f"holdout={holdout} is invalid for a panel with {T} periods.")
    panel = merge_sparse_nodes(panel, cleaning, train_end)
    panel = drop_thin_nodes(panel, cleaning, train_end)
    panel = fill_training_gaps(panel, cleaning, train_end)
    complete = np.flatnonzero(np.isfinite(panel.prices).all(axis=1))
    if complete.size == 0:
        raise ValueError("All nodes were dropped because they have missing observations.")
    panel = panel.take(complete)
    if (panel.prices[:, :train_end] <= 0).any():
        raise ValueError("Non-positive training prices: the log transform needs positive values.")
    return panel
