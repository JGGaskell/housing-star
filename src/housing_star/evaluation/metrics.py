"""Forecast-error metrics.

All functions ignore pairs where either value is NaN and return NaN if no pairs remain.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from housing_star._types import FloatArray
from housing_star.data.dataset import PriceSeriesDataset


def _errors(y_true: FloatArray, y_pred: FloatArray) -> tuple[FloatArray, FloatArray]:
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[ok], y_pred[ok]


def mae(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Mean absolute error."""
    t, p = _errors(y_true, y_pred)
    return float(np.mean(np.abs(p - t))) if t.size else float("nan")


def rmse(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Root mean squared error."""
    t, p = _errors(y_true, y_pred)
    return float(np.sqrt(np.mean((p - t) ** 2))) if t.size else float("nan")


def mape(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Mean absolute percentage error (%), over pairs with ``y_true != 0``."""
    t, p = _errors(y_true, y_pred)
    keep = t != 0
    return (
        float(np.mean(np.abs((p[keep] - t[keep]) / t[keep])) * 100.0)
        if keep.any()
        else float("nan")
    )


def smape(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Symmetric mean absolute percentage error (%), over pairs where ``|t| + |p| > 0``."""
    t, p = _errors(y_true, y_pred)
    scale = np.abs(t) + np.abs(p)
    keep = scale > 0
    return (
        float(np.mean(2.0 * np.abs(p[keep] - t[keep]) / scale[keep]) * 100.0)
        if keep.any()
        else float("nan")
    )


def score_holdout(
    ds: PriceSeriesDataset, y_pred: FloatArray, nodes: Sequence[int] | None = None
) -> dict[str, float]:
    """Score hold-out predictions for ``nodes`` (all nodes if ``None``).

    Parameters
    ----------
    y_pred : ``(N, n_eval)`` predictions in the dataset's log-demeaned levels.

    Returns
    -------
    ``log_rmse`` and ``log_mae`` (on log-demeaned values, approximately relative errors) and
    ``mae``, ``rmse`` (pounds), ``mape``, ``smape`` (%) on absolute prices.
    """
    rows = list(range(ds.N) if nodes is None else nodes)
    log_true, log_pred = ds.y_eval()[rows], y_pred[rows]
    abs_true, abs_pred = ds.to_abs(ds.y_eval())[rows], ds.to_abs(y_pred)[rows]
    return {
        "log_rmse": rmse(log_true, log_pred),
        "log_mae": mae(log_true, log_pred),
        "mae": mae(abs_true, abs_pred),
        "rmse": rmse(abs_true, abs_pred),
        "mape": mape(abs_true, abs_pred),
        "smape": smape(abs_true, abs_pred),
    }
