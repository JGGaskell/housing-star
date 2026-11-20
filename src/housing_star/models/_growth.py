r"""Conversions between log price levels and growth relative to each node's average growth.

With :math:`y_{i,t}` the log price and :math:`\bar g_i` the node's average growth per period over
the training window, the models describe the *relative growth*

.. math:: z_{i,t} = y_{i,t} - y_{i,t-1} - \bar g_i,

and every prediction is rebuilt on the latest observed price, :math:`\hat y_t = y_{t-1} + \bar g +
\hat z_t`. A model that predicts :math:`\hat z = 0` is the drift benchmark.
"""

from __future__ import annotations

import numpy as np

from housing_star._types import FloatArray
from housing_star.data.dataset import PriceSeriesDataset


def relative_growth(y: FloatArray, growth: FloatArray) -> FloatArray:
    """``(N, T-1)`` growth of each node net of its average: ``z[:, k] = y[:, k+1] - y[:, k] - g``."""
    return np.asarray(np.diff(y, axis=1) - growth[:, None], dtype=np.float64)


def one_step_levels(y: FloatArray, growth: FloatArray, z_hat: FloatArray) -> FloatArray:
    """``(N, T)`` one-step predictions of ``y`` from predicted relative growth ``(N, T-1)``.

    Column ``t`` is the observed ``y[:, t-1]`` plus average growth plus ``z_hat``; column 0 has
    no previous value and is NaN, as is any column whose ``z_hat`` is NaN.
    """
    out = np.full(y.shape, np.nan)
    out[:, 1:] = y[:, :-1] + growth[:, None] + z_hat
    return out


def path_levels(y_last: FloatArray, growth: FloatArray, z_path: FloatArray) -> FloatArray:
    """``(N, H)`` levels reached by growing from ``y_last`` at ``growth + z`` each period."""
    return np.asarray(y_last[:, None] + np.cumsum(growth[:, None] + z_path, axis=1))


def relative_growth_of_path(y_last: float, growth: float, path: FloatArray) -> FloatArray:
    """``(H,)`` relative growth that takes a node from ``y_last`` along the level ``path``."""
    return np.asarray(np.diff(np.concatenate([[y_last], path])) - growth)


def fitted_growth(growth: FloatArray, ds: PriceSeriesDataset) -> FloatArray:
    """The growth rates a model was fitted with, checked against ``ds``'s node count."""
    if growth.shape[0] != ds.N:
        raise ValueError(f"Model was fitted on {growth.shape[0]} nodes but the dataset has {ds.N}.")
    return growth
