r"""Per-node autoregression, the natural no-spatial-information benchmark for STAR."""

from __future__ import annotations

from typing import Any, Self

import numpy as np

from housing_star._types import FloatArray
from housing_star.data.dataset import PriceSeriesDataset
from housing_star.models._growth import (
    fitted_growth,
    one_step_levels,
    path_levels,
    relative_growth,
)
from housing_star.models.base import TimeSeriesModel


class ArModel(TimeSeriesModel):
    r"""AR(p) on each node's growth, fitted independently for every node by least squares:

    .. math:: z_{i,t} = \sum_{\ell=1}^{p} \phi_{i,\ell}\, z_{i,t-\ell} + \varepsilon_{i,t},
        \qquad z_{i,t} = y_{i,t} - y_{i,t-1} - \bar g_i,

    where :math:`\bar g_i` is the node's average growth over the training window. There is no
    intercept (the average growth is already removed) and every forecast starts from the latest
    observed price. ``ArModel(p=1)`` is the classic AR(1) benchmark. There is nothing to
    calibrate: ``p`` is a modelling choice, not a tuned hyperparameter.
    """

    def __init__(self, p: int = 1) -> None:
        if p < 1:
            raise ValueError("p must be >= 1.")
        self.p = p
        self._coef: FloatArray | None = None  # (N, p): lags 1..p
        self._growth: FloatArray | None = None

    def fit(self, ds: PriceSeriesDataset) -> Self:
        """Estimate per-node coefficients on the training window."""
        z = relative_growth(ds.y_train(), ds.growth_means)
        if z.shape[1] <= self.p + 1:
            raise ValueError(f"Training window too short (T={z.shape[1] + 1}) for AR(p={self.p}).")
        self._coef = np.vstack([_ols(row, self.p) for row in z])
        self._growth = ds.growth_means.copy()
        self._fitted = True
        return self

    @property
    def coef_(self) -> FloatArray:
        """``(N, p)`` coefficients of lags ``1..p`` of growth."""
        self._check_fitted()
        assert self._coef is not None
        return self._coef

    @property
    def growth_(self) -> FloatArray:
        """``(N,)`` average growth per period the model was fitted with."""
        self._check_fitted()
        assert self._growth is not None
        return self._growth

    def predict(self, ds: PriceSeriesDataset) -> FloatArray:
        """One-step predictions from observed lags; the first ``p + 1`` columns are NaN."""
        coef, growth = self.coef_, fitted_growth(self.growth_, ds)
        z = relative_growth(ds.y, growth)
        z_hat = np.full(z.shape, np.nan)
        n = z.shape[1]
        if self.p < n:
            z_hat[:, self.p :] = sum(
                coef[:, ell - 1 : ell] * z[:, self.p - ell : n - ell]
                for ell in range(1, self.p + 1)
            )
        return one_step_levels(ds.y, growth, z_hat)

    def forecast(self, ds: PriceSeriesDataset, horizon: int) -> FloatArray:
        """Recursive forecast: each node's own growth forecasts are fed back as lags."""
        coef, growth = self.coef_, fitted_growth(self.growth_, ds)
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1; got {horizon}.")
        z = relative_growth(ds.y, growth)
        if self.p > z.shape[1]:
            raise ValueError(f"Need at least p+1={self.p + 1} observed periods; got {ds.T}.")
        lags = [z[:, -ell] for ell in range(1, self.p + 1)]
        z_path = np.empty((ds.N, horizon))
        for h in range(horizon):
            nxt = np.sum([coef[:, ell] * lags[ell] for ell in range(self.p)], axis=0)
            z_path[:, h] = nxt
            lags = [nxt, *lags[:-1]]
        return path_levels(ds.y[:, -1], growth, z_path)

    @property
    def params_(self) -> dict[str, Any]:
        return {"p": self.p, "coef": self.coef_.tolist()}


def _ols(series: FloatArray, p: int) -> FloatArray:
    """Least-squares ``[phi_1, ..., phi_p]`` (no intercept) for one growth series."""
    T = len(series)
    X = np.column_stack([series[p - ell : T - ell] for ell in range(1, p + 1)])
    return np.asarray(np.linalg.lstsq(X, series[p:], rcond=None)[0], dtype=np.float64)
