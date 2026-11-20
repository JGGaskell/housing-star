r"""Per-node random walk with drift: the naive "average growth from the latest price" benchmark."""

from __future__ import annotations

from typing import Any, Self

import numpy as np

from housing_star._types import FloatArray
from housing_star.data.dataset import PriceSeriesDataset
from housing_star.models._growth import fitted_growth, one_step_levels, path_levels
from housing_star.models.base import TimeSeriesModel


class DriftModel(TimeSeriesModel):
    r"""Each node grows at its average training-window rate, starting from its latest price:

    .. math:: \hat y_{i,t} = y_{i,t-1} + \bar g_i, \qquad \bar g_i = \text{average growth of node } i.

    A one-step prediction adds one period of growth to the observed previous value; a recursive
    forecast adds ``h`` periods of growth to the last observed value. This is STAR and AR with
    every dynamic coefficient set to zero, so it is the benchmark they must beat.
    """

    def __init__(self) -> None:
        self._growth: FloatArray | None = None  # (N,) average log growth per period

    def fit(self, ds: PriceSeriesDataset) -> Self:
        """Take each node's average growth per period from the training window."""
        if ds.train_end < 2:
            raise ValueError("A drift model needs at least 2 training periods.")
        self._growth = ds.growth_means.copy()
        self._fitted = True
        return self

    @property
    def growth_(self) -> FloatArray:
        """``(N,)`` average log growth per period (``0.05`` is about 5% a period)."""
        self._check_fitted()
        assert self._growth is not None
        return self._growth

    def predict(self, ds: PriceSeriesDataset) -> FloatArray:
        """Observed previous value plus one period of growth; the first column is NaN."""
        growth = fitted_growth(self.growth_, ds)
        return one_step_levels(ds.y, growth, np.zeros((ds.N, ds.T - 1)))

    def forecast(self, ds: PriceSeriesDataset, horizon: int) -> FloatArray:
        """Last observed value plus ``1..horizon`` periods of growth."""
        growth = fitted_growth(self.growth_, ds)
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1; got {horizon}.")
        return path_levels(ds.y[:, -1], growth, np.zeros((ds.N, horizon)))

    @property
    def params_(self) -> dict[str, Any]:
        return {"growth": self.growth_.tolist()}


class NoChangeModel(TimeSeriesModel):
    r"""The latest observed price carried forward, with no growth: the naive benchmark.

    .. math:: \hat y_{i,t} = y_{i,t-1}.

    A one-step prediction is the observed previous value; a recursive forecast repeats the last
    observed value at every horizon. This is :class:`DriftModel` with the average growth set to
    zero, so it shows how much of a growth model's accuracy depends on that average.
    """

    def __init__(self) -> None:
        self._n_nodes: int | None = None

    def fit(self, ds: PriceSeriesDataset) -> Self:
        """Record the node count; there is nothing to estimate."""
        self._n_nodes = ds.N
        self._fitted = True
        return self

    def _zero_growth(self, ds: PriceSeriesDataset) -> FloatArray:
        self._check_fitted()
        assert self._n_nodes is not None
        return fitted_growth(np.zeros(self._n_nodes), ds)

    def predict(self, ds: PriceSeriesDataset) -> FloatArray:
        """The observed previous value; the first column is NaN."""
        growth = self._zero_growth(ds)
        return one_step_levels(ds.y, growth, np.zeros((ds.N, ds.T - 1)))

    def forecast(self, ds: PriceSeriesDataset, horizon: int) -> FloatArray:
        """The last observed value, repeated ``horizon`` times."""
        growth = self._zero_growth(ds)
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1; got {horizon}.")
        return path_levels(ds.y[:, -1], growth, np.zeros((ds.N, horizon)))

    @property
    def params_(self) -> dict[str, Any]:
        self._check_fitted()
        return {}
