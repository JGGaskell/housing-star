"""``StarModel``: spatial-temporal autoregression on growth."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, Self

from housing_star._types import FloatArray
from housing_star.data.dataset import PriceSeriesDataset
from housing_star.models._growth import fitted_growth, relative_growth
from housing_star.models.base import TimeSeriesModel
from housing_star.models.star.estimation import StarFit, fit_star, persistence
from housing_star.models.star.prediction import forecast, one_step_predictions


class UnstableModelWarning(UserWarning):
    """A fitted STAR model whose shocks to growth would persist or grow instead of fading."""


def warn_if_unstable(model: StarModel, W: FloatArray) -> None:
    """Emit :class:`UnstableModelWarning` when ``model.persistence(W) >= 1``."""
    value = model.persistence(W)
    if value >= 1.0:
        warnings.warn(
            f"STAR persistence is {value:.3f} (>= 1): shocks to growth do not fade, so forecasts "
            "and scenario effects will keep building. Treat them with caution.",
            UnstableModelWarning,
            stacklevel=3,
        )


class StarModel(TimeSeriesModel):
    r"""STAR(p): each node's growth depends on its own and its neighbours' recent growth.

    Growth is measured relative to the node's average growth over the training window (see
    :mod:`housing_star.models.star.estimation` for the equations), so every forecast starts from
    the latest observed price.

    Parameters
    ----------
    p : number of autoregressive lags.
    rho_bounds : search interval for the spatial parameter :math:`\rho`.
    grid_size : points in the initial coarse search over :math:`\rho` (refined afterwards).
    """

    def __init__(
        self,
        p: int = 1,
        *,
        rho_bounds: tuple[float, float] = (-0.95, 0.95),
        grid_size: int = 81,
    ) -> None:
        self.p = p
        self.rho_bounds = rho_bounds
        self.grid_size = grid_size
        self._fit: StarFit | None = None
        self._growth: FloatArray | None = None

    @classmethod
    def from_fit(cls, fit: StarFit, growth: FloatArray) -> Self:
        """A fitted model with known parameters (e.g. for what-if analysis or loading results).

        ``growth`` is each node's average growth per period (``ds.growth_means``).
        """
        model = cls(p=len(fit.phi))
        model._fit = fit
        model._growth = growth
        model._fitted = True
        return model

    def fit(self, ds: PriceSeriesDataset) -> Self:
        """Estimate ``rho, phi, theta, sigma2`` on the training window.

        Warns (:class:`UnstableModelWarning`) when the fitted dynamics do not die out.
        """
        z = relative_growth(ds.y_train(), ds.growth_means)
        self._fit = fit_star(z, ds.W, self.p, rho_bounds=self.rho_bounds, grid_size=self.grid_size)
        self._growth = ds.growth_means.copy()
        self._fitted = True
        warn_if_unstable(self, ds.W)
        return self

    @property
    def fit_(self) -> StarFit:
        """The estimated parameters."""
        self._check_fitted()
        assert self._fit is not None
        return self._fit

    @property
    def growth_(self) -> FloatArray:
        """``(N,)`` average growth per period the model was fitted with."""
        self._check_fitted()
        assert self._growth is not None
        return self._growth

    def persistence(self, W: FloatArray) -> float:
        """Spectral radius of the fitted dynamics under weights ``W``; 1 or more is unstable."""
        return persistence(self.fit_, W)

    def predict(self, ds: PriceSeriesDataset) -> FloatArray:
        """One-step-ahead predictions using observed previous prices and growth."""
        return one_step_predictions(ds.y, ds.W, self.fit_, fitted_growth(self.growth_, ds))

    def forecast(
        self,
        ds: PriceSeriesDataset,
        horizon: int,
        *,
        clamp: Mapping[int, FloatArray] | None = None,
        impulse: Mapping[int, FloatArray] | None = None,
    ) -> FloatArray:
        """Recursive forecast; ``clamp``/``impulse`` apply scenario shocks (see ``prediction``)."""
        growth = fitted_growth(self.growth_, ds)
        return forecast(ds.y, ds.W, self.fit_, growth, horizon, clamp=clamp, impulse=impulse)

    @property
    def params_(self) -> dict[str, Any]:
        f = self.fit_
        return {
            "p": self.p,
            "rho": f.rho,
            "phi": f.phi.tolist(),
            "theta": f.theta.tolist(),
            "sigma2": f.sigma2,
            "loglik": f.loglik,
        }
