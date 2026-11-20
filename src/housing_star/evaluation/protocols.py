"""How a fitted model is asked to predict a hold-out window.

``one_step``
    Each hold-out period is predicted from the *observed* previous periods ("given what we saw
    last year, what happens next?").
``recursive``
    The model sees only the training window and forecasts the whole hold-out by feeding its own
    forecasts back as lags. This is the protocol that matches multi-year scenario forecasts.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Literal

from housing_star._types import FloatArray
from housing_star.data.dataset import PriceSeriesDataset
from housing_star.models.base import TimeSeriesModel

Protocol = Literal["one_step", "recursive"]


def holdout_predictions(
    model: TimeSeriesModel, ds: PriceSeriesDataset, protocol: Protocol = "one_step"
) -> FloatArray:
    """Fit a *copy* of ``model`` on ``ds``'s training window; return ``(N, n_eval)`` predictions."""
    if protocol not in ("one_step", "recursive"):
        raise ValueError(f"protocol must be 'one_step' or 'recursive'; got {protocol!r}.")
    if ds.n_eval < 1:
        raise ValueError("The dataset has no hold-out periods to predict.")
    fitted = deepcopy(model).fit(ds)
    if protocol == "one_step":
        return fitted.predict(ds)[:, ds.train_end :]
    return fitted.forecast(ds.train_only(), ds.n_eval)
