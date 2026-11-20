"""Common interface of all time-series models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, Self, overload

import numpy as np
import polars as pl

from housing_star._output import export
from housing_star._types import FloatArray
from housing_star.data.dataset import PriceSeriesDataset


class TimeSeriesModel(ABC):
    """scikit-learn style model: ``fit`` on a dataset's training window, then ``predict``/``forecast``.

    Models describe each node's growth relative to its training-window average growth, but all
    arrays they return are log price levels in the dataset's demeaned space (``ds.y``); use
    :meth:`predict_abs` for pounds.
    """

    _fitted: bool = False

    @abstractmethod
    def fit(self, ds: PriceSeriesDataset) -> Self:
        """Estimate parameters from ``ds.y_train()`` only and return ``self``."""

    @abstractmethod
    def predict(self, ds: PriceSeriesDataset) -> FloatArray:
        """One-step-ahead predictions of ``ds.y``, shape ``(N, T)``; entries without enough lags are NaN.

        Each prediction uses *observed* lags, so columns from ``ds.train_end`` onwards are
        genuine out-of-sample one-step forecasts.
        """

    @abstractmethod
    def forecast(self, ds: PriceSeriesDataset, horizon: int) -> FloatArray:
        """Recursive ``(N, horizon)`` forecast from the last column of ``ds``.

        Forecasts are fed back as lags; no hold-out data is used. Pass ``ds.train_only()`` to
        forecast the hold-out period.
        """

    @property
    @abstractmethod
    def params_(self) -> dict[str, Any]:
        """Fitted parameters as plain Python values."""

    def predict_abs(self, ds: PriceSeriesDataset) -> FloatArray:
        """One-step predictions in pounds."""
        return ds.to_abs(self.predict(ds))

    @overload
    def predict_frame(
        self, ds: PriceSeriesDataset, as_pandas: Literal[False] = False
    ) -> pl.DataFrame: ...
    @overload
    def predict_frame(self, ds: PriceSeriesDataset, as_pandas: bool) -> Any: ...
    def predict_frame(self, ds: PriceSeriesDataset, as_pandas: bool = False) -> Any:
        """One-step predictions next to actual prices, one row per ``(postcode, time)``.

        Columns: ``postcode, time, actual, predicted`` (pounds; null where a lag is missing) and
        ``is_holdout``.
        """
        frame = ds.to_frame().select(
            "postcode", "time", pl.col("price").alias("actual"), "is_holdout"
        )
        predicted = self.predict_abs(ds).ravel()
        frame = frame.with_columns(pl.Series("predicted", predicted).fill_nan(None)).select(
            "postcode", "time", "actual", "predicted", "is_holdout"
        )
        return export(frame, as_pandas)

    @overload
    def forecast_frame(
        self, ds: PriceSeriesDataset, horizon: int, as_pandas: Literal[False] = False
    ) -> pl.DataFrame: ...
    @overload
    def forecast_frame(self, ds: PriceSeriesDataset, horizon: int, as_pandas: bool) -> Any: ...
    def forecast_frame(self, ds: PriceSeriesDataset, horizon: int, as_pandas: bool = False) -> Any:
        """Recursive forecast in pounds, one row per ``(postcode, step)`` (step 1 = next period)."""
        prices = ds.to_abs(self.forecast(ds, horizon))
        frame = pl.DataFrame(
            {
                "postcode": np.repeat(ds.postcodes, horizon),
                "step": np.tile(np.arange(1, horizon + 1), ds.N),
                "forecast": prices.ravel(),
            }
        )
        return export(frame, as_pandas)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} is not fitted; call .fit(dataset) first.")
