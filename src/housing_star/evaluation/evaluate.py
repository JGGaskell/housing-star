"""Hold-out evaluation and cross-validation of several models on one dataset."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Literal, overload

import polars as pl

from housing_star._output import export
from housing_star.data.dataset import PriceSeriesDataset
from housing_star.evaluation.cv import WalkForwardCV
from housing_star.evaluation.metrics import score_holdout
from housing_star.evaluation.protocols import Protocol, holdout_predictions
from housing_star.models.base import TimeSeriesModel
from housing_star.models.star.model import UnstableModelWarning

logger = logging.getLogger(__name__)


def _nodes(ds: PriceSeriesDataset) -> Sequence[int] | None:
    """Score the focal node when there is one, otherwise every node."""
    return None if ds.target is None else [ds.target_idx]


@overload
def evaluate(
    ds: PriceSeriesDataset,
    models: Mapping[str, TimeSeriesModel],
    *,
    protocol: Protocol = ...,
    as_pandas: Literal[False] = ...,
) -> pl.DataFrame: ...
@overload
def evaluate(
    ds: PriceSeriesDataset,
    models: Mapping[str, TimeSeriesModel],
    *,
    protocol: Protocol = ...,
    as_pandas: bool,
) -> Any: ...
def evaluate(
    ds: PriceSeriesDataset,
    models: Mapping[str, TimeSeriesModel],
    *,
    protocol: Protocol = "one_step",
    as_pandas: bool = False,
) -> Any:
    """Fit every model on the training window and score the hold-out.

    Returns one row per model: ``model, protocol, n_eval`` and the metrics of
    :func:`~housing_star.evaluation.metrics.score_holdout` (for the target node, if the dataset
    has one). A polars DataFrame, or a pandas one if ``as_pandas`` is true.
    """
    rows = [
        {
            "model": name,
            "protocol": protocol,
            "n_eval": ds.n_eval,
            **score_holdout(ds, holdout_predictions(model, ds, protocol), _nodes(ds)),
        }
        for name, model in models.items()
    ]
    return export(pl.DataFrame(rows), as_pandas)


@overload
def target_predictions(
    ds: PriceSeriesDataset,
    models: Mapping[str, TimeSeriesModel],
    *,
    protocol: Protocol = ...,
    as_pandas: Literal[False] = ...,
) -> pl.DataFrame: ...
@overload
def target_predictions(
    ds: PriceSeriesDataset,
    models: Mapping[str, TimeSeriesModel],
    *,
    protocol: Protocol = ...,
    as_pandas: bool,
) -> Any: ...
def target_predictions(
    ds: PriceSeriesDataset,
    models: Mapping[str, TimeSeriesModel],
    *,
    protocol: Protocol = "one_step",
    as_pandas: bool = False,
) -> Any:
    """Actual and predicted prices (£) of the target node over the hold-out, one row per model-period."""
    idx = ds.target_idx
    actual = ds.to_abs(ds.y_eval())[idx]
    frames = [
        pl.DataFrame(
            {
                "model": name,
                "time": ds.times[ds.train_end :],
                "actual": actual,
                "predicted": ds.to_abs(holdout_predictions(model, ds, protocol))[idx],
            }
        )
        for name, model in models.items()
    ]
    return export(pl.concat(frames), as_pandas)


@overload
def cross_validate(
    models: Mapping[str, TimeSeriesModel],
    ds: PriceSeriesDataset,
    cv: WalkForwardCV | None = ...,
    *,
    metric: str = ...,
    protocol: Protocol = ...,
    as_pandas: Literal[False] = ...,
) -> pl.DataFrame: ...
@overload
def cross_validate(
    models: Mapping[str, TimeSeriesModel],
    ds: PriceSeriesDataset,
    cv: WalkForwardCV | None = ...,
    *,
    metric: str = ...,
    protocol: Protocol = ...,
    as_pandas: bool,
) -> Any: ...
def cross_validate(
    models: Mapping[str, TimeSeriesModel],
    ds: PriceSeriesDataset,
    cv: WalkForwardCV | None = None,
    *,
    metric: str = "log_rmse",
    protocol: Protocol = "one_step",
    as_pandas: bool = False,
) -> Any:
    """Score every model on the same walk-forward folds of the *training* window.

    ``cv`` defaults to ``WalkForwardCV()``: the last three training periods, one per fold.

    Each fold re-fits the model on periods before its origin (and re-demeans on them), so the
    outer hold-out is never touched. Instability warnings are suppressed inside the folds. A fold a model cannot be fitted on (for example because
    ``p`` is too large for the short window) scores NaN.

    Returns one row per ``(model, fold)``: ``model, fold, train_end, eval_end, score``.
    """
    train = ds.train_only()
    try:
        folds = list(WalkForwardCV().split(train.T) if cv is None else cv.split(train.T))
    except ValueError as exc:
        raise ValueError(f"{exc} (This dataset holds out {ds.n_eval} of {ds.T} periods.)") from exc
    rows = []
    for fold in folds:
        view = train.window(fold.train_end, fold.eval_end)
        for name, model in models.items():
            try:
                with warnings.catch_warnings():
                    # Candidates fitted on a handful of periods are often unstable; they simply
                    # score badly, so only a fit you keep should warn.
                    warnings.simplefilter("ignore", UnstableModelWarning)
                    preds = holdout_predictions(model, view, protocol)
                score = score_holdout(view, preds, _nodes(view))[metric]
            except ValueError as exc:
                logger.debug("%s failed on fold %d: %s", name, fold.index, exc)
                score = float("nan")
            rows.append(
                {
                    "model": name,
                    "fold": fold.index,
                    "train_end": fold.train_end,
                    "eval_end": fold.eval_end,
                    "score": score,
                }
            )
    return export(pl.DataFrame(rows, schema_overrides={"score": pl.Float64}), as_pandas)
