"""Hyperparameter selection by walk-forward grid search."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, overload

import polars as pl

from housing_star._output import export
from housing_star.data.dataset import PriceSeriesDataset
from housing_star.evaluation.cv import WalkForwardCV
from housing_star.evaluation.evaluate import cross_validate
from housing_star.evaluation.protocols import Protocol
from housing_star.models.base import TimeSeriesModel

Frame = TypeVar("Frame")


@dataclass(frozen=True)
class TuningResult(Generic[Frame]):
    """Outcome of :func:`tune`.

    Attributes
    ----------
    best_params : the candidate with the lowest mean fold score.
    scores : one row per ``(candidate, fold)``: the hyperparameter columns, ``fold``, ``score``.
    summary : one row per candidate: hyperparameters, ``mean_score`` and ``n_valid_folds``,
        sorted best first.
    """

    best_params: dict[str, Any]
    scores: Frame
    summary: Frame


@overload
def tune(
    model_cls: type[TimeSeriesModel],
    ds: PriceSeriesDataset,
    grid: Mapping[str, Sequence[Any]],
    cv: WalkForwardCV | None = ...,
    *,
    metric: str = ...,
    protocol: Protocol = ...,
    as_pandas: Literal[False] = ...,
) -> TuningResult[pl.DataFrame]: ...
@overload
def tune(
    model_cls: type[TimeSeriesModel],
    ds: PriceSeriesDataset,
    grid: Mapping[str, Sequence[Any]],
    cv: WalkForwardCV | None = ...,
    *,
    metric: str = ...,
    protocol: Protocol = ...,
    as_pandas: bool,
) -> TuningResult[Any]: ...
def tune(
    model_cls: type[TimeSeriesModel],
    ds: PriceSeriesDataset,
    grid: Mapping[str, Sequence[Any]],
    cv: WalkForwardCV | None = None,
    *,
    metric: str = "log_rmse",
    protocol: Protocol = "one_step",
    as_pandas: bool = False,
) -> TuningResult[Any]:
    """Pick hyperparameters by walk-forward cross-validation on the training window.

    Every combination in ``grid`` is scored on the same folds (see
    :func:`~housing_star.evaluation.evaluate.cross_validate`). Only ``ds``'s training window is
    used, so the hold-out stays untouched.

    Parameters
    ----------
    grid : ``{"p": [1, 2, 3]}``-style mapping; all combinations are tried.
    cv : the folds; by default the last three periods of the training window (see
        :class:`~housing_star.evaluation.cv.WalkForwardCV`). The size of the training window is
        set once, by ``holdout`` when the dataset is built.
    as_pandas : return ``scores`` and ``summary`` as pandas DataFrames instead of polars.

    Raises
    ------
    ValueError
        If no candidate can be scored on any fold.
    """
    names = list(grid)
    candidates = [
        dict(zip(names, values, strict=True)) for values in itertools.product(*grid.values())
    ]
    models = {str(i): model_cls(**params) for i, params in enumerate(candidates)}
    folds = cross_validate(models, ds, cv, metric=metric, protocol=protocol)

    lookup = pl.DataFrame([{"model": str(i), **params} for i, params in enumerate(candidates)])
    scores = folds.join(lookup, on="model").drop("model", "train_end", "eval_end")
    scores = scores.select(*names, "fold", "score")
    summary = (
        scores.group_by(names, maintain_order=True)
        .agg(
            pl.col("score").drop_nans().mean().alias("mean_score"),
            pl.col("score").is_not_nan().sum().alias("n_valid_folds"),
        )
        .sort("mean_score", nulls_last=True)
    )
    best = summary.row(0, named=True)
    if best["mean_score"] is None:
        raise ValueError(
            "No candidate could be scored on any fold: the training window "
            f"({ds.train_end} periods; {ds.n_eval} held out) is too short for these models. "
            "Hold out fewer periods, use fewer folds, or try smaller hyperparameters."
        )
    return TuningResult(
        {k: best[k] for k in names}, export(scores, as_pandas), export(summary, as_pandas)
    )
