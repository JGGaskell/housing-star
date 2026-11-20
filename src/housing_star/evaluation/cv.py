"""Expanding-window walk-forward cross-validation."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    """Train on periods ``[0, train_end)``; evaluate on ``[train_end, eval_end)``."""

    index: int
    train_end: int
    eval_end: int


@dataclass(frozen=True)
class WalkForwardCV:
    """Forecast origins inside a dataset's *training* window; the hold-out is never touched.

    Each fold trains on everything before its origin and evaluates on the next ``horizon``
    periods, so training data always precede evaluation data. By default the folds are the last
    ``n_folds`` blocks of ``horizon`` periods of the training window, back to back: with 10
    training periods, ``WalkForwardCV(n_folds=3)`` trains on 7, 8 and 9 periods and scores
    periods 8, 9 and 10. How much is left to train on is therefore decided by where you put the
    hold-out, not by a second setting.

    Parameters
    ----------
    n_folds : number of forecast origins.
    horizon : periods evaluated per fold.
    min_train : optional. If given, the first fold trains on exactly this many periods and the
        other origins are spread evenly between it and the end of the training window, which
        guarantees a minimum amount of training data.
    """

    n_folds: int = 3
    horizon: int = 1
    min_train: int | None = None

    def __post_init__(self) -> None:
        if min(self.n_folds, self.horizon) < 1 or (
            self.min_train is not None and self.min_train < 1
        ):
            raise ValueError("n_folds, horizon and min_train must all be >= 1.")

    def split(self, T: int) -> Iterator[Fold]:
        """Yield the folds for a training window of ``T`` periods.

        Raises
        ------
        ValueError
            If the window is too short for ``n_folds`` folds of ``horizon`` periods (plus
            ``min_train`` periods to train the first on, when given).
        """
        evaluated = self.n_folds * self.horizon
        needed = evaluated + (self.min_train if self.min_train is not None else 1)
        if needed > T:
            raise ValueError(
                f"The training window has {T} periods but {self.n_folds} folds of {self.horizon} "
                f"need at least {needed}. Use fewer folds, a shorter horizon"
                f"{', a smaller min_train' if self.min_train is not None else ''}, "
                "or hold out fewer periods."
            )
        if self.min_train is None:
            for i in range(self.n_folds):
                origin = T - evaluated + i * self.horizon
                yield Fold(i, origin, origin + self.horizon)
            return
        step = (T - self.min_train) / self.n_folds
        for i in range(self.n_folds):
            origin = self.min_train + round(i * step)
            yield Fold(i, origin, min(origin + self.horizon, T))
