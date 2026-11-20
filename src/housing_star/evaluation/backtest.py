"""Rolling-window backtest across postcodes, with optional resumable checkpoints."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
from joblib import Parallel, delayed

from housing_star._dates import add_months, parse_date
from housing_star._output import export
from housing_star.config import CleaningConfig, GraphConfig
from housing_star.data.cache import TransactionCache
from housing_star.data.dataset import PriceSeriesDataset
from housing_star.evaluation.cv import WalkForwardCV
from housing_star.evaluation.evaluate import evaluate
from housing_star.evaluation.protocols import Protocol
from housing_star.evaluation.tuning import tune
from housing_star.models.ar import ArModel
from housing_star.models.drift import DriftModel
from housing_star.models.star.model import StarModel

logger = logging.getLogger(__name__)

DatasetBuilder = Callable[[str, date, date], PriceSeriesDataset]


def rolling_windows(
    start: str | date, end: str | date, *, window_years: int, step_months: int = 12
) -> list[tuple[date, date]]:
    """Consecutive ``window_years``-long windows inside ``[start, end]``, ``step_months`` apart."""
    first, last = parse_date(start), parse_date(end)
    windows = []
    cursor = first
    while add_months(cursor, window_years * 12) <= last:
        windows.append((cursor, add_months(cursor, window_years * 12)))
        cursor = add_months(cursor, step_months)
    return windows


def _run_task(
    postcode: str,
    window: tuple[date, date],
    build: DatasetBuilder,
    *,
    p_grid: Sequence[int],
    cv: WalkForwardCV,
    protocol: Protocol,
    checkpoint: Path | None,
) -> pl.DataFrame:
    """Build, tune STAR's ``p`` and evaluate STAR, AR(1) and a drift (average growth) benchmark for one window."""
    if checkpoint is not None and checkpoint.exists():
        return pl.read_parquet(checkpoint)
    try:
        ds = build(postcode, *window)
        best = tune(StarModel, ds, {"p": list(p_grid)}, cv, protocol=protocol).best_params
        models = {"STAR": StarModel(**best), "AR(1)": ArModel(p=1), "Drift": DriftModel()}
        result = evaluate(ds, models, protocol=protocol).with_columns(
            pl.lit(postcode).alias("postcode"),
            pl.lit(window[0]).alias("window_start"),
            pl.lit(window[1]).alias("window_end"),
            pl.lit(best["p"]).alias("star_p"),
        )
    except ValueError as exc:
        logger.warning("Skipping %s %s-%s: %s", postcode, window[0], window[1], exc)
        return pl.DataFrame()
    if checkpoint is not None:
        result.write_parquet(checkpoint)
    return result


def backtest(
    postcodes: Sequence[str],
    *,
    start: str | date,
    end: str | date,
    window_years: int = 10,
    step_months: int = 12,
    holdout: int = 1,
    freq: str = "1y",
    p_grid: Sequence[int] = (1, 2, 3),
    cv: WalkForwardCV | None = None,
    protocol: Protocol = "one_step",
    graph: GraphConfig | None = None,
    cleaning: CleaningConfig | None = None,
    cache: TransactionCache | None = None,
    build_dataset: DatasetBuilder | None = None,
    checkpoint_dir: str | Path | None = None,
    n_jobs: int = 1,
    as_pandas: bool = False,
) -> Any:
    """Evaluate STAR (``p`` tuned per window), AR(1) and a drift (average growth) benchmark over rolling windows.

    For each ``(postcode, window)``: build a dataset with ``holdout`` trailing periods, choose
    STAR's ``p`` by walk-forward CV on the training window, then score all three models on the
    hold-out. Windows that cannot be built or fitted are logged and skipped.

    Parameters
    ----------
    window_years : length of each window; ``step_months`` apart between window starts.
    build_dataset : ``(postcode, start, end) -> PriceSeriesDataset``; defaults to
        :meth:`PriceSeriesDataset.from_postcode` with the given ``graph``/``cleaning``/``cache``.
    checkpoint_dir : if given, each finished task is saved there and reused on re-runs.
    n_jobs : joblib workers; ``build_dataset`` must be picklable when this exceeds 1.
    as_pandas : return a pandas DataFrame instead of polars.

    Returns
    -------
    One row per ``(postcode, window, model)`` with the metrics of ``evaluate``.
    """
    cv = cv or WalkForwardCV()

    def default_build(postcode: str, window_start: date, window_end: date) -> PriceSeriesDataset:
        return PriceSeriesDataset.from_postcode(
            postcode,
            start=window_start,
            end=window_end,
            freq=freq,
            holdout=holdout,
            graph=graph,
            cleaning=cleaning,
            cache=cache,
        )

    folder = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
    tasks = [
        (postcode, window)
        for postcode in postcodes
        for window in rolling_windows(
            start, end, window_years=window_years, step_months=step_months
        )
    ]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_task)(
            postcode,
            window,
            build_dataset or default_build,
            p_grid=p_grid,
            cv=cv,
            protocol=protocol,
            checkpoint=None
            if folder is None
            else folder / f"{postcode.replace(' ', '_')}__{window[0]}__{window[1]}.parquet",
        )
        for postcode, window in tasks
    )
    frames = [r for r in results if not r.is_empty()]
    return export(pl.concat(frames) if frames else pl.DataFrame(), as_pandas)
