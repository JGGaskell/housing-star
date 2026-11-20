from datetime import date

import polars as pl
import pytest

from housing_star.config import CleaningConfig, GraphConfig
from housing_star.data.cache import TransactionCache
from housing_star.evaluation.backtest import backtest, rolling_windows
from housing_star.evaluation.cv import WalkForwardCV
from tests.conftest import synthetic_transactions


def test_rolling_windows_step_through_the_range():
    windows = rolling_windows("2000-01-01", "2004-01-01", window_years=2, step_months=12)
    assert windows == [
        (date(2000, 1, 1), date(2002, 1, 1)),
        (date(2001, 1, 1), date(2003, 1, 1)),
        (date(2002, 1, 1), date(2004, 1, 1)),
    ]


def test_range_shorter_than_a_window_gives_no_windows():
    assert rolling_windows("2000-01-01", "2003-01-01", window_years=5) == []


def stub_builder(make_dataset, calls):
    def build(postcode, start, end):
        calls.append((postcode, start, end))
        return make_dataset(n_periods=14, holdout=2)

    return build


def run(make_dataset, calls, **kwargs):
    return backtest(
        ["PC0"],
        start="2000-01-01",
        end="2013-01-01",
        window_years=10,
        step_months=24,
        cv=WalkForwardCV(n_folds=2, min_train=6),
        build_dataset=stub_builder(make_dataset, calls),
        **kwargs,
    )


def test_backtest_scores_three_models_per_window(make_dataset):
    calls = []
    out = run(make_dataset, calls)
    assert len(calls) == 2
    assert out["model"].to_list() == ["STAR", "AR(1)", "Drift"] * 2
    assert {"postcode", "window_start", "window_end", "star_p", "smape"} <= set(out.columns)


def test_finished_tasks_are_reused_from_checkpoints(make_dataset, tmp_path):
    first, second = [], []
    a = run(make_dataset, first, checkpoint_dir=tmp_path)
    b = run(make_dataset, second, checkpoint_dir=tmp_path)
    assert len(first) == 2 and second == []
    assert a.equals(b)
    assert len(list(tmp_path.glob("*.parquet"))) == 2


def test_windows_that_fail_are_skipped(make_dataset):
    def failing(postcode, start, end):
        raise ValueError("no data")

    out = backtest(["PC0"], start="2000-01-01", end="2013-01-01", build_dataset=failing)
    assert out.is_empty()


# The synthetic prices are an exact exponential, so the fitted growth dynamics are degenerate.
@pytest.mark.filterwarnings("ignore::housing_star.UnstableModelWarning")
def test_default_builder_reads_through_the_cache(tmp_path):
    out = backtest(
        ["GL5"],
        start="2000-01-01",
        end="2010-01-01",
        window_years=10,
        graph=GraphConfig(radius_m=15_000, max_nodes=4),
        cleaning=CleaningConfig(min_nodes=2),
        cache=TransactionCache(root=tmp_path, fetcher=synthetic_transactions),
        cv=WalkForwardCV(n_folds=3, min_train=6),
    )
    assert out["postcode"].unique().to_list() == ["GL5"] and out.height == 3
    assert isinstance(out, pl.DataFrame)
