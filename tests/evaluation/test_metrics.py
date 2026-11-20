import numpy as np
import pytest

from housing_star.evaluation.metrics import mae, mape, rmse, score_holdout, smape

T = np.array([100.0, 200.0, np.nan, 400.0])
P = np.array([110.0, 180.0, 5.0, np.nan])  # only the first two pairs are usable


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        (mae, 15.0),
        (rmse, np.sqrt((100 + 400) / 2)),
        (mape, (10 + 10) / 2),
        (smape, (2 * 10 / 210 + 2 * 20 / 380) / 2 * 100),
    ],
)
def test_metrics_ignore_missing_pairs(metric, expected):
    assert metric(T, P) == pytest.approx(expected)


@pytest.mark.parametrize("metric", [mae, rmse, mape, smape])
def test_metrics_of_nothing_are_nan(metric):
    assert np.isnan(metric(np.array([np.nan]), np.array([1.0])))


def test_percentage_metrics_skip_degenerate_denominators():
    zeros = np.zeros(2)
    assert np.isnan(mape(zeros, np.ones(2)))
    assert np.isnan(smape(zeros, zeros))


def test_score_holdout_reports_log_and_price_metrics(ds):
    perfect = score_holdout(ds, ds.y_eval())
    assert set(perfect) == {"log_rmse", "log_mae", "mae", "rmse", "mape", "smape"}
    assert all(v == pytest.approx(0.0, abs=1e-9) for v in perfect.values())


def test_a_ten_percent_error_is_ten_percent_in_price_terms(ds):
    scores = score_holdout(ds, ds.y_eval() + np.log(1.1), nodes=[0])
    assert scores["mape"] == pytest.approx(10.0)
    assert scores["log_mae"] == pytest.approx(np.log(1.1))
