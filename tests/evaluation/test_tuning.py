import numpy as np
import pytest

from housing_star.data.dataset import PriceSeriesDataset
from housing_star.evaluation.cv import WalkForwardCV
from housing_star.evaluation.tuning import tune
from housing_star.models.ar import ArModel


def ar2_dataset(n_nodes=3, n_periods=60, holdout=2):
    """Log prices follow a noiseless AR(2): only p >= 2 can fit them."""
    rng = np.random.default_rng(0)
    z = np.zeros((n_nodes, n_periods))
    z[:, :2] = rng.normal(size=(n_nodes, 2))
    for t in range(2, n_periods):
        z[:, t] = 0.5 * z[:, t - 1] - 0.4 * z[:, t - 2] + 0.01 * rng.normal(size=n_nodes)
    return PriceSeriesDataset(
        np.exp(z),
        np.full((n_nodes, n_nodes), 1 / n_nodes),
        list("ABC")[:n_nodes],
        [str(t) for t in range(n_periods)],
        n_periods - holdout,
        target="A",
    )


def test_tuning_picks_the_lag_order_that_fits_the_data():
    result = tune(ArModel, ar2_dataset(), {"p": [1, 2, 3]}, WalkForwardCV(n_folds=5, min_train=20))
    assert result.best_params["p"] in (2, 3)
    assert result.best_params["p"] != 1
    assert result.summary["mean_score"].is_sorted()


def test_scores_have_one_row_per_candidate_and_fold():
    result = tune(ArModel, ar2_dataset(), {"p": [1, 2]}, WalkForwardCV(n_folds=3, min_train=20))
    assert result.scores.columns == ["p", "fold", "score"]
    assert result.scores.height == 6
    assert result.summary["n_valid_folds"].to_list() == [3, 3]


def test_every_combination_of_a_multi_parameter_grid_is_tried(ds):
    class Dummy(ArModel):
        def __init__(self, p=1, flag=False):
            super().__init__(p)

    result = tune(
        Dummy, ds, {"p": [1, 2], "flag": [True, False]}, WalkForwardCV(n_folds=2, min_train=6)
    )
    assert result.summary.height == 4
    assert set(result.best_params) == {"p", "flag"}


def test_candidates_that_fail_are_ignored_but_not_when_all_fail(ds):
    cv = WalkForwardCV(n_folds=2, min_train=6)
    assert tune(ArModel, ds, {"p": [1, 50]}, cv).best_params == {"p": 1}
    with pytest.raises(ValueError, match="No candidate"):
        tune(ArModel, ds, {"p": [50]}, cv)


def test_the_folds_default_to_the_end_of_the_training_window(ds):
    result = tune(ArModel, ds, {"p": [1, 2]})
    assert result.scores.height == 2 * 3  # two candidates, three default folds


def test_a_training_window_that_is_too_short_explains_how_much_is_held_out(ds):
    with pytest.raises(ValueError, match=r"holds out 3 of 24 periods"):
        tune(ArModel, ds, {"p": [1]}, WalkForwardCV(n_folds=30))
    with pytest.raises(ValueError, match=r"21 periods; 3 held out"):
        tune(ArModel, ds, {"p": [99]})
