import pytest

from housing_star.evaluation.cv import Fold, WalkForwardCV


def test_folds_are_evenly_spaced_and_never_overlap_training():
    folds = list(WalkForwardCV(n_folds=3, horizon=1, min_train=6).split(15))
    assert folds == [Fold(0, 6, 7), Fold(1, 9, 10), Fold(2, 12, 13)]
    assert all(f.eval_end > f.train_end for f in folds)


def test_by_default_the_folds_are_the_last_periods_of_the_training_window():
    folds = list(WalkForwardCV(n_folds=3).split(10))
    assert folds == [Fold(0, 7, 8), Fold(1, 8, 9), Fold(2, 9, 10)]


def test_default_folds_step_by_the_horizon_and_leave_the_rest_for_training():
    assert [(f.train_end, f.eval_end) for f in WalkForwardCV(n_folds=2, horizon=3).split(10)] == [
        (4, 7),
        (7, 10),
    ]


def test_horizon_widens_each_evaluation_window():
    assert [(f.train_end, f.eval_end) for f in WalkForwardCV(2, 2, 4).split(10)] == [(4, 6), (7, 9)]


def test_a_window_that_is_too_short_is_rejected_with_the_remedies():
    with pytest.raises(ValueError, match="at least 9.*smaller min_train.*hold out fewer"):
        list(WalkForwardCV(n_folds=3, horizon=1, min_train=6).split(8))


def test_the_default_folds_need_at_least_one_training_period():
    assert len(list(WalkForwardCV(n_folds=3).split(4))) == 3
    with pytest.raises(ValueError, match="at least 4.*hold out fewer") as info:
        list(WalkForwardCV(n_folds=3).split(3))
    assert "min_train" not in str(info.value)


@pytest.mark.parametrize("kwargs", [{"n_folds": 0}, {"horizon": 0}, {"min_train": 0}])
def test_parameters_must_be_positive(kwargs):
    with pytest.raises(ValueError, match=">= 1"):
        WalkForwardCV(**kwargs)
