import numpy as np
import pytest

from housing_star.evaluation.protocols import holdout_predictions
from housing_star.models.ar import ArModel
from housing_star.models.star.model import StarModel


@pytest.mark.parametrize("protocol", ["one_step", "recursive"])
def test_predictions_cover_the_holdout(ds, protocol):
    assert holdout_predictions(ArModel(), ds, protocol).shape == (ds.N, ds.n_eval)


def test_one_step_uses_observed_lags_and_recursive_does_not(ds):
    one = holdout_predictions(ArModel(), ds, "one_step")
    rec = holdout_predictions(ArModel(), ds, "recursive")
    np.testing.assert_allclose(one[:, 0], rec[:, 0])  # first step: identical information
    assert not np.allclose(one[:, 1:], rec[:, 1:])


def test_the_passed_model_is_not_modified(ds):
    model = StarModel()
    holdout_predictions(model, ds)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(ds)


def test_unknown_protocols_are_rejected(ds):
    with pytest.raises(ValueError, match="protocol"):
        holdout_predictions(ArModel(), ds, "oracle")  # type: ignore[arg-type]


def test_a_dataset_without_holdout_cannot_be_evaluated(ds):
    with pytest.raises(ValueError, match="no hold-out"):
        holdout_predictions(ArModel(), ds.train_only())
