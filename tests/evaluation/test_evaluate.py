import numpy as np
import polars as pl
import pytest

from housing_star.evaluation.cv import WalkForwardCV
from housing_star.evaluation.evaluate import cross_validate, evaluate, target_predictions
from housing_star.models.ar import ArModel
from housing_star.models.drift import DriftModel
from housing_star.models.star.estimation import StarFit
from housing_star.models.star.model import StarModel

MODELS = {"STAR": StarModel(), "AR(1)": ArModel(), "Drift": DriftModel()}


def test_evaluate_returns_one_row_per_model(ds):
    out = evaluate(ds, MODELS)
    assert out["model"].to_list() == list(MODELS)
    assert set(out["protocol"]) == {"one_step"} and set(out["n_eval"]) == {3}
    assert out["smape"].is_not_nan().all()


def test_evaluate_scores_the_target_node_when_there_is_one(ds):
    target_only = evaluate(ds, {"AR": ArModel()})["mae"][0]
    all_nodes = evaluate(
        ds.__class__(ds.prices, ds.W, ds.postcodes, ds.times, ds.train_end), {"AR": ArModel()}
    )["mae"][0]
    assert target_only != pytest.approx(all_nodes)


def test_recursive_protocol_is_labelled(ds):
    assert evaluate(ds, {"AR": ArModel()}, protocol="recursive")["protocol"].to_list() == [
        "recursive"
    ]


def test_target_predictions_are_in_pounds_with_one_row_per_model_period(ds):
    out = target_predictions(ds, {"AR": ArModel(), "Drift": DriftModel()})
    assert out.height == 2 * ds.n_eval
    np.testing.assert_allclose(
        out.filter(pl.col("model") == "AR")["actual"].to_numpy(), ds.prices[0, ds.train_end :]
    )
    assert out["time"].to_list()[: ds.n_eval] == ds.times[ds.train_end :]


def test_cross_validation_scores_every_model_on_every_fold(ds):
    out = cross_validate(MODELS, ds, WalkForwardCV(n_folds=3, min_train=6))
    assert out.height == 9 and out["score"].is_not_nan().all()
    assert sorted(out["fold"].unique().to_list()) == [0, 1, 2]


def test_cross_validation_never_looks_at_the_holdout(make_dataset):
    ds = make_dataset()
    tampered = make_dataset()
    tampered.prices[:, ds.train_end :] *= 9.0
    tampered = tampered.with_holdout(ds.n_eval)
    cv = WalkForwardCV(n_folds=3, min_train=6)
    a, b = cross_validate(MODELS, ds, cv), cross_validate(MODELS, tampered, cv)
    assert a.equals(b)


def test_a_model_that_cannot_be_fitted_on_a_fold_scores_nan(ds):
    out = cross_validate({"AR(8)": ArModel(p=8)}, ds, WalkForwardCV(n_folds=3, min_train=6))
    first, *later = out.sort("fold")["score"].to_list()  # only 6 training periods in fold 0
    assert np.isnan(first) and not any(np.isnan(later))


def test_metric_choice_changes_the_score(ds):
    cv = WalkForwardCV(n_folds=2, min_train=6)
    log = cross_validate({"AR": ArModel()}, ds, cv, metric="log_rmse")["score"]
    gbp = cross_validate({"AR": ArModel()}, ds, cv, metric="rmse")["score"]
    assert gbp.min() > 100 * log.max()


def test_unknown_metrics_raise(ds):
    with pytest.raises(KeyError):
        cross_validate({"AR": ArModel()}, ds, WalkForwardCV(n_folds=2, min_train=6), metric="nope")


def test_unstable_candidates_score_badly_without_warning_inside_cross_validation(
    ds, monkeypatch, recwarn
):
    unstable = StarFit(0.0, np.array([1.1]), np.array([0.0]), 0.01, 0.0)
    monkeypatch.setattr("housing_star.models.star.model.fit_star", lambda *a, **k: unstable)
    out = cross_validate({"STAR": StarModel()}, ds, WalkForwardCV(n_folds=2, min_train=6))
    assert out["score"].is_not_nan().all()
    assert not recwarn


def test_cross_validation_defaults_to_three_single_period_folds(ds):
    out = cross_validate({"AR": ArModel()}, ds)
    assert out["fold"].to_list() == [0, 1, 2]
    assert out["train_end"].to_list() == [18, 19, 20]
