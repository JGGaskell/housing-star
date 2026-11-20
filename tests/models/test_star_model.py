import numpy as np
import pytest

from housing_star.models.star.estimation import StarFit
from housing_star.models.star.model import StarModel, UnstableModelWarning, warn_if_unstable
from tests.conftest import dataset_from_log_prices


def test_fit_predict_forecast_roundtrip(ds):
    model = StarModel(p=1).fit(ds)
    assert model.predict(ds).shape == ds.y.shape
    assert model.predict_abs(ds).shape == ds.prices.shape
    assert model.forecast(ds.train_only(), 4).shape == (ds.N, 4)
    np.testing.assert_allclose(model.growth_, ds.growth_means)


def test_params_are_plain_python_values(ds):
    params = StarModel(p=2).fit(ds).params_
    assert params["p"] == 2 and len(params["phi"]) == len(params["theta"]) == 2
    assert isinstance(params["rho"], float) and "alpha" not in params


def test_fit_uses_only_the_training_window(make_dataset):
    ds = make_dataset()
    changed = make_dataset()
    changed.prices[:, ds.train_end :] *= 5.0  # hold-out changes; training prices untouched
    changed = changed.with_holdout(ds.n_eval)
    a, b = StarModel().fit(ds), StarModel().fit(changed)
    assert a.params_ == b.params_
    np.testing.assert_array_equal(a.growth_, b.growth_)


def test_unfitted_models_refuse_to_predict(ds):
    with pytest.raises(RuntimeError, match="not fitted"):
        StarModel().predict(ds)


def test_from_fit_wraps_known_parameters(ds, star_fit):
    model = StarModel.from_fit(star_fit, ds.growth_means)
    assert model.p == 1 and model.params_["rho"] == star_fit.rho
    np.testing.assert_array_equal(model.fit_.phi, star_fit.phi)


def test_forecast_accepts_scenario_shocks(ds, star_fit):
    model = StarModel.from_fit(star_fit, ds.growth_means)
    path = ds.y[1, -1] + np.array([0.1, 0.1])
    out = model.forecast(ds, 2, clamp={1: path})
    np.testing.assert_allclose(out[1], path)


def test_model_and_dataset_must_have_the_same_nodes(make_dataset):
    model = StarModel().fit(make_dataset(n_nodes=4))
    with pytest.raises(ValueError, match="fitted on 4 nodes"):
        model.predict(make_dataset(n_nodes=5))


def flat_then_jump(n_train=14, n_after=6, jump=0.5, n_nodes=5, seed=0):
    """Prices wander around a constant level, then shift up for good after the training window."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=0.03, size=(n_nodes, n_train + n_after))
    level = noise + np.where(np.arange(n_train + n_after) >= n_train, jump, 0.0)
    return dataset_from_log_prices(level + 12.0, train_end=n_train)


def test_forecasts_start_from_the_latest_price_after_a_level_shift():
    # A model of price levels would fall back to the old training average, about 0.5 below.
    ds = flat_then_jump()
    model = StarModel(p=1).fit(ds)
    forecast = model.forecast(ds, 1)[:, 0]
    assert np.all(np.abs(forecast - ds.y[:, -1]) < 0.15)


def test_persistence_of_a_known_model(star_fit):
    W = np.array([[0.0, 1.0], [1.0, 0.0]])
    fit = StarFit(
        0.0, np.array([0.5]), np.array([0.2]), 0.01, 0.0
    )  # 0.5 I + 0.2 W: eigenvalues 0.7, 0.3
    model = StarModel.from_fit(fit, np.zeros(2))
    assert model.persistence(W) == pytest.approx(0.7)


def test_persistence_of_a_two_lag_model_uses_the_companion_matrix():
    W = np.array([[0.0, 1.0], [1.0, 0.0]])
    fit = StarFit(
        0.0, np.array([0.5, 0.2]), np.zeros(2), 0.01, 0.0
    )  # AR(2): lambda^2 = 0.5 lambda + 0.2
    model = StarModel.from_fit(fit, np.zeros(2))
    assert model.persistence(W) == pytest.approx((0.5 + np.sqrt(1.05)) / 2)


def explosive_model():
    fit = StarFit(0.0, np.array([1.1]), np.array([0.0]), 0.01, 0.0)
    return StarModel.from_fit(fit, np.zeros(2)), np.full((2, 2), 0.5)


def test_an_unstable_model_triggers_a_warning():
    model, W = explosive_model()
    with pytest.warns(UnstableModelWarning, match=r"persistence is 1\.1"):
        warn_if_unstable(model, W)


def test_a_stable_model_does_not_warn(star_fit, ds, recwarn):
    warn_if_unstable(StarModel.from_fit(star_fit, ds.growth_means), ds.W)
    assert not recwarn


def test_fit_warns_when_the_fitted_dynamics_do_not_die_out(ds, monkeypatch):
    unstable = StarFit(0.0, np.array([1.1]), np.array([0.0]), 0.01, 0.0)
    monkeypatch.setattr("housing_star.models.star.model.fit_star", lambda *a, **k: unstable)
    with pytest.warns(UnstableModelWarning):
        StarModel().fit(ds)
