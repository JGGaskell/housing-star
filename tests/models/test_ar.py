import numpy as np
import pytest

from housing_star.models.ar import ArModel
from tests.conftest import dataset_from_log_prices


def alternating_dataset(n_nodes=3, n_train=13, holdout=3):
    """Growth is ``g_i`` plus ``+/- a``, alternating, so z_t = -z_{t-1} exactly.

    With an even number of training changes the average growth is exactly ``g_i``, so an AR(1)
    on relative growth must find phi = -1 and predict every later year without error.
    """
    n_periods = n_train + holdout
    signs = (-1.0) ** np.arange(1, n_periods)
    g = 0.02 * (np.arange(n_nodes) + 1)
    steps = g[:, None] + 0.05 * (np.arange(n_nodes) + 1)[:, None] * signs
    log_prices = np.hstack([np.zeros((n_nodes, 1)), np.cumsum(steps, axis=1)])
    return dataset_from_log_prices(log_prices, train_end=n_train)


def test_ar1_recovers_its_coefficient():
    model = ArModel(p=1).fit(alternating_dataset())
    np.testing.assert_allclose(model.coef_, -1.0, atol=1e-8)
    assert model.coef_.shape == (3, 1)


def test_one_step_predictions_of_a_perfect_ar_are_exact():
    ds = alternating_dataset()
    pred = ArModel(p=1).fit(ds).predict(ds)
    assert np.isnan(pred[:, :2]).all()  # needs the previous price and one lag of growth
    np.testing.assert_allclose(pred[:, 2:], ds.y[:, 2:], atol=1e-8)


def test_recursive_forecast_of_a_perfect_ar_matches_the_hold_out():
    ds = alternating_dataset()
    out = ArModel(p=1).fit(ds).forecast(ds.train_only(), 3)
    np.testing.assert_allclose(out, ds.y_eval(), atol=1e-8)


def test_forecasts_start_from_the_latest_price_not_the_training_mean(ds):
    model = ArModel(p=1).fit(ds)
    first = model.forecast(ds, 1)[:, 0]
    assert np.all(np.abs(first - ds.y[:, -1]) < 0.5)


def test_predictions_use_the_dataset_they_are_given(make_dataset):
    ds, other = make_dataset(seed=0), make_dataset(seed=1)
    model = ArModel(p=1).fit(ds)
    z = np.diff(other.y, axis=1) - model.growth_[:, None]
    expected = other.y[:, 1:-1] + model.growth_[:, None] + model.coef_ * z[:, :-1]
    np.testing.assert_allclose(model.predict(other)[:, 2:], expected)


def test_recursive_forecast_feeds_back_its_own_values(ds):
    model = ArModel(p=2).fit(ds)
    train = ds.train_only()
    z = np.diff(train.y, axis=1) - model.growth_[:, None]
    phi1, phi2 = model.coef_[:, 0], model.coef_[:, 1]
    z1 = phi1 * z[:, -1] + phi2 * z[:, -2]
    z2 = phi1 * z1 + phi2 * z[:, -1]
    out = model.forecast(train, 2)
    level1 = train.y[:, -1] + model.growth_ + z1
    np.testing.assert_allclose(out[:, 0], level1)
    np.testing.assert_allclose(out[:, 1], level1 + model.growth_ + z2)


def test_prediction_of_a_too_short_dataset_is_all_missing(ds):
    assert np.isnan(ArModel(p=1).fit(ds).predict(ds.window(1, 1))).all()


def test_params_expose_coefficients(ds):
    params = ArModel(p=1).fit(ds).params_
    assert params["p"] == 1 and np.array(params["coef"]).shape == (ds.N, 1)


@pytest.mark.parametrize("p", [0, -1])
def test_lag_order_must_be_positive(p):
    with pytest.raises(ValueError, match="p must be"):
        ArModel(p=p)


def test_fit_needs_enough_training_periods(ds):
    with pytest.raises(ValueError, match="too short"):
        ArModel(p=2).fit(ds.window(3, 3))


def test_model_and_dataset_must_have_the_same_nodes(make_dataset):
    model = ArModel().fit(make_dataset(n_nodes=4))
    with pytest.raises(ValueError, match="fitted on 4 nodes"):
        model.predict(make_dataset(n_nodes=5))


def test_forecast_validates_horizon_and_history(ds):
    model = ArModel(p=2).fit(ds)
    with pytest.raises(ValueError, match="horizon"):
        model.forecast(ds, 0)
    with pytest.raises(ValueError, match="at least p\\+1=3"):
        model.forecast(ds.window(1, 1), 1)


def test_unfitted_model_refuses_to_predict(ds):
    with pytest.raises(RuntimeError, match="not fitted"):
        ArModel().predict(ds)
