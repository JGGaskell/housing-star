import numpy as np
import pytest

from housing_star.models.star.estimation import StarFit
from housing_star.models.star.prediction import forecast, one_step_predictions
from housing_star.spatial.weights import inverse_distance_weights

N = 5
W = inverse_distance_weights(np.arange(N) * 1_000.0, np.zeros(N))
# log prices: a random walk, so growth is non-trivial
Y = np.cumsum(np.random.default_rng(0).normal(scale=0.1, size=(N, 12)), axis=1)
G = np.full(N, 0.03)  # average growth per period


def make_fit(rho=0.3, phi=(0.4,), theta=(0.2,)):
    return StarFit(rho, np.array(phi), np.array(theta), 0.01, 0.0)


def growth(y):
    return np.diff(y, axis=1) - G[:, None]


def test_one_step_predictions_leave_the_first_p_plus_one_columns_missing():
    out = one_step_predictions(Y, W, make_fit(phi=(0.4, 0.1), theta=(0.2, 0.0)), G)
    assert np.isnan(out[:, :3]).all() and np.isfinite(out[:, 3:]).all()


def test_one_step_predictions_of_a_short_series_are_all_missing():
    assert np.isnan(one_step_predictions(Y[:, :1], W, make_fit(), G)).all()


def test_one_step_matches_the_model_equation():
    fit = make_fit()
    pred = one_step_predictions(Y, W, fit, G)[:, 6]
    z_hat = pred - Y[:, 5] - G
    z_lag = growth(Y)[:, 4]  # the growth observed in period 5
    np.testing.assert_allclose((np.eye(N) - 0.3 * W) @ z_hat, 0.4 * z_lag + 0.2 * W @ z_lag)


def test_without_dynamics_predictions_are_the_previous_price_plus_average_growth():
    fit = make_fit(rho=0.0, phi=(0.0,), theta=(0.0,))
    np.testing.assert_allclose(one_step_predictions(Y, W, fit, G)[:, 2:], Y[:, 1:-1] + G[:, None])
    np.testing.assert_allclose(forecast(Y, W, fit, G, 3), Y[:, [-1]] + G[:, None] * [1, 2, 3])


def test_first_forecast_step_equals_a_one_step_prediction():
    fit = make_fit(phi=(0.4, 0.1), theta=(0.2, 0.05))
    extended = np.hstack([Y, np.zeros((N, 1))])
    np.testing.assert_allclose(
        forecast(Y, W, fit, G, 1)[:, 0], one_step_predictions(extended, W, fit, G)[:, -1]
    )


def test_forecasts_are_fed_back_as_lags():
    fit = make_fit()
    out = forecast(Y, W, fit, G, 3)
    extended = np.hstack([Y, out[:, :2], np.zeros((N, 1))])
    np.testing.assert_allclose(out[:, 2], one_step_predictions(extended, W, fit, G)[:, -1])


def test_clamped_nodes_follow_their_price_path_exactly():
    path = Y[2, -1] + np.array([0.5, 0.2, -0.1])
    out = forecast(Y, W, make_fit(), G, 3, clamp={2: path})
    np.testing.assert_allclose(out[2], path)


def test_clamping_spills_over_in_the_same_period():
    base = forecast(Y, W, make_fit(), G, 2)
    out = forecast(Y, W, make_fit(), G, 2, clamp={2: base[2] + 0.3})
    assert np.all(out[[1, 3], 0] > base[[1, 3], 0])


def test_without_spatial_dependence_nothing_spills_over():
    fit = make_fit(rho=0.0, theta=(0.0,))
    base = forecast(Y, W, fit, G, 4)
    out = forecast(Y, W, fit, G, 4, clamp={2: base[2] + 0.3}, impulse={0: np.full(4, 0.1)})
    keep = [1, 3, 4]
    np.testing.assert_allclose(out[keep], base[keep])


def test_impulse_moves_the_node_by_exactly_the_offset_and_spills_over():
    base = forecast(Y, W, make_fit(), G, 3)
    out = forecast(Y, W, make_fit(), G, 3, impulse={1: np.array([0.2, 0.0, 0.0])})
    assert out[1, 0] - base[1, 0] == pytest.approx(0.2)
    assert np.all(out[[0, 2], 0] > base[[0, 2], 0])


def test_an_impulse_shifts_the_price_level_for_good():
    fit = make_fit(rho=0.0, phi=(0.0,), theta=(0.0,))
    base = forecast(Y, W, fit, G, 4)
    out = forecast(Y, W, fit, G, 4, impulse={1: np.array([0.2, 0.0, 0.0, 0.0])})
    np.testing.assert_allclose(out[1] - base[1], 0.2)  # no dynamics: the new level is kept
    np.testing.assert_allclose(np.delete(out, 1, axis=0), np.delete(base, 1, axis=0))


def test_impulses_compound():
    fit = make_fit(rho=0.0, phi=(0.0,), theta=(0.0,))
    base = forecast(Y, W, fit, G, 3)
    out = forecast(Y, W, fit, G, 3, impulse={1: np.array([0.2, 0.1, 0.0])})
    np.testing.assert_allclose(out[1] - base[1], [0.2, 0.3, 0.3])


def test_simultaneous_impulses_each_hit_their_offset():
    base = forecast(Y, W, make_fit(), G, 1)
    out = forecast(Y, W, make_fit(), G, 1, impulse={0: np.array([0.1]), 3: np.array([-0.2])})
    np.testing.assert_allclose(out[[0, 3], 0] - base[[0, 3], 0], [0.1, -0.2])


def test_clamp_overrides_an_impulse_on_the_same_node():
    path = Y[1, -1] + np.array([0.3, 0.3])
    clamped = forecast(Y, W, make_fit(), G, 2, clamp={1: path})
    both = forecast(Y, W, make_fit(), G, 2, clamp={1: path}, impulse={1: np.array([0.5, 0.5])})
    np.testing.assert_allclose(both, clamped)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"horizon": 0}, "horizon"),
        ({"horizon": 2, "clamp": {9: np.zeros(2)}}, "Invalid clamp"),
        ({"horizon": 2, "impulse": {1: np.zeros(3)}}, "Invalid impulse"),
    ],
)
def test_invalid_forecast_requests_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        forecast(Y, W, make_fit(), G, **kwargs)


def test_forecast_needs_p_plus_one_observed_periods():
    with pytest.raises(ValueError, match="seed"):
        forecast(Y[:, :2], W, make_fit(phi=(0.4, 0.1), theta=(0.2, 0.0)), G, 1)
