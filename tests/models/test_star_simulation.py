import numpy as np

from housing_star.models.star.simulation import random_weights, simulate_log_prices, simulate_star


def test_random_weights_are_row_standardised_and_reproducible():
    W = random_weights(5, seed=4)
    np.testing.assert_allclose(W.sum(axis=1), 1.0)
    assert np.all(np.diag(W) == 0)
    np.testing.assert_array_equal(W, random_weights(5, seed=4))


def test_simulation_shape_and_reproducibility():
    W = random_weights(4)
    kwargs = dict(rho=0.2, phi=np.array([0.3]), theta=np.array([0.1]))
    a = simulate_star(W, 30, seed=1, **kwargs)
    assert a.shape == (4, 30)
    np.testing.assert_array_equal(a, simulate_star(W, 30, seed=1, **kwargs))
    assert not np.array_equal(a, simulate_star(W, 30, seed=2, **kwargs))


def test_log_prices_grow_at_the_requested_rate_and_accumulate_stochastic_growth():
    W = random_weights(3)
    quiet = simulate_log_prices(
        W,
        6,
        growth=np.array([0.0, 0.1, 0.2]),
        rho=0.0,
        phi=np.zeros(1),
        theta=np.zeros(1),
        sigma=0.0,
    )
    np.testing.assert_allclose(quiet, np.outer([0.0, 0.1, 0.2], np.arange(6)), atol=1e-12)
    noisy = simulate_log_prices(W, 6, rho=0.2, phi=np.array([0.3]), theta=np.array([0.1]), seed=3)
    z = simulate_star(W, 6, rho=0.2, phi=np.array([0.3]), theta=np.array([0.1]), seed=3)
    np.testing.assert_allclose(np.diff(noisy, axis=1), z[:, 1:])
