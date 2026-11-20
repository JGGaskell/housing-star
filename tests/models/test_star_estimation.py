import numpy as np
import pytest

from housing_star.models.star.estimation import (
    StarFit,
    fit_star,
    golden_section_maximise,
    persistence,
)
from housing_star.models.star.simulation import random_weights, simulate_star


def test_golden_section_finds_the_peak_of_a_parabola():
    assert golden_section_maximise(lambda x: -((x - 0.3) ** 2), -1.0, 1.0) == pytest.approx(
        0.3, abs=1e-5
    )


def test_golden_section_handles_a_peak_at_either_edge():
    assert golden_section_maximise(lambda x: x, 0.0, 1.0) == pytest.approx(1.0, abs=1e-4)
    assert golden_section_maximise(lambda x: -x, 0.0, 1.0) == pytest.approx(0.0, abs=1e-4)


def test_estimation_recovers_known_parameters():
    W = random_weights(10, seed=3)
    truth = dict(rho=0.3, phi=np.array([0.4]), theta=np.array([0.2]))
    y = simulate_star(W, 600, sigma=0.05, seed=1, **truth)
    fit = fit_star(y, W, p=1)
    assert fit.rho == pytest.approx(0.3, abs=0.07)
    assert fit.phi[0] == pytest.approx(0.4, abs=0.05)
    assert fit.theta[0] == pytest.approx(0.2, abs=0.08)
    assert fit.sigma2 == pytest.approx(0.05**2, rel=0.2)
    assert np.isfinite(fit.loglik) and not hasattr(fit, "alpha")


def test_estimation_supports_several_lags():
    W = random_weights(8, seed=0)
    y = simulate_star(
        W, 400, rho=0.2, phi=np.array([0.3, 0.1]), theta=np.array([0.1, 0.05]), seed=2
    )
    fit = fit_star(y, W, p=2)
    assert fit.phi.shape == fit.theta.shape == (2,)


def test_fit_is_invariant_to_the_grid_resolution():
    W = random_weights(6, seed=0)
    y = simulate_star(W, 200, rho=0.3, phi=np.array([0.4]), theta=np.array([0.2]), seed=0)
    coarse, fine = fit_star(y, W, 1, grid_size=11), fit_star(y, W, 1, grid_size=201)
    assert coarse.rho == pytest.approx(fine.rho, abs=1e-3)


@pytest.mark.parametrize(
    ("n", "t", "p", "w_n", "kwargs", "message"),
    [
        (3, 10, 1, 4, {}, "must be"),
        (1, 10, 1, 1, {}, "at least 2 nodes"),
        (3, 10, 0, 3, {}, "p must be"),
        (3, 10, 1, 3, {"grid_size": 2}, "grid_size"),
        (3, 2, 2, 3, {}, "too short"),
        (2, 2, 1, 2, {}, "too short"),
    ],
)
def test_invalid_inputs_are_rejected(n, t, p, w_n, kwargs, message):
    with pytest.raises(ValueError, match=message):
        fit_star(
            np.random.default_rng(0).normal(size=(n, t)),
            np.full((w_n, w_n), 1 / max(w_n, 1)),
            p,
            **kwargs,
        )


def test_a_degenerate_likelihood_is_reported():
    W = random_weights(3, seed=0)
    with pytest.raises(ValueError, match="degenerate"):
        fit_star(np.zeros((3, 10)), W, p=1)


def test_persistence_is_zero_without_dynamics():
    W = random_weights(4, seed=0)
    assert persistence(StarFit(0.3, np.zeros(1), np.zeros(1), 0.01, 0.0), W) == 0.0


def test_persistence_of_a_scalar_ar_is_its_coefficient():
    W = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert persistence(StarFit(0.0, np.array([0.6]), np.zeros(1), 0.01, 0.0), W) == pytest.approx(
        0.6
    )
