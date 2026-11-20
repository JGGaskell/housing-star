"""Synthetic STAR data, for demos and for checking that estimation recovers known parameters.

``simulate_star`` draws *growth* from the STAR dynamics; ``simulate_log_prices`` accumulates it
into log prices that grow at a given average rate.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from housing_star._types import FloatArray
from housing_star.spatial.weights import inverse_distance_weights


def random_weights(n_nodes: int, *, seed: int = 0) -> FloatArray:
    """Row-standardised inverse-distance weights for ``n_nodes`` points drawn at random."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.0, 10_000.0, size=(n_nodes, 2))
    return inverse_distance_weights(xy[:, 0], xy[:, 1], eps_m=10.0)


def simulate_star(
    W: FloatArray,
    n_periods: int,
    *,
    rho: float,
    phi: FloatArray,
    theta: FloatArray,
    sigma: float = 0.05,
    burn_in: int = 100,
    seed: int = 0,
) -> FloatArray:
    """Simulate an ``(N, n_periods)`` panel of STAR(p) growth (after discarding ``burn_in`` periods)."""
    N = W.shape[0]
    p = len(phi)
    rng = np.random.default_rng(seed)
    A_inv = np.linalg.inv(np.eye(N) - rho * W)
    y = np.zeros((N, n_periods + burn_in + p))
    for t in range(p, y.shape[1]):
        rhs = rng.normal(0.0, sigma, N)
        for ell in range(1, p + 1):
            rhs = rhs + phi[ell - 1] * y[:, t - ell] + theta[ell - 1] * (W @ y[:, t - ell])
        y[:, t] = A_inv @ rhs
    return y[:, -n_periods:]


def simulate_log_prices(
    W: FloatArray, n_periods: int, *, growth: FloatArray | float = 0.0, **kwargs: Any
) -> FloatArray:
    """Simulate ``(N, n_periods)`` log prices: node ``i`` grows at ``growth[i]`` plus STAR growth.

    ``kwargs`` are those of :func:`simulate_star`. The first column is the starting log price (0).
    """
    z = simulate_star(W, n_periods, **kwargs)
    steps = z + np.asarray(growth, dtype=float).reshape(-1, 1)
    return np.hstack([np.zeros((W.shape[0], 1)), np.cumsum(steps[:, 1:], axis=1)])
