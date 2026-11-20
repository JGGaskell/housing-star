r"""One-step-ahead prediction and recursive forecasting for a fitted STAR(p) model.

Inputs and outputs are log price levels. Internally each node's growth net of its average,
:math:`z_t`, follows the STAR dynamics, and levels are rebuilt from the latest observed price
(see :mod:`housing_star.models._growth`).

Both predictions reduce to solving :math:`(I - \rho W) z_t = r_t` where :math:`r_t` collects the
lagged terms. Scenario shocks enter the *same-period* solve, so they spill over to neighbours
immediately:

* **clamp**: a set :math:`C` of nodes is forced onto given price paths. In growth terms each such
  node is forced to the growth :math:`c` that reaches its path, and the free nodes :math:`F` solve
  :math:`(I - \rho W_{FF})\, z_F = r_F + \rho\, W_{FC}\, c`.
* **impulse**: an innovation :math:`s` is added to :math:`r_t` so that each shocked node
  :math:`j \in J` has growth :math:`\delta_j` above the unshocked solution, which moves its price
  by exactly :math:`\delta_j`: :math:`z = A^{-1} r + A^{-1}_{\cdot J}\, s`, with
  :math:`A^{-1}_{JJ}\, s = \delta`. Later periods grow on from the new level, so the shift persists
  unless the dynamics reverse it.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from housing_star._types import FloatArray
from housing_star.models._growth import (
    one_step_levels,
    path_levels,
    relative_growth,
    relative_growth_of_path,
)
from housing_star.models.star.estimation import StarFit


def _rhs(fit: StarFit, W: FloatArray, lags: list[FloatArray]) -> FloatArray:
    """Right-hand side for lags ``[z_{t-1}, ..., z_{t-p}]``; columns may be stacked over time."""
    out = np.zeros_like(lags[0])
    for ell, z_lag in enumerate(lags):
        out = out + fit.phi[ell] * z_lag + fit.theta[ell] * (W @ z_lag)
    return out


def one_step_predictions(
    y: FloatArray, W: FloatArray, fit: StarFit, growth: FloatArray
) -> FloatArray:
    """Predict each column of ``y`` from the *observed* previous periods.

    The first ``p + 1`` columns are NaN: a prediction needs the previous price and ``p`` lags of
    growth.
    """
    N, T = y.shape
    p = len(fit.phi)
    z = relative_growth(y, growth)
    z_hat = np.full((N, T - 1), np.nan)
    if p < T - 1:
        lags = [z[:, p - ell : T - 1 - ell] for ell in range(1, p + 1)]
        z_hat[:, p:] = np.linalg.solve(np.eye(N) - fit.rho * W, _rhs(fit, W, lags))
    return one_step_levels(y, growth, z_hat)


def forecast(
    y: FloatArray,
    W: FloatArray,
    fit: StarFit,
    growth: FloatArray,
    horizon: int,
    *,
    clamp: Mapping[int, FloatArray] | None = None,
    impulse: Mapping[int, FloatArray] | None = None,
) -> FloatArray:
    """Recursively forecast ``horizon`` columns beyond ``y``, feeding forecasts back as lags.

    Parameters
    ----------
    clamp : ``{node: (horizon,) path}`` of log price levels (same units as ``y``) the node is
        forced onto.
    impulse : ``{node: (horizon,) offsets}``; node's price moves by exactly this much relative to
        what the dynamics would give it that period, and later periods grow on from there.
        Clamped nodes ignore impulses.

    Raises
    ------
    ValueError
        If ``horizon < 1``, ``y`` has fewer than ``p + 1`` periods, or a node index or path
        length is invalid.
    """
    N, T = y.shape
    p = len(fit.phi)
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1; got {horizon}.")
    if p > T - 1:
        raise ValueError(
            f"Need at least p+1={p + 1} observed periods to seed the forecast; got {T}."
        )
    clamp, impulse = dict(clamp or {}), dict(impulse or {})
    for name, paths in (("clamp", clamp), ("impulse", impulse)):
        for node, path in paths.items():
            if not 0 <= node < N or len(path) != horizon:
                raise ValueError(
                    f"Invalid {name} entry for node {node}: need 0 <= node < {N} and {horizon} values."
                )

    z = relative_growth(y, growth)
    forced = sorted(clamp)
    free = [i for i in range(N) if i not in clamp]
    position = {node: k for k, node in enumerate(free)}
    shocked = [node for node in sorted(impulse) if node in position]
    pos = [position[node] for node in shocked]
    forced_z = {i: relative_growth_of_path(y[i, -1], growth[i], clamp[i]) for i in forced}

    B = np.linalg.inv(np.eye(len(free)) - fit.rho * W[np.ix_(free, free)])
    spill = fit.rho * W[np.ix_(free, forced)]
    B_shock = B[:, pos]
    B_diag = B[np.ix_(pos, pos)]

    history = [z[:, T - 1 - ell] for ell in range(1, p + 1)]  # z_{t-1}, ..., z_{t-p}
    out = np.empty((N, horizon))
    for h in range(horizon):
        values = np.array([forced_z[i][h] for i in forced])
        z_free = B @ (_rhs(fit, W, history)[free] + spill @ values)
        if shocked:
            offsets = np.array([impulse[i][h] for i in shocked])
            z_free = z_free + B_shock @ np.linalg.solve(B_diag, offsets)
        column = np.empty(N)
        column[free], column[forced] = z_free, values
        out[:, h] = column
        history = [column, *history[:-1]]
    return path_levels(y[:, -1], growth, out)
