r"""Maximum-likelihood estimation of the STAR(p) model.

The model for the growth :math:`z_t \in \mathbb{R}^N` of ``N`` neighbouring price series, each net
of its own average growth (see :mod:`housing_star.models._growth`), is

.. math::

    (I - \rho W)\, z_t = \sum_{\ell=1}^{p} (\phi_\ell\, z_{t-\ell}
        + \theta_\ell\, W z_{t-\ell}) + \varepsilon_t, \qquad \varepsilon_t \sim N(0, \sigma^2 I).

There is no intercept: the average growth has already been removed.

Given :math:`\rho`, the remaining coefficients solve an ordinary least-squares problem, so
:math:`\rho` is found by maximising the *concentrated* (profile) log-likelihood

.. math:: \ell(\rho) = (T-p)\log\lvert I - \rho W\rvert - \tfrac{M}{2}\bigl(\log(2\pi\hat\sigma^2(\rho)) + 1\bigr),

with :math:`M = N(T-p)`. A coarse grid locates the optimum and golden-section search refines it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from housing_star._types import FloatArray

_GOLDEN = (np.sqrt(5.0) + 1.0) / 2.0


@dataclass(frozen=True)
class StarFit:
    """Estimated STAR(p) parameters."""

    rho: float
    phi: FloatArray
    theta: FloatArray
    sigma2: float
    loglik: float


def persistence(fit: StarFit, W: FloatArray) -> float:
    r"""How slowly a shock to growth dies out: the spectral radius of the STAR(p) companion matrix.

    For :math:`p=1` this is the largest absolute eigenvalue of
    :math:`(I-\rho W)^{-1}(\phi_1 I + \theta_1 W)`. Below 1 a shock fades; at 1 or above it
    persists or grows, and forecasts accelerate.
    """
    N, p = W.shape[0], len(fit.phi)
    A_inv = np.linalg.inv(np.eye(N) - fit.rho * W)
    companion = np.zeros((N * p, N * p))
    companion[:N, :] = np.hstack(
        [A_inv @ (fit.phi[k] * np.eye(N) + fit.theta[k] * W) for k in range(p)]
    )
    companion[N:, :-N] = np.eye(N * (p - 1))
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def _vec(A: FloatArray) -> FloatArray:
    """Stack the columns of ``A`` into one vector."""
    return A.reshape(-1, order="F")


def golden_section_maximise(
    f: Callable[[float], float], a: float, b: float, *, tol: float = 1e-6, max_iter: int = 200
) -> float:
    """Maximise a unimodal ``f`` on ``[a, b]``; returns the maximiser."""
    c, d = b - (b - a) / _GOLDEN, a + (b - a) / _GOLDEN
    fc, fd = f(c), f(d)
    iterations = 0
    while abs(b - a) >= tol and iterations < max_iter:
        iterations += 1
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - (b - a) / _GOLDEN
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) / _GOLDEN
            fd = f(d)
    return (a + b) / 2.0


def fit_star(
    Y: FloatArray,
    W: FloatArray,
    p: int,
    *,
    rho_bounds: tuple[float, float] = (-0.95, 0.95),
    grid_size: int = 81,
) -> StarFit:
    """Fit STAR(p) to relative growth ``Y`` (``(N, T)``) with spatial weights ``W`` (``(N, N)``).

    Raises
    ------
    ValueError
        On inconsistent shapes, too few nodes or periods, or a degenerate likelihood.
    """
    N, T = Y.shape
    if W.shape != (N, N):
        raise ValueError(f"W must be ({N}, {N}) to match Y; got {W.shape}.")
    if N < 2:
        raise ValueError("STAR requires at least 2 nodes.")
    if p < 1 or grid_size < 3:
        raise ValueError("p must be >= 1 and grid_size >= 3.")
    n_obs = N * (T - p)
    if p >= T or n_obs <= 2 * p:
        raise ValueError(f"Training window too short (T={T}) for STAR(p={p}) with N={N} nodes.")

    Y0 = Y[:, p:]
    lags = [Y[:, p - ell : T - ell] for ell in range(1, p + 1)]
    Z = np.column_stack([*(_vec(L) for L in lags), *(_vec(W @ L) for L in lags)])

    # The dependent variable is vec(Y0) - rho * vec(W Y0), linear in rho. Regress both parts on Z
    # once; then beta(rho), the residuals and sigma^2(rho) are available in closed form.
    targets = np.column_stack([_vec(Y0), _vec(W @ Y0)])
    coef = np.linalg.lstsq(Z, targets, rcond=None)[0]
    resid = targets - Z @ coef

    def profile(rho: float) -> tuple[float, FloatArray, float]:
        sign, logdet = np.linalg.slogdet(np.eye(N) - rho * W)
        r = resid[:, 0] - rho * resid[:, 1]
        sigma2 = float(r @ r / n_obs)
        if sign <= 0 or sigma2 <= 0:
            return -np.inf, coef[:, 0] - rho * coef[:, 1], sigma2
        llf = (T - p) * logdet - 0.5 * n_obs * (np.log(2.0 * np.pi * sigma2) + 1.0)
        return float(llf), coef[:, 0] - rho * coef[:, 1], sigma2

    grid = np.linspace(rho_bounds[0], rho_bounds[1], grid_size)
    llfs = np.array([profile(r)[0] for r in grid])
    if not np.isfinite(llfs).any():
        raise ValueError("The STAR likelihood is degenerate for every rho on the grid.")
    j = int(np.argmax(llfs))
    rho = golden_section_maximise(
        lambda r: profile(r)[0], grid[max(j - 1, 0)], grid[min(j + 1, grid_size - 1)]
    )
    llf, beta, sigma2 = profile(rho)
    return StarFit(
        rho=float(rho),
        phi=beta[:p].astype(float),
        theta=beta[p:].astype(float),
        sigma2=sigma2,
        loglik=llf,
    )
