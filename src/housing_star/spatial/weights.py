"""Spatial weight matrices."""

from __future__ import annotations

import numpy as np

from housing_star._types import FloatArray


def pairwise_distances(eastings: FloatArray, northings: FloatArray) -> FloatArray:
    """Dense ``(N, N)`` Euclidean distance matrix in the units of the inputs (metres)."""
    de = eastings[:, None] - eastings[None, :]
    dn = northings[:, None] - northings[None, :]
    return np.sqrt(de * de + dn * dn)


def row_standardise(W: FloatArray) -> FloatArray:
    """Scale each row to sum to one (all-zero rows are left as zeros)."""
    sums = W.sum(axis=1, keepdims=True)
    return np.asarray(W / np.where(sums == 0.0, 1.0, sums), dtype=np.float64)


def inverse_distance_weights(
    eastings: FloatArray, northings: FloatArray, *, eps_m: float = 10.0
) -> FloatArray:
    r"""Row-standardised inverse-distance weights with a zero diagonal.

    .. math:: w_{ij} = \frac{1}{d_{ij} + \varepsilon}, \quad w_{ii} = 0,

    where :math:`\varepsilon` (``eps_m``) keeps weights finite for near-coincident centroids.
    """
    D = pairwise_distances(eastings, northings)
    W = 1.0 / (D + eps_m)
    np.fill_diagonal(W, 0.0)
    return row_standardise(W)
