import numpy as np

from housing_star.spatial.weights import (
    inverse_distance_weights,
    pairwise_distances,
    row_standardise,
)


def test_pairwise_distances_are_euclidean_and_symmetric():
    D = pairwise_distances(np.array([0.0, 3.0]), np.array([0.0, 4.0]))
    np.testing.assert_allclose(D, [[0, 5], [5, 0]])


def test_row_standardise_normalises_rows_and_keeps_zero_rows():
    W = row_standardise(np.array([[0.0, 2.0, 2.0], [0.0, 0.0, 0.0], [1.0, 0.0, 3.0]]))
    np.testing.assert_allclose(W.sum(axis=1), [1.0, 0.0, 1.0])


def test_inverse_distance_weights_decay_with_distance_and_have_zero_diagonal():
    W = inverse_distance_weights(np.array([0.0, 100.0, 1_000.0]), np.zeros(3), eps_m=10.0)
    assert np.all(np.diag(W) == 0)
    np.testing.assert_allclose(W.sum(axis=1), 1.0)
    assert W[0, 1] > W[0, 2]
