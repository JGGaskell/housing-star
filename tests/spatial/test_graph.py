import numpy as np
import polars as pl
import pytest

from housing_star.spatial.graph import select_neighbourhood

CENTROIDS = pl.DataFrame(
    {
        "level": ["district"] * 4 + ["sector"],
        "key": ["GL5", "GL6", "GL10", "E1", "GL5 2"],
        "easting": [0.0, 1_000.0, 500.0, 90_000.0, 1.0],
        "northing": [0.0, 0.0, 0.0, 0.0, 1.0],
        "n_units": [1] * 5,
    }
)


def test_nodes_are_sorted_nearest_first_and_limited_by_radius():
    hood = select_neighbourhood("gl5", radius_m=2_000, max_nodes=10, centroids=CENTROIDS)
    assert hood.keys == ["GL5", "GL10", "GL6"]
    np.testing.assert_allclose(hood.eastings, [0, 500, 1_000])


def test_max_nodes_keeps_the_nearest():
    hood = select_neighbourhood("GL5", radius_m=2_000, max_nodes=2, centroids=CENTROIDS)
    assert hood.keys == ["GL5", "GL10"]


def test_only_nodes_at_the_query_level_are_returned():
    hood = select_neighbourhood("GL5 2", radius_m=1e6, max_nodes=10, centroids=CENTROIDS)
    assert hood.keys == ["GL5 2"]


def test_unknown_postcode_raises():
    with pytest.raises(ValueError, match="not found"):
        select_neighbourhood("W1A", radius_m=1_000, max_nodes=5, centroids=CENTROIDS)


def test_bundled_centroids_are_used_by_default():
    hood = select_neighbourhood("GL5", radius_m=15_000, max_nodes=5)
    assert hood.keys[0] == "GL5" and len(hood.keys) == 5
