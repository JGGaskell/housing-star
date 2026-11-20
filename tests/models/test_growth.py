import numpy as np
import pytest

from housing_star.models._growth import (
    fitted_growth,
    one_step_levels,
    path_levels,
    relative_growth,
    relative_growth_of_path,
)

Y = np.array([[0.0, 0.1, 0.4, 0.3], [1.0, 1.0, 1.2, 1.5]])
G = np.array([0.1, 0.2])


def test_relative_growth_is_the_change_net_of_the_average():
    np.testing.assert_allclose(
        relative_growth(Y, G), [[0.0, 0.2, -0.2], [-0.2, 0.0, 0.1]], atol=1e-12
    )


def test_one_step_levels_start_from_the_previous_observation():
    out = one_step_levels(Y, G, np.zeros((2, 3)))
    assert np.isnan(out[:, 0]).all()
    np.testing.assert_allclose(out[:, 1:], Y[:, :-1] + G[:, None])


def test_missing_growth_predictions_stay_missing():
    z_hat = np.array([[np.nan, 0.0, 0.0], [np.nan, 0.0, 0.0]])
    assert np.isnan(one_step_levels(Y, G, z_hat)[:, 1]).all()


def test_levels_and_growth_convert_back_and_forth_exactly():
    z = relative_growth(Y, G)
    np.testing.assert_allclose(path_levels(Y[:, 0], G, z), Y[:, 1:])


def test_a_level_path_converts_to_the_growth_that_reaches_it():
    z = relative_growth_of_path(0.0, 0.1, Y[0, 1:])
    np.testing.assert_allclose(z, relative_growth(Y, G)[0])


def test_growth_must_match_the_dataset_nodes(ds):
    assert fitted_growth(ds.growth_means, ds) is ds.growth_means
    with pytest.raises(ValueError, match="fitted on 2 nodes"):
        fitted_growth(np.zeros(2), ds)
