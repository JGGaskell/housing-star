import pytest

from housing_star.config import CleaningConfig, GraphConfig


@pytest.mark.parametrize("kwargs", [{"radius_m": 0}, {"eps_m": -1}, {"max_nodes": 1}])
def test_graph_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        GraphConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs", [{"min_coverage": 0}, {"min_coverage": 1.5}, {"min_nodes": 1}, {"max_gap": -1}]
)
def test_cleaning_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        CleaningConfig(**kwargs)


def test_defaults_are_valid():
    assert GraphConfig().max_nodes == 20
    assert CleaningConfig().keep_extrapolated is False
