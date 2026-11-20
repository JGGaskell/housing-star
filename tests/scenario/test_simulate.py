import numpy as np
import pytest

from housing_star.models.ar import ArModel
from housing_star.models.star.estimation import StarFit
from housing_star.models.star.model import StarModel, UnstableModelWarning
from housing_star.scenario.config import NodeSelector, Perturbation, PerturbationStep, Scenario
from housing_star.scenario.simulate import simulate


def scenario(*perts, horizon=4, target="PC0"):
    return Scenario(target, horizon, tuple(perts))


def clamp(node, *pcts, name="c"):
    steps = tuple(PerturbationStep(i + 1, p) for i, p in enumerate(pcts))
    return Perturbation(name, NodeSelector(postcode=node), steps, mode="clamp")


def impulse(node, *pcts, name="i"):
    steps = tuple(PerturbationStep(i + 1, p) for i, p in enumerate(pcts))
    return Perturbation(name, NodeSelector(postcode=node), steps, mode="impulse")


@pytest.fixture
def model(star_fit, ds):
    return StarModel.from_fit(star_fit, ds.growth_means)


def test_without_perturbations_the_forecasts_coincide(ds, model):
    result = simulate(model, ds, scenario())
    np.testing.assert_allclose(result.perturbed, result.baseline)
    np.testing.assert_allclose(result.target_effect(), 0.0, atol=1e-9)


def test_clamped_nodes_hit_the_requested_multiplier_and_compound(ds, model):
    result = simulate(model, ds, scenario(clamp("PC2", 5, 15)))
    ratio = result.perturbed_abs()[2] / result.baseline_abs()[2]
    np.testing.assert_allclose(ratio, [1.05, 1.05 * 1.15, 1.05 * 1.15, 1.05 * 1.15])


def test_clamping_a_neighbour_moves_the_target_in_the_first_period(ds, model):
    result = simulate(model, ds, scenario(clamp("PC1", 10)))
    assert result.target_effect()[0] > 0


def test_impulse_moves_the_node_and_the_new_price_level_persists(ds, model):
    effect = simulate(model, ds, scenario(impulse("PC2", 10))).effect_pct()[2]
    assert effect[0] == pytest.approx(10.0)
    assert np.all(effect > 8.0)  # the node grows on from its new level instead of reverting


def test_without_dynamics_an_impulse_shifts_the_level_by_exactly_the_shock(ds):
    flat = StarModel.from_fit(StarFit(0.0, np.zeros(1), np.zeros(1), 0.01, 0.0), ds.growth_means)
    result = simulate(flat, ds, scenario(impulse("PC2", 10)))
    np.testing.assert_allclose(result.effect_pct()[2], 10.0)
    np.testing.assert_allclose(np.delete(result.effect_pct(), 2, axis=0), 0.0, atol=1e-9)


def test_impulse_spills_over_to_the_target_in_the_same_period(ds, model):
    assert simulate(model, ds, scenario(impulse("PC1", 10))).target_effect()[0] > 0


def test_multiple_shocks_on_one_node_accumulate(ds, model):
    both = simulate(model, ds, scenario(clamp("PC2", 5), clamp("PC2", 5, name="again")))
    np.testing.assert_allclose(both.perturbed_abs()[2] / both.baseline_abs()[2], 1.05**2)


def test_nodes_can_be_selected_by_proximity(ds, model):
    pert = Perturbation("near", NodeSelector(nth_closest=1), (PerturbationStep(1, 5),))
    result = simulate(model, ds, scenario(pert))
    assert result.effect_pct()[1, 0] == pytest.approx(5.0)


def test_without_spatial_effects_the_target_is_unaffected(ds, star_fit):
    flat = StarModel.from_fit(StarFit(0.0, star_fit.phi, np.zeros(1), 0.01, 0.0), ds.growth_means)
    result = simulate(flat, ds, scenario(clamp("PC1", 20), impulse("PC2", -10)))
    np.testing.assert_allclose(result.target_effect(), 0.0, atol=1e-9)


def test_result_frame_is_tidy(ds, model):
    frame = simulate(model, ds, scenario(clamp("PC1", 5), horizon=3)).to_frame()
    assert frame.height == ds.N * 3
    assert frame.columns == [
        "postcode",
        "is_target",
        "step",
        "baseline_abs",
        "perturbed_abs",
        "pct_effect",
    ]
    assert frame.filter(frame["is_target"])["postcode"].unique().to_list() == ["PC0"]
    assert sorted(frame["step"].unique().to_list()) == [1, 2, 3]


def test_result_exposes_shape_and_horizon(ds, model):
    result = simulate(model, ds, scenario(horizon=5))
    assert result.horizon == 5 and result.effect_pct().shape == (ds.N, 5)


def test_only_star_models_can_be_simulated(ds):
    with pytest.raises(TypeError, match="StarModel"):
        simulate(ArModel().fit(ds), ds, scenario())  # type: ignore[arg-type]


def test_unknown_targets_are_reported(ds, model):
    with pytest.raises(ValueError, match="Target"):
        simulate(model, ds, scenario(target="ZZ"))


def test_simulating_an_unstable_model_warns(ds):
    unstable = StarModel.from_fit(
        StarFit(0.0, np.array([1.1]), np.zeros(1), 0.01, 0.0), ds.growth_means
    )
    with pytest.warns(UnstableModelWarning):
        simulate(unstable, ds, scenario())


def test_stable_models_simulate_without_warnings(ds, model, recwarn):
    simulate(model, ds, scenario())
    assert not recwarn
