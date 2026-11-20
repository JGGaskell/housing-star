import pytest

from housing_star.scenario.config import NodeSelector, Perturbation, PerturbationStep, Scenario


def test_selector_needs_exactly_one_field():
    with pytest.raises(ValueError, match="exactly one"):
        NodeSelector()
    with pytest.raises(ValueError, match="exactly one"):
        NodeSelector(postcode="A", nth_closest=1)


def test_selector_rank_must_be_positive():
    with pytest.raises(ValueError, match=">= 1"):
        NodeSelector(nth_closest=0)


def test_selector_resolves_postcodes_including_merged_labels(ds):
    assert NodeSelector(postcode="PC3").resolve(ds, 0) == 3


def test_selector_resolves_proximity_ranks_from_the_target(ds):
    assert NodeSelector(nth_closest=1).resolve(ds, 2) == 1  # tie broken towards lower index
    assert NodeSelector(nth_closest=5).resolve(ds, 0) == 5


def test_selector_rejects_unknown_postcodes_and_ranks_beyond_the_graph(ds):
    with pytest.raises(ValueError, match="not found"):
        NodeSelector(postcode="ZZ").resolve(ds, 0)
    with pytest.raises(ValueError, match="only 5 neighbours"):
        NodeSelector(nth_closest=6).resolve(ds, 0)


def test_step_validation_and_log_offset():
    assert PerturbationStep(1, 10).log_offset == pytest.approx(0.0953101798)
    assert PerturbationStep(1, -50).log_offset < 0
    with pytest.raises(ValueError, match="step"):
        PerturbationStep(0, 5)
    with pytest.raises(ValueError, match="-100"):
        PerturbationStep(1, -100)


def test_perturbation_validation():
    sel, step = NodeSelector(nth_closest=1), PerturbationStep(1, 5)
    with pytest.raises(ValueError, match="mode"):
        Perturbation("p", sel, (step,), mode="shove")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no steps"):
        Perturbation("p", sel, ())
    with pytest.raises(ValueError, match="repeats"):
        Perturbation("p", sel, (step, PerturbationStep(1, 6)))


def test_scenario_validation():
    pert = Perturbation("p", NodeSelector(nth_closest=1), (PerturbationStep(3, 5),))
    with pytest.raises(ValueError, match="horizon"):
        Scenario("PC0", 0, ())
    with pytest.raises(ValueError, match="> horizon"):
        Scenario("PC0", 2, (pert,))


def make_scenario():
    return Scenario(
        "PC0",
        4,
        (
            Perturbation(
                "boom",
                NodeSelector(nth_closest=1),
                (PerturbationStep(1, 5), PerturbationStep(2, 15)),
            ),
            Perturbation(
                "dip", NodeSelector(postcode="PC3"), (PerturbationStep(1, -10),), mode="impulse"
            ),
        ),
    )


def test_dict_and_yaml_round_trips(tmp_path):
    scenario = make_scenario()
    assert Scenario.from_dict(scenario.to_dict()) == scenario
    scenario.to_yaml(tmp_path / "s.yaml")
    assert Scenario.from_yaml(tmp_path / "s.yaml") == scenario


def test_mode_defaults_to_clamp_and_perturbations_to_none():
    data = {
        "target": "A",
        "horizon": 2,
        "perturbations": [
            {"name": "x", "select": {"postcode": "B"}, "steps": [{"step": 1, "change_pct": 5}]}
        ],
    }
    assert Scenario.from_dict(data).perturbations[0].mode == "clamp"
    assert Scenario.from_dict({"target": "A", "horizon": 2}).perturbations == ()


def test_missing_yaml_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Scenario.from_yaml(tmp_path / "absent.yaml")
