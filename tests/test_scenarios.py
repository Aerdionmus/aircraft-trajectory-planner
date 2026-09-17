from __future__ import annotations

import json

import pytest

from atp.scenarios.library import SCENARIO_LIBRARY, get_scenario, random_scenario
from atp.scenarios.spec import ScenarioError, ScenarioSpec, build_scenario


@pytest.mark.parametrize("name", sorted(SCENARIO_LIBRARY))
def test_every_library_scenario_builds_and_is_well_posed(name):
    spec = get_scenario(name)
    built = build_scenario(spec)
    assert built.problem.start_is_valid(), "origin must not be inside a restriction"
    assert built.problem.goal_is_valid(), "destination must not be inside a restriction"
    assert not built.cost_model.weights.is_degenerate()
    assert spec.description, "scenarios must document what they isolate"


@pytest.mark.parametrize("name", sorted(SCENARIO_LIBRARY))
def test_scenarios_round_trip_through_json(name):
    spec = get_scenario(name)
    restored = ScenarioSpec.from_json(spec.to_json())
    assert restored == spec
    # And the rebuilt airspace is equivalent where it matters.
    assert build_scenario(restored).airspace.spec == build_scenario(spec).airspace.spec


def test_unknown_keys_are_rejected_rather_than_ignored():
    doc = json.loads(get_scenario("empty-cruise").to_json())
    doc["cruise_speed"] = 450
    with pytest.raises(ScenarioError, match="unknown key"):
        ScenarioSpec.from_dict(doc)


def test_missing_required_key_is_reported_with_context():
    doc = json.loads(get_scenario("empty-cruise").to_json())
    del doc["grid"]["cells_x"]
    with pytest.raises(ScenarioError, match="scenario.grid"):
        ScenarioSpec.from_dict(doc)


def test_schema_version_mismatch_is_rejected():
    doc = json.loads(get_scenario("empty-cruise").to_json())
    doc["schema_version"] = 99
    with pytest.raises(ScenarioError, match="schema version"):
        ScenarioSpec.from_dict(doc)


def test_unknown_component_types_are_rejected():
    doc = json.loads(get_scenario("empty-cruise").to_json())
    doc["wind"] = [{"type": "hurricane"}]
    with pytest.raises(ScenarioError, match="unknown wind type"):
        build_scenario(ScenarioSpec.from_dict(doc))


def test_unknown_aircraft_is_rejected():
    doc = json.loads(get_scenario("empty-cruise").to_json())
    doc["aircraft"] = "spaceship"
    with pytest.raises(ScenarioError, match="unknown aircraft"):
        build_scenario(ScenarioSpec.from_dict(doc))


def test_scenario_file_round_trip(tmp_path):
    spec = get_scenario("two-no-fly")
    path = tmp_path / "scenario.json"
    spec.save(path)
    assert ScenarioSpec.load(path) == spec


def test_random_scenarios_are_reproducible_and_feasible_by_construction():
    a = random_scenario(2024)
    b = random_scenario(2024)
    c = random_scenario(2025)
    assert a == b
    assert a != c
    built = build_scenario(a)
    assert built.problem.start_is_valid()
    assert built.problem.goal_is_valid()


def test_random_scenario_does_not_disturb_the_global_rng():
    import random

    random.seed(1)
    expected = [random.random() for _ in range(3)]
    random.seed(1)
    random_scenario(999)
    assert [random.random() for _ in range(3)] == expected
