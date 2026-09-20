"""M3-REQ-011 / 012 / 014 / 016: the Milestone 3 experiments and closure checks.

Three distinct properties are asserted here and the milestone brief is explicit
that they must not be conflated:

* **speed non-degeneracy** -- a cost-sensitivity result.  Multiple legal speeds
  exist, and which one is selected changes time, fuel, turn characteristics and
  total cost.  Nothing about feasibility changes.
* **the feasibility frontier** -- a feasibility result.  Changing speed changes
  whether a turn can be flown at all, at a threshold derived analytically and
  then *measured* from the implementation.
* **determinism** -- repeated identical runs reproduce the state sequence, the
  cost, the speed sequence and the evaluation metrics.

The scenarios are built so that each isolates one of these: ``speed-choice``
uses 12 NM cells where every speed can fly every turn, so feasibility is
constant across the decision; ``speed-turn-frontier`` uses 5 NM cells where it
is not.
"""

from __future__ import annotations

import ast
import json
import math
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import pytest

from atp.experiments.runner import (
    NON_DETERMINISTIC_COLUMNS,
    ExperimentSpec,
    run_matrix,
    run_plan,
)
from atp.scenarios.library import SCENARIO_LIBRARY, get_scenario
from atp.scenarios.spec import ScenarioSpec, build_scenario

CONFIGS = Path(__file__).parent.parent / "configs"
SPEEDS = [330.0, 360.0, 390.0, 420.0, 450.0, 480.0]


def _fixed(spec: ScenarioSpec, tas_kt: float) -> ScenarioSpec:
    """The same scenario pinned to one speed, keeping its operating limits."""
    return replace(
        spec,
        speed_envelope=dict(
            spec.speed_envelope, planning_tas_kt=[tas_kt], cruise_tas_kt=tas_kt
        ),
        start_speed_kt=tas_kt,
    )


def _run_uncached(spec: ScenarioSpec):
    return run_plan(build_scenario(spec), heuristic="optimistic", max_expansions=400000)


@lru_cache(maxsize=None)
def _run_named(name: str, tas_kt: float | None):
    """Plan a library scenario, optionally pinned to one speed.

    Cached by ``(name, tas_kt)``.  Caching is only sound because these runs are
    deterministic, which is itself asserted below by
    ``test_repeated_runs_reproduce_states_cost_speeds_and_metrics`` -- and that
    test deliberately bypasses this cache, or it would be asserting that a
    dictionary returns the same value twice.

    A free-choice run on a six-speed envelope takes several seconds (the
    branching factor is six times the Milestone 2 one), so re-planning the same
    instance for each of a dozen assertions would dominate the suite.
    """
    spec = get_scenario(name)
    return _run_uncached(spec if tas_kt is None else _fixed(spec, tas_kt))


# -- registration ------------------------------------------------------------
@pytest.mark.parametrize(
    "name",
    [
        "speed-choice",
        "speed-choice-min-time",
        "speed-choice-min-fuel",
        "speed-turn-frontier",
        "turn-limited-fixed-speed",
    ],
)
def test_the_milestone_3_scenarios_are_registered_and_speed_aware(name):
    assert name in SCENARIO_LIBRARY
    spec = get_scenario(name)
    assert spec.speed_envelope is not None
    assert spec.schema_version == 3
    built = build_scenario(spec)
    assert built.aircraft.speed_envelope is not None


def test_the_three_speed_choice_variants_differ_only_in_their_weights():
    """Controlled comparison: the selected speed must be attributable to the
    objective and to nothing else."""
    specs = [
        get_scenario(n)
        for n in (
            "speed-choice",
            "speed-choice-min-time",
            "speed-choice-min-fuel",
        )
    ]
    reference = specs[0]
    for other in specs[1:]:
        for field in (
            "grid",
            "start",
            "goal",
            "aircraft",
            "start_heading_deg",
            "turn_model",
            "restrictions",
            "risk",
            "wind",
            "speed_envelope",
            "start_speed_kt",
        ):
            assert getattr(reference, field) == getattr(other, field), field
        assert reference.weights != other.weights


# -- M3-REQ-011: speed non-degeneracy ----------------------------------------
def test_every_speed_is_legal_on_the_cost_scenario():
    """``speed-choice`` must not smuggle a feasibility effect into a cost
    comparison, so every option has to be flyable on its grid."""
    spec = get_scenario("speed-choice")
    for tas in SPEEDS:
        report = _run_named(spec.name, tas)
        assert report.status == "solved", tas
        assert report.evaluation.num_turns > 0, tas


def test_fixed_speed_arms_differ_in_time_fuel_cost_and_turn_characteristics():
    """The milestone's minimum bar: if every speed gave the same answer, the
    speed decision would be cosmetic."""
    spec = get_scenario("speed-choice")
    rows = {tas: _run_named(spec.name, tas).evaluation for tas in SPEEDS}
    times = [e.time_h for e in rows.values()]
    fuels = [e.fuel_kg for e in rows.values()]
    costs = [e.comparable_cost for e in rows.values()]
    turn_times = [e.turn_time_min for e in rows.values()]

    assert len(set(times)) == len(times)
    assert len(set(fuels)) == len(fuels)
    assert len(set(costs)) == len(costs)
    assert len(set(turn_times)) == len(turn_times)

    # Faster is strictly quicker, and turning at speed takes strictly longer.
    ordered = [rows[t] for t in SPEEDS]
    for slower, faster in zip(ordered, ordered[1:]):
        assert faster.time_h < slower.time_h
        assert faster.turn_time_min > slower.turn_time_min


def test_the_selected_speed_moves_with_the_objective():
    """Minimum time should select the fast end; minimum fuel should select near
    best specific range, which is *not* the slow end by accident but by the
    shape of the power curve."""
    min_time = _run_named("speed-choice-min-time", None).evaluation
    min_fuel = _run_named("speed-choice-min-fuel", None).evaluation
    balanced = _run_named("speed-choice", None).evaluation

    assert min_time.max_tas_kt == max(SPEEDS)
    assert min_time.min_tas_kt == max(SPEEDS)
    assert min_fuel.max_tas_kt < min_time.min_tas_kt
    assert min_time.time_h < min_fuel.time_h
    assert min_fuel.fuel_kg < min_time.fuel_kg
    # The balanced objective lands strictly between the two single-objective
    # extremes, which is the statement that it is a genuine trade-off.
    assert min_fuel.mean_tas_kt <= balanced.mean_tas_kt <= min_time.mean_tas_kt
    assert balanced.mean_tas_kt not in (min(SPEEDS), max(SPEEDS))


def test_free_speed_choice_is_never_worse_than_the_best_fixed_speed():
    """A superset of options cannot raise the optimum.  This is also the
    concrete statement that the Milestone 2 monotone-refinement invariant does
    **not** survive: the free-choice cost is below fixed-speed costs, which
    under ``c3 >= c2`` would be impossible."""
    spec = get_scenario("speed-choice")
    free = _run_named(spec.name, None).evaluation.comparable_cost
    fixed = [_run_named(spec.name, tas).evaluation.comparable_cost for tas in SPEEDS]
    assert free <= min(fixed) + 1e-9
    assert free < max(fixed)


def test_the_planner_actually_uses_more_than_one_speed_somewhere():
    """Non-degeneracy in its strongest form: the optimum is not reproducible by
    any single speed."""
    report = _run_named("speed-turn-frontier", None)
    assert report.status == "solved"
    assert report.evaluation.min_tas_kt < report.evaluation.max_tas_kt
    assert report.evaluation.speed_changes > 0


# -- M3-REQ-012: the feasibility frontier ------------------------------------
def analytic_turn_threshold_kt(cell_nm: float, bank_deg: float, turn_deg: float) -> float:
    """Speed at which a fly-by of ``turn_deg`` stops fitting on a ``cell_nm``
    grid.

    From ``R tan(dpsi/2) <= 0.5 * min(L_in, L_out)`` with
    ``R = V^2 / (g tan phi)``, and ``min(L_in, L_out) = cell`` because on an
    8-connected grid a 45 degree turn always joins an axis leg to a diagonal
    one.  Derived here from the equations, not read back from the code.
    """
    from atp.core.units import G_NM_PER_H2

    return math.sqrt(
        0.5
        * cell_nm
        * G_NM_PER_H2
        * math.tan(math.radians(bank_deg))
        / math.tan(math.radians(turn_deg / 2.0))
    )


def test_the_measured_turn_threshold_matches_the_analytic_one():
    """The threshold is *measured* by bisecting the implementation's own gate
    and then compared with the derivation.  Neither number is asserted from the
    other, which is the whole point."""
    from atp.aircraft.turn import turn_fits, turn_radius_nm

    cell, bank = 5.0, 25.0
    for turn_deg in (45.0, 90.0):
        predicted = analytic_turn_threshold_kt(cell, bank, turn_deg)

        def fits(tas: float) -> bool:
            return turn_fits(
                turn_radius_nm(tas, bank), math.radians(turn_deg), cell, cell
            )

        low, high = 50.0, 2000.0
        assert fits(low) and not fits(high)
        for _ in range(200):
            mid = 0.5 * (low + high)
            if fits(mid):
                low = mid
            else:
                high = mid
        assert low == pytest.approx(predicted, rel=1e-9), turn_deg

    # The numbers the scenario docstring quotes, to the precision it quotes them.
    assert analytic_turn_threshold_kt(5.0, 25.0, 45.0) == pytest.approx(439.5, abs=0.5)
    assert analytic_turn_threshold_kt(5.0, 25.0, 90.0) == pytest.approx(282.8, abs=0.5)


def test_speed_changes_turn_feasibility_on_the_frontier_scenario():
    """The frontier crossed end to end: slow speeds solve the instance, fast
    speeds cannot turn at all and the instance becomes unsolvable."""
    spec = get_scenario("speed-turn-frontier")
    threshold = analytic_turn_threshold_kt(spec.grid.cell_size_nm, 25.0, 45.0)
    statuses = {}
    for tas in SPEEDS:
        report = _run_named(spec.name, tas)
        statuses[tas] = report.status
        if tas < threshold:
            assert report.status == "solved", tas
            assert report.evaluation.num_turns > 0
        else:
            assert report.status == "unsolvable", tas

    assert set(statuses.values()) == {"solved", "unsolvable"}, (
        "the frontier must actually be crossed by the shipped speed set"
    )
    # The Milestone 2 operating point is on the infeasible side here, which is
    # the sharpest statement of the result.
    assert statuses[450.0] == "unsolvable"


def test_the_frontier_is_a_feasibility_result_not_a_cost_result():
    """Guard against the two experiment types being conflated.  On the cost
    scenario every speed is feasible; on the frontier scenario feasibility is
    what moves."""
    cost_spec = get_scenario("speed-choice")
    frontier_spec = get_scenario("speed-turn-frontier")
    cost_statuses = {_run_named(cost_spec.name, t).status for t in SPEEDS}
    frontier_statuses = {_run_named(frontier_spec.name, t).status for t in SPEEDS}
    assert cost_statuses == {"solved"}
    assert frontier_statuses == {"solved", "unsolvable"}


def test_free_choice_solves_the_frontier_instance_the_cruise_speed_cannot():
    """The practical consequence: a speed-aware planner finds a trajectory where
    a fixed-speed one reports the instance unsolvable."""
    spec = get_scenario("speed-turn-frontier")
    assert _run_named(spec.name, 450.0).status == "unsolvable"
    free = _run_named(spec.name, None)
    assert free.status == "solved"
    # It gets there by slowing down for the corners, not by flying slowly
    # throughout.
    assert free.evaluation.max_tas_kt > free.evaluation.min_tas_kt


# -- M3-REQ-014: determinism -------------------------------------------------
@pytest.mark.parametrize(
    "name", ["speed-choice", "speed-turn-frontier", "speed-choice-min-fuel"]
)
def test_repeated_runs_reproduce_states_cost_speeds_and_metrics(name):
    def once():
        # Deliberately uncached: this test exists to re-run the planner.
        report = _run_uncached(get_scenario(name))
        return (
            [tuple(s) for s in report.path],
            report.search_cost,
            [getattr(s, "isp", None) for s in report.path],
            report.evaluation.as_dict(),
        )

    first, second = once(), once()
    assert first == second


def test_a_repeated_experiment_matrix_is_identical_apart_from_wall_clock():
    spec = ExperimentSpec.load(CONFIGS / "experiment_m3_speed_sweep.json")
    first = [r.as_row() for r in run_matrix(spec)]
    second = [r.as_row() for r in run_matrix(spec)]
    assert len(first) == len(second)
    for left, right in zip(first, second):
        for key in left:
            if key in NON_DETERMINISTIC_COLUMNS:
                continue
            assert left[key] == right[key], key


# -- experiment plumbing -----------------------------------------------------
@pytest.mark.parametrize(
    "name",
    [
        "experiment_m3_parity.json",
        "experiment_m3_speed_sweep.json",
        "experiment_m3_objective.json",
        "experiment_m3_turn_frontier.json",
    ],
)
def test_the_shipped_m3_configs_load_and_name_known_scenarios(name):
    spec = ExperimentSpec.load(CONFIGS / name)
    for scenario in spec.scenarios:
        assert scenario in SCENARIO_LIBRARY


def test_the_speed_sets_axis_expands_one_arm_per_set():
    spec = ExperimentSpec.load(CONFIGS / "experiment_m3_speed_sweep.json")
    reports = run_matrix(spec)
    assert len(reports) == len(spec.speed_sets)
    names = [r.scenario for r in reports]
    assert len(set(names)) == len(names), "each arm must be identifiable"
    assert all("sp" in n for n in names)


def test_the_speed_sets_axis_does_not_move_the_fuel_reference():
    """A sweep arm must differ from the free-choice arm only in which speeds are
    selectable, not in how the aircraft burns fuel at a given speed."""
    spec = get_scenario("speed-choice")
    free = build_scenario(spec)
    pinned = build_scenario(_fixed(spec, 330.0))
    assert free.aircraft.reference_tas_kt == pinned.aircraft.reference_tas_kt
    assert free.aircraft.fuel_flow_kg_per_h(
        30000.0, 330.0
    ) == pinned.aircraft.fuel_flow_kg_per_h(30000.0, 330.0)


def test_a_speed_set_on_a_scenario_without_an_envelope_is_refused():
    """Inventing operating limits for a scenario that declares none would be a
    silent fabrication."""
    spec = ExperimentSpec(
        name="bad", scenarios=["turn-limited"], speed_sets=[[450.0]]
    )
    with pytest.raises(ValueError, match="speed_envelope"):
        run_matrix(spec)


def test_speed_columns_are_emitted_in_the_experiment_output():
    reports = run_matrix(
        ExperimentSpec(
            name="m3",
            scenarios=["speed-choice"],
            heuristics=["optimistic"],
            include_baseline=False,
        )
    )
    row = reports[0].as_row()
    for column in ("mean_tas_kt", "min_tas_kt", "max_tas_kt", "speed_changes"):
        assert column in row


# -- M3-REQ-016: architecture ------------------------------------------------
def test_the_generic_search_still_imports_no_aerospace_module():
    """The project's central architectural invariant, asserted rather than
    trusted.  Milestone 3 adds an atmosphere and a speed envelope, which are
    exactly the kind of modules that leak into a search implementation if nobody
    is checking.
    """
    source = (
        Path(__file__).parent.parent / "src" / "atp" / "planning" / "astar.py"
    ).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")

    forbidden = (
        "aircraft",
        "environment",
        "atmosphere",
        "envelope",
        "wind",
        "turn",
        "speeds",
        "performance",
        "kinematics",
        "airspace",
        "cost",
    )
    for module in imported:
        assert not any(word in module for word in forbidden), module
    assert "problem" in imported, (
        "the search must still be written against the SearchProblem protocol"
    )


def test_the_atmosphere_and_speed_modules_sit_in_the_core_layer():
    """Dependency direction: ``core`` may not import from any layer above it."""
    root = Path(__file__).parent.parent / "src" / "atp"
    for name in ("atmosphere.py", "speeds.py"):
        source = (root / "core" / name).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            module = None
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level and node.level > 1:
                    pytest.fail(f"core/{name} reaches outside core: {module}")
            elif isinstance(node, ast.Import):
                module = node.names[0].name
            if module and "atp." in module:
                assert ".core" in module, f"core/{name} imports {module}"


def test_the_speed_envelope_lives_in_the_aircraft_layer_and_uses_only_core():
    root = Path(__file__).parent.parent / "src" / "atp"
    source = (root / "aircraft" / "envelope.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.level:
            assert (node.module or "").startswith(("core", "turn", "kinematics")), (
                node.module
            )


def test_the_results_document_exists_and_declares_its_tags():
    doc = (
        Path(__file__).parent.parent / "docs" / "milestone3_results.md"
    ).read_text(encoding="utf-8")
    for tag in ("[MEASURED]", "[CODE-DERIVED]", "[INTERPRETATION]", "[LIMITATION]"):
        assert tag in doc, tag
    assert "not" in doc.lower() and "certification" in doc.lower()


def test_the_requirements_and_traceability_documents_cover_every_requirement():
    docs = Path(__file__).parent.parent / "docs"
    requirements = (docs / "requirements_m3.md").read_text(encoding="utf-8")
    traceability = (docs / "m3_traceability.md").read_text(encoding="utf-8")
    ids = sorted(
        {
            line.split("`")[1]
            for line in requirements.splitlines()
            if "`M3-REQ-" in line
        }
    )
    assert len(ids) >= 16
    for identifier in ids:
        assert identifier in traceability, identifier
