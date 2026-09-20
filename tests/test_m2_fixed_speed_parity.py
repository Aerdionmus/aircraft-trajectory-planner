"""M3-REQ-010: fixed-speed parity, the property that replaces monotone
refinement as Milestone 3's backward-compatibility guard.

Milestone 2 could assert "no Milestone 2 edge is cheaper than its Milestone 1
value" and inherit every admissibility claim from it.  Milestone 3 cannot: the
whole point of a speed decision is that a faster option may cost less.  The
regression property that replaces it is stated as an *equivalence* rather than
an inequality:

    If the speed envelope contains exactly the Milestone 2 cruise speed, the
    Milestone 3 planner must reproduce Milestone 2 exactly.

"Exactly" is meant literally.  These tests assert bit-level equality of cost,
metrics, path and expansion counts rather than a tolerance, because the
reduction is *structural* -- one speed option means one iteration of the speed
loop and a speed factor that is short-circuited to 1.0 -- and anything less than
exact equality would mean the structural claim is false and only a numerical
coincidence is holding it up.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from atp.aircraft.envelope import SpeedEnvelope
from atp.aircraft.performance import MEDIUM_TWIN_JET
from atp.core.geometry import Vec2
from atp.environment.airspace import GridState
from atp.environment.risk import GaussianHazard
from atp.environment.wind import UniformWind, VortexWind
from atp.evaluation.metrics import evaluate_trajectory
from atp.experiments.runner import run_baseline, run_plan
from atp.planning.astar import astar, dijkstra
from atp.planning.cost import CostWeights
from atp.planning.heuristics import OptimisticCostHeuristic, build_heuristic
from atp.planning.state import FlightState
from atp.scenarios.library import get_scenario
from atp.scenarios.spec import build_scenario
from conftest import make_problem

#: The Milestone 2 operating point, as a Milestone 3 envelope.
M2_ENVELOPE = SpeedEnvelope.fixed(450.0)

WEIGHTS = CostWeights(
    time_cost_per_hour=3000.0,
    fuel_cost_per_kg=0.8,
    distance_cost_per_nm=2.0,
    risk_cost_per_exposure=5000.0,
)

ENVIRONMENTS = {
    "still-air": {},
    "wind": {"wind": UniformWind(Vec2(70.0, -30.0))},
    "vortex": {"wind": VortexWind(Vec2(40.0, 40.0), 90.0, 30.0)},
    "hazard": {"risk": GaussianHazard(Vec2(40.0, 40.0), peak=2.0, sigma_nm=20.0)},
}

TURN_MODELS = ["none", "gate", "gate+cost"]


def _pair(env_name, turn_model, **kw):
    """The same problem twice: once as Milestone 2, once with the fixed-speed
    Milestone 3 envelope attached."""
    common = dict(
        cells=6,
        cell_size_nm=12.0,
        weights=WEIGHTS,
        turn_model=turn_model,
        **ENVIRONMENTS[env_name],
        **kw,
    )
    return make_problem(**common), make_problem(**common, speed_envelope=M2_ENVELOPE)


# -- the structural claim ----------------------------------------------------
def test_an_aircraft_with_no_envelope_already_has_the_fixed_one():
    """Parity does not depend on a scenario opting in: the *default* is the
    singleton envelope, so an untouched Milestone 1 or 2 configuration cannot
    accidentally acquire a speed decision."""
    assert MEDIUM_TWIN_JET.speed_envelope is None
    assert MEDIUM_TWIN_JET.envelope.is_fixed
    assert MEDIUM_TWIN_JET.envelope.planning_tas_kt == (450.0,)
    assert MEDIUM_TWIN_JET.envelope.cruise_tas_kt == MEDIUM_TWIN_JET.cruise_tas_kt


@pytest.mark.parametrize("turn_model", TURN_MODELS)
def test_a_fixed_envelope_reports_no_speed_decision(turn_model):
    m2, m3 = _pair("still-air", turn_model)
    assert not m2.speed_aware and not m3.speed_aware
    assert m2.cost_model.models_speed is False
    assert m3.cost_model.models_speed is False
    assert m2.uses_flight_state == m3.uses_flight_state


# -- transition-level parity -------------------------------------------------
@pytest.mark.parametrize("env_name", sorted(ENVIRONMENTS))
@pytest.mark.parametrize("turn_model", TURN_MODELS)
def test_every_successor_is_identical_state_for_state(env_name, turn_model):
    m2, m3 = _pair(env_name, turn_model, start_heading=2)
    for cell in m2.airspace.spec.states():
        state = (
            FlightState(cell.ix, cell.iy, cell.il, 2)
            if m2.uses_flight_state
            else cell
        )
        left = list(m2.successors(state))
        right = list(m3.successors(state))
        assert [t.state for t in left] == [t.state for t in right]
        assert [t.cost for t in left] == [t.cost for t in right], (
            f"edge cost moved at {state} in {env_name}/{turn_model}"
        )


@pytest.mark.parametrize("env_name", sorted(ENVIRONMENTS))
def test_segment_metrics_are_identical_field_for_field(env_name):
    m2, m3 = _pair(env_name, "gate+cost")
    pairs = [
        (GridState(1, 1, 0), GridState(2, 1, 0)),
        (GridState(2, 2, 0), GridState(3, 3, 0)),
        (GridState(4, 3, 0), GridState(3, 4, 0)),
    ]
    for a, b in pairs:
        left = m2.cost_model.evaluate(a, b)
        right = m3.cost_model.evaluate(a, b, 0)
        for field in (
            "feasible",
            "ground_distance_nm",
            "vertical_ft",
            "time_h",
            "fuel_kg",
            "risk_exposure",
            "restriction_penalty",
            "ground_speed_kt",
            "tas_kt",
        ):
            assert getattr(left, field) == getattr(right, field), field


@pytest.mark.parametrize("env_name", sorted(ENVIRONMENTS))
def test_turn_metrics_are_identical(env_name):
    m2, m3 = _pair(env_name, "gate+cost")
    node = GridState(3, 3, 0)
    east = Vec2(1.0, 0.0)
    diagonal = Vec2(1.0, 1.0).normalized()
    leg = m2.turn_table.cell_size_nm
    left = m2.cost_model.turn_metrics(node, east, diagonal, leg, leg)
    right = m3.cost_model.turn_metrics(
        node, east, diagonal, leg, leg, speed_index_in=0, speed_index_out=0
    )
    assert left == right


# -- heuristic parity --------------------------------------------------------
@pytest.mark.parametrize("env_name", sorted(ENVIRONMENTS))
def test_the_lower_bound_and_offset_are_bit_identical(env_name):
    """The bound was re-derived as a minimisation over the envelope.  Over a
    one-element envelope the minimisation must collapse to exactly the
    Milestone 2 quotient -- not to a value that merely rounds to it."""
    m2, m3 = _pair(env_name, "gate+cost")
    assert (
        m3.cost_model.speed_cost_lower_bound_per_nm()
        == m2.cost_model.speed_cost_lower_bound_per_nm()
    )
    assert m3.cost_model.constant_cost_offset() == m2.cost_model.constant_cost_offset()


@pytest.mark.parametrize("env_name", sorted(ENVIRONMENTS))
def test_heuristic_values_match_at_every_state(env_name):
    m2, m3 = _pair(env_name, "gate+cost")
    left, right = OptimisticCostHeuristic(m2), OptimisticCostHeuristic(m3)
    for cell in m2.airspace.spec.states():
        assert left(cell) == right(cell)


# -- search parity -----------------------------------------------------------
@pytest.mark.parametrize("env_name", sorted(ENVIRONMENTS))
@pytest.mark.parametrize("turn_model", TURN_MODELS)
def test_the_search_returns_the_same_path_and_expands_the_same_nodes(
    env_name, turn_model
):
    """Expansion counts are included deliberately.  Equal cost with different
    expansion counts would mean the two are searching different graphs and
    arriving at the same answer by luck."""
    m2, m3 = _pair(env_name, turn_model, start_heading=2)
    left = astar(m2, build_heuristic("optimistic", m2))
    right = astar(m3, build_heuristic("optimistic", m3))
    assert left.status == right.status
    assert left.cost == right.cost
    assert [tuple(s) for s in left.path] == [tuple(s) for s in right.path]
    assert left.statistics.expansions == right.statistics.expansions
    assert left.statistics.generated == right.statistics.generated
    assert left.statistics.reopened == right.statistics.reopened


@pytest.mark.parametrize("env_name", sorted(ENVIRONMENTS))
def test_dijkstra_parity(env_name):
    m2, m3 = _pair(env_name, "gate+cost", start_heading=2)
    left, right = dijkstra(m2), dijkstra(m3)
    assert left.cost == right.cost
    assert left.statistics.expansions == right.statistics.expansions


@pytest.mark.parametrize("env_name", sorted(ENVIRONMENTS))
def test_evaluation_parity_over_the_returned_trajectory(env_name):
    m2, m3 = _pair(env_name, "gate+cost", start_heading=2)
    path = astar(m3, build_heuristic("optimistic", m3)).path
    left = evaluate_trajectory(m2.cost_model, path, start_move=m2.start_move)
    right = evaluate_trajectory(m3.cost_model, path, start_move=m3.start_move)
    for key, value in left.as_dict().items():
        assert right.as_dict()[key] == value, key
    assert right.speed_changes == 0
    assert right.mean_tas_kt == pytest.approx(450.0, rel=1e-12)


# -- scenario-level parity ---------------------------------------------------
def test_the_fixed_speed_scenario_differs_from_turn_limited_only_by_the_envelope():
    """Controlled comparison: anything that moves between these two is caused by
    the speed machinery and nothing else."""
    base = get_scenario("turn-limited")
    fixed = get_scenario("turn-limited-fixed-speed")
    for field in (
        "grid",
        "start",
        "goal",
        "aircraft",
        "start_heading_deg",
        "goal_heading_deg",
        "turn_model",
        "max_bank_deg",
        "max_turn_deg",
        "restrictions",
        "risk",
        "wind",
        "weights",
        "match_goal_level",
        "integration_samples",
    ):
        assert getattr(base, field) == getattr(fixed, field), field
    assert base.speed_envelope is None
    assert fixed.speed_envelope is not None
    assert fixed.speed_envelope["planning_tas_kt"] == [450.0]


@pytest.mark.parametrize("heuristic", ["zero", "optimistic", "octile"])
def test_the_fixed_speed_scenario_reproduces_turn_limited_exactly(heuristic):
    base = run_plan(build_scenario(get_scenario("turn-limited")), heuristic=heuristic)
    fixed = run_plan(
        build_scenario(get_scenario("turn-limited-fixed-speed")), heuristic=heuristic
    )
    assert base.status == fixed.status
    assert base.search_cost == fixed.search_cost
    assert [tuple(s) for s in base.path] == [tuple(s) for s in fixed.path]
    assert base.statistics.expansions == fixed.statistics.expansions
    assert base.statistics.generated == fixed.statistics.generated

    left, right = base.evaluation.as_dict(), fixed.evaluation.as_dict()
    # The Milestone 3 columns are new; every column they share must be equal.
    new_columns = {"mean_tas_kt", "min_tas_kt", "max_tas_kt", "speed_changes"}
    for key, value in left.items():
        if key in new_columns:
            continue
        assert right[key] == value, key


def test_the_baselines_are_unaffected_by_the_fixed_envelope():
    for analytic in (False, True):
        base = run_baseline(
            build_scenario(get_scenario("turn-limited")), analytic=analytic
        )
        fixed = run_baseline(
            build_scenario(get_scenario("turn-limited-fixed-speed")),
            analytic=analytic,
        )
        assert base.evaluation.distance_nm == fixed.evaluation.distance_nm
        assert base.evaluation.fuel_kg == fixed.evaluation.fuel_kg
        assert base.search_cost == fixed.search_cost


def test_a_singleton_envelope_at_a_different_speed_is_not_claimed_to_be_parity():
    """Guard against the parity property being read too widely.  Parity is a
    statement about the *Milestone 2 speed*, not about any fixed speed: pinning
    a different one legitimately changes the answer."""
    _, at_450 = _pair("wind", "gate+cost", start_heading=2)
    at_390 = make_problem(
        cells=6,
        cell_size_nm=12.0,
        weights=WEIGHTS,
        turn_model="gate+cost",
        start_heading=2,
        speed_envelope=SpeedEnvelope.fixed(390.0),
        **ENVIRONMENTS["wind"],
    )
    left = astar(at_450, build_heuristic("optimistic", at_450))
    right = astar(at_390, build_heuristic("optimistic", at_390))
    assert left.solved and right.solved
    assert left.cost != pytest.approx(right.cost, rel=1e-6)


def test_milestone_1_scenarios_still_plan_over_grid_states():
    """The strongest form of the structural claim: with no envelope declared,
    the Milestone 1 code path is what runs."""
    for name in ("empty-cruise", "two-no-fly", "jetstream", "convective-risk"):
        built = build_scenario(get_scenario(name))
        assert not built.problem.speed_aware
        assert not built.problem.uses_flight_state
        assert isinstance(built.problem.initial_state(), GridState)
        assert math.isfinite(
            built.cost_model.speed_cost_lower_bound_per_nm()
        )
