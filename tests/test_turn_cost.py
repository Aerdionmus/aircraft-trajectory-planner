"""T-4, T-5, T-14, T-33: the cost-model invariants under turn dynamics.

Invariant 4 -- no Milestone 2 edge may be cheaper than its Milestone 1
projection -- is the one the whole inherited-admissibility argument rests on, so
it is checked directly rather than inferred from the heuristics passing.
"""

from __future__ import annotations

import math

import pytest

from atp.core.geometry import Vec2
from atp.environment.airspace import GridState
from atp.environment.restrictions import CircularRestriction, RestrictionSet
from atp.environment.risk import GaussianHazard
from atp.environment.wind import UniformWind, VortexWind
from atp.evaluation.metrics import evaluate_trajectory
from atp.experiments.runner import run_plan
from atp.planning.cost import NO_TURN, CostWeights
from atp.planning.state import NO_HEADING, FlightState, cell_of
from atp.scenarios.library import SCENARIO_LIBRARY, get_scenario
from atp.scenarios.spec import build_scenario
from conftest import make_problem

WEIGHTS = CostWeights(
    time_cost_per_hour=3000.0,
    fuel_cost_per_kg=0.8,
    distance_cost_per_nm=2.0,
    risk_cost_per_exposure=5000.0,
)

ENVIRONMENTS = {
    "still-air": {},
    "wind": {"wind": UniformWind(Vec2(70.0, -30.0))},
    "vortex": {"wind": VortexWind(Vec2(60.0, 60.0), 90.0, 30.0)},
    "hazard": {"risk": GaussianHazard(Vec2(60.0, 60.0), peak=2.0, sigma_nm=20.0)},
    "restricted": {
        "restrictions": RestrictionSet(
            regions=(
                CircularRestriction(
                    region_id="P-1", centre_nm=Vec2(60.0, 60.0), radius_nm=22.0
                ),
            )
        )
    },
}


def _pair(env_name, turn_model):
    common = dict(
        cells=7,
        cell_size_nm=12.0,
        levels=(300, 320),
        weights=WEIGHTS,
        **ENVIRONMENTS[env_name],
    )
    return (
        make_problem(turn_model="none", **common),
        make_problem(turn_model=turn_model, **common),
    )


@pytest.mark.parametrize("env_name", sorted(ENVIRONMENTS))
@pytest.mark.parametrize("turn_model", ["gate", "gate+cost"])
def test_no_milestone_2_edge_is_cheaper_than_its_milestone_1_projection(
    env_name, turn_model
):
    """T-4: invariant 4, edge by edge, exhaustively on a small grid."""
    m1, m2 = _pair(env_name, turn_model)
    m1_costs = {}
    for ix in range(7):
        for iy in range(7):
            for il in range(2):
                state = GridState(ix, iy, il)
                if m1.airspace.is_blocked(state):
                    continue
                for transition in m1.successors(state):
                    m1_costs[(state, transition.state)] = transition.cost

    compared = 0
    for ix in range(7):
        for iy in range(7):
            for il in range(2):
                for ih in range(len(m2.turn_table)):
                    state = FlightState(ix, iy, il, ih)
                    if m2.airspace.is_blocked(cell_of(state)):
                        continue
                    for transition in m2.successors(state):
                        key = (cell_of(state), cell_of(transition.state))
                        assert key in m1_costs, key
                        assert transition.cost >= m1_costs[key] - 1e-9
                        compared += 1
    assert compared > 500


@pytest.mark.parametrize("env_name", sorted(ENVIRONMENTS))
def test_every_edge_cost_is_finite_and_non_negative(env_name):
    """T-5: invariant 1, which Dijkstra depends on outright."""
    _, m2 = _pair(env_name, "gate+cost")
    seen = 0
    for ix in range(7):
        for iy in range(7):
            for ih in list(range(len(m2.turn_table))) + [NO_HEADING]:
                for transition in m2.successors(FlightState(ix, iy, 0, ih)):
                    assert math.isfinite(transition.cost)
                    assert transition.cost >= 0.0
                    assert transition.metrics.turn_time_h >= 0.0
                    assert transition.metrics.turn_fuel_kg >= 0.0
                    assert transition.metrics.corner_cut_nm >= 0.0
                    seen += 1
    assert seen > 200


def test_a_turn_adds_time_fuel_and_risk_but_never_distance():
    """T-4: the specific decision that makes invariant 4 hold -- the fly-by
    shortening is measured but not credited."""
    problem = make_problem(
        cells=7,
        cell_size_nm=12.0,
        weights=WEIGHTS,
        turn_model="gate+cost",
        risk=GaussianHazard(Vec2(30.0, 30.0), peak=2.0, sigma_nm=40.0),
    )
    table = problem.turn_table
    node, nxt = GridState(2, 2, 0), GridState(2, 3, 0)
    straight = problem.cost_model.transition_cost_with_turn(node, nxt, NO_TURN)
    turn = problem.cost_model.turn_metrics(
        node, table.track_unit(1), table.track_unit(0), table.length_nm[1], table.length_nm[0]
    )
    assert turn.feasible and turn.delta_heading_rad > 0.0
    turned = problem.cost_model.transition_cost_with_turn(node, nxt, turn)

    assert turned[1].ground_distance_nm == straight[1].ground_distance_nm
    assert turned[1].time_h > straight[1].time_h
    assert turned[1].fuel_kg > straight[1].fuel_kg
    assert turned[1].risk_exposure > straight[1].risk_exposure
    assert turned[0] > straight[0]
    # The shortening is reported, and it is real, but it is not priced.
    assert turn.corner_cut_nm > 0.0


def test_the_gate_only_model_prunes_without_charging():
    problem_gate = make_problem(cells=7, cell_size_nm=12.0, turn_model="gate")
    problem_cost = make_problem(cells=7, cell_size_nm=12.0, turn_model="gate+cost")
    table = problem_gate.turn_table
    args = (
        GridState(3, 3, 0),
        table.track_unit(1),
        table.track_unit(0),
        table.length_nm[1],
        table.length_nm[0],
    )
    gated = problem_gate.cost_model.turn_metrics(*args)
    charged = problem_cost.cost_model.turn_metrics(*args)
    assert gated.feasible and charged.feasible
    assert gated.delta_heading_rad == pytest.approx(charged.delta_heading_rad)
    assert gated.time_h == 0.0 and gated.fuel_kg == 0.0
    assert charged.time_h > 0.0 and charged.fuel_kg > 0.0


def test_turn_time_does_not_loosen_the_vertical_rate_check():
    """T-14.  The required rate is computed on the straight-leg time only; if
    turn time were folded in first, the aircraft would appear to have longer to
    climb and an unflyable level change would be admitted.

    An 8 NM leg at 450 kt takes 64 s, so a 2000 ft step needs 1875 fpm, above
    the 1800 fpm limit.  A 45 degree turn adds ~40 s, which would drop the
    apparent requirement to about 1160 fpm.
    """
    problem = make_problem(
        cells=7,
        cell_size_nm=8.0,
        levels=(300, 320),
        turn_model="gate+cost",
        weights=WEIGHTS,
    )
    node = GridState(3, 3, 0)
    climbing = GridState(3, 4, 1)
    straight_time_h = 8.0 / 450.0
    assert 2000.0 / (straight_time_h * 60.0) > 1800.0  # the premise

    metrics = problem.cost_model.evaluate(node, climbing)
    assert not metrics.feasible
    assert "fpm" in metrics.infeasible_reason

    table = problem.turn_table
    turn = problem.cost_model.turn_metrics(
        node, table.track_unit(1), table.track_unit(0), table.length_nm[1], table.length_nm[0]
    )
    assert turn.feasible and turn.time_h > 0.0
    cost, combined = problem.cost_model.transition_cost_with_turn(node, climbing, turn)
    assert not combined.feasible
    assert cost == math.inf

    # And the same state is genuinely absent from the successor set.
    successors = {t.state for t in problem.successors(FlightState(3, 3, 0, 1))}
    assert FlightState(3, 4, 1, 0) not in successors


def test_an_infeasible_turn_makes_the_whole_transition_infeasible():
    problem = make_problem(cells=7, cell_size_nm=5.0, turn_model="gate+cost")
    table = problem.turn_table
    turn = problem.cost_model.turn_metrics(
        GridState(3, 3, 0),
        table.track_unit(0),
        table.track_unit(1),
        table.length_nm[0],
        table.length_nm[1],
    )
    assert not turn.feasible
    cost, metrics = problem.cost_model.transition_cost_with_turn(
        GridState(3, 3, 0), GridState(4, 4, 0), turn
    )
    assert cost == math.inf
    assert not metrics.feasible
    assert metrics.infeasible_reason == turn.infeasible_reason


# -- evaluator / search agreement -------------------------------------------
@pytest.mark.parametrize("heuristic", ["zero", "optimistic", "octile", "euclidean"])
def test_evaluator_reproduces_the_search_cost_with_turns_enabled(heuristic):
    """T-33.  The strongest single check that the successor generator and the
    independent evaluator agree about what a trajectory costs."""
    built = build_scenario(get_scenario("turn-limited"))
    report = run_plan(built, heuristic=heuristic)
    assert report.status == "solved"
    assert report.evaluation.feasible
    assert report.evaluation.comparable_cost == pytest.approx(
        report.search_cost, rel=1e-9
    )
    assert report.evaluation.num_turns > 0
    assert report.evaluation.turn_time_min > 0.0


@pytest.mark.parametrize("name", sorted(SCENARIO_LIBRARY))
def test_evaluator_agrees_with_the_search_on_every_library_scenario(name):
    """T-33 across the library, under each scenario's own turn model."""
    from dataclasses import replace

    spec = get_scenario(name)
    built = build_scenario(spec)
    report = run_plan(built, heuristic="optimistic")
    if report.status == "solved":
        assert report.evaluation.comparable_cost == pytest.approx(
            report.search_cost, rel=1e-9
        )
    # And again with turns forced on, where the grid permits them at all.
    forced = build_scenario(replace(spec, turn_model="gate+cost", max_bank_deg=35.0))
    forced_report = run_plan(forced, heuristic="optimistic", max_expansions=200000)
    if forced_report.status == "solved":
        assert forced_report.evaluation.comparable_cost == pytest.approx(
            forced_report.search_cost, rel=1e-9
        )


def test_the_evaluator_recomputes_turns_and_ignores_the_heading_index():
    """T-33: evaluation must be driven by the geometry of consecutive triples,
    not by anything the planner stored in the state."""
    built = build_scenario(get_scenario("turn-limited"))
    report = run_plan(built, heuristic="optimistic")
    from_flight_states = evaluate_trajectory(
        built.cost_model, report.path, start_move=built.problem.start_move
    )
    from_cells = evaluate_trajectory(
        built.cost_model,
        [cell_of(s) for s in report.path],
        start_move=built.problem.start_move,
    )
    assert from_cells.cost.total == pytest.approx(from_flight_states.cost.total)
    assert from_cells.num_turns == from_flight_states.num_turns


def test_the_declared_start_heading_is_charged_by_the_evaluator():
    """A departure heading that disagrees with the first leg is a real corner."""
    built = build_scenario(get_scenario("turn-limited"))
    path = [GridState(2, 20, 0), GridState(3, 21, 0), GridState(4, 22, 0)]
    without = evaluate_trajectory(built.cost_model, path)
    with_start = evaluate_trajectory(built.cost_model, path, start_move=(1, 0))
    assert without.num_turns == 0
    assert with_start.num_turns == 1
    assert with_start.cost.total > without.cost.total


def test_wind_heading_excess_is_zero_without_wind_and_not_with_it():
    """The wind/heading coupling metric, which is the point of M2-2."""
    from dataclasses import replace

    spec = get_scenario("turn-limited")
    still = build_scenario(spec)
    report = run_plan(still, heuristic="optimistic")
    assert report.evaluation.wind_heading_excess_deg == pytest.approx(0.0, abs=1e-9)

    windy_spec = replace(
        spec,
        wind=[{"type": "uniform", "direction_from_deg": 180.0, "speed_kt": 120.0}],
    )
    windy = build_scenario(windy_spec)
    windy_report = run_plan(windy, heuristic="optimistic")
    assert windy_report.status == "solved"
    assert windy_report.evaluation.num_turns > 0
    assert abs(windy_report.evaluation.wind_heading_excess_deg) > 1e-6
