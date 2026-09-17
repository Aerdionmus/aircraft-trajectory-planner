"""Regression tests for defects found in the Milestone 1 adversarial audit.

Each test below failed against commit ``e88ece5`` and documents a specific
mathematical or experimental-validity defect. They are kept in one file so the
audit findings stay traceable.
"""

from __future__ import annotations

import math

import pytest

from atp.aircraft.performance import AircraftPerformance
from atp.core.geometry import Vec2, segment_segment_distance
from atp.environment.airspace import Airspace, GridSpec, GridState
from atp.environment.restrictions import (
    CircularRestriction,
    CorridorRestriction,
    RestrictionSet,
)
from atp.evaluation.metrics import evaluate_trajectory
from atp.planning.astar import dijkstra
from atp.planning.cost import CostModel, CostWeights
from atp.planning.heuristics import OctileHeuristic, OptimisticCostHeuristic
from atp.planning.problem import GoalSpec, TrajectoryPlanningProblem
from atp.scenarios.feasibility import is_goal_reachable
from atp.scenarios.library import get_scenario, random_scenario
from atp.scenarios.spec import build_scenario


# ---------------------------------------------------------------------------
# CRITICAL: the speed-derived cost bound must not be charged against vertical
# distance. Before the fix h(start) exceeded the true optimum by ~1.5% here.
# ---------------------------------------------------------------------------
def _level_constrained_problem() -> TrajectoryPlanningProblem:
    """Time-priced, exactly diagonal, goal level constrained.

    The diagonal alignment is what makes this sharp: on an 8-connected grid the
    optimal horizontal path length equals the straight-line distance exactly, so
    the heuristic has no slack to absorb a spurious vertical term. A slow
    aircraft with a large level spread and no climb fuel penalty removes the
    other sources of slack.
    """
    aircraft = AircraftPerformance(
        name="audit-slow",
        cruise_tas_kt=200.0,
        cruise_fuel_flow_kg_per_h=500.0,
        max_climb_rate_fpm=6000.0,
        max_descent_rate_fpm=6000.0,
        climb_fuel_penalty_kg_per_1000ft=0.0,
        descent_fuel_credit_kg_per_1000ft=0.0,
        service_ceiling_ft=60000.0,
    )
    spec = GridSpec(
        cells_x=6,
        cells_y=6,
        cell_size_nm=4.0,
        flight_levels=(200, 300, 400, 500),
        connectivity=8,
    )
    airspace = Airspace(spec=spec)
    model = CostModel(
        airspace,
        aircraft,
        CostWeights(
            time_cost_per_hour=3000.0,
            fuel_cost_per_kg=0.0,
            distance_cost_per_nm=0.0,
        ),
    )
    return TrajectoryPlanningProblem(
        airspace,
        model,
        start=GridState(0, 0, 0),
        goal=GoalSpec(GridState(5, 5, 3), match_level=True),
    )


@pytest.mark.parametrize(
    "factory", [OptimisticCostHeuristic, OctileHeuristic], ids=["optimistic", "octile"]
)
def test_heuristic_does_not_overestimate_on_level_constrained_diagonal(factory):
    problem = _level_constrained_problem()
    heuristic = factory(problem)
    assert heuristic.admissible
    optimum = dijkstra(problem)
    assert optimum.solved
    assert heuristic(problem.start) <= optimum.cost + 1e-9


@pytest.mark.parametrize(
    "factory", [OptimisticCostHeuristic, OctileHeuristic], ids=["optimistic", "octile"]
)
def test_level_constrained_admissibility_holds_at_every_state(factory):
    """The start state alone is not enough: sweep the whole state space."""
    problem = _level_constrained_problem()
    heuristic = factory(problem)
    for state in problem.airspace.spec.states():
        sub = TrajectoryPlanningProblem(
            problem.airspace,
            problem.cost_model,
            start=state,
            goal=problem.goal,
        )
        result = dijkstra(sub)
        if result.solved:
            assert heuristic(state) <= result.cost + 1e-9, f"overestimate at {state}"


def test_admissibility_with_priced_fuel_and_a_required_descent():
    """Second defect found by the audit: descent fuel credits.

    A descending segment burns less than ``ff_min * time``, so a purely
    per-NM bound overestimates a descending trajectory. Priced fuel, several
    levels and a goal below the start is the configuration that exposes it.
    """
    aircraft = AircraftPerformance(
        name="audit-descent",
        cruise_tas_kt=300.0,
        cruise_fuel_flow_kg_per_h=1200.0,
        reference_altitude_ft=30000.0,
        max_climb_rate_fpm=4000.0,
        max_descent_rate_fpm=4000.0,
        climb_fuel_penalty_kg_per_1000ft=20.0,
        descent_fuel_credit_kg_per_1000ft=20.0,
        service_ceiling_ft=60000.0,
    )
    spec = GridSpec(
        cells_x=5,
        cells_y=5,
        cell_size_nm=15.0,
        flight_levels=(240, 280, 320, 360),
        connectivity=8,
    )
    airspace = Airspace(spec=spec)
    model = CostModel(
        airspace,
        aircraft,
        CostWeights(time_cost_per_hour=1000.0, fuel_cost_per_kg=5.0),
    )
    problem = TrajectoryPlanningProblem(
        airspace,
        model,
        start=GridState(0, 0, 3),
        goal=GoalSpec(GridState(4, 4, 0), match_level=True),
    )
    assert model.constant_cost_offset() > 0.0
    heuristic = OptimisticCostHeuristic(problem)
    for state in spec.states():
        sub = TrajectoryPlanningProblem(
            airspace, model, start=state, goal=problem.goal
        )
        result = dijkstra(sub)
        if result.solved:
            assert heuristic(state) <= result.cost + 1e-9, f"overestimate at {state}"


def test_speed_bound_excludes_the_distance_price():
    """The two bound components are priced against different lengths and must
    stay separate."""
    problem = _level_constrained_problem()
    model = problem.cost_model.with_weights(
        CostWeights(time_cost_per_hour=0.0, distance_cost_per_nm=7.0)
    )
    assert model.speed_cost_lower_bound_per_nm() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# MAJOR: hard corridor restrictions were tested by 16-point sampling and missed
# segments that clipped the capsule between samples.
# ---------------------------------------------------------------------------
def test_corridor_intersection_is_exact_at_the_capsule_cap():
    corridor = CorridorRestriction(
        region_id="C-AUDIT",
        start_nm=Vec2(0.0, 0.0),
        end_nm=Vec2(10.0, 0.0),
        half_width_nm=0.05,
        hard=True,
    )
    # Crosses y = 0 at x = 10.03, i.e. 0.03 NM from the spine endpoint, well
    # inside the 0.05 NM capsule. No sample of the old 16-point scan landed
    # close enough to notice.
    a, b = Vec2(10.03, -3.1), Vec2(10.03, 2.9)
    assert corridor.intersects_segment_2d(a, b)
    # And a segment that genuinely clears the capsule is still allowed.
    assert not corridor.intersects_segment_2d(Vec2(10.2, -3.1), Vec2(10.2, 2.9))


def test_segment_segment_distance_matches_brute_force():
    import random

    rng = random.Random(4242)
    for _ in range(300):
        pts = [Vec2(rng.uniform(-5, 5), rng.uniform(-5, 5)) for _ in range(4)]
        exact = segment_segment_distance(*pts)
        sampled = min(
            math.dist(
                (
                    pts[0].x + (pts[1].x - pts[0].x) * i / 400,
                    pts[0].y + (pts[1].y - pts[0].y) * i / 400,
                ),
                (
                    pts[2].x + (pts[3].x - pts[2].x) * j / 400,
                    pts[2].y + (pts[3].y - pts[2].y) * j / 400,
                ),
            )
            for i in range(0, 401, 8)
            for j in range(0, 401, 8)
        )
        assert exact <= sampled + 1e-9


def test_hard_corridor_blocks_a_transition_in_the_cost_model():
    spec = GridSpec(cells_x=6, cells_y=6, cell_size_nm=10.0, flight_levels=(300,))
    corridor = CorridorRestriction(
        region_id="C-HARD",
        start_nm=Vec2(0.0, 25.0),
        end_nm=Vec2(60.0, 25.0),
        half_width_nm=2.0,
        hard=True,
    )
    airspace = Airspace(spec=spec, restrictions=RestrictionSet(regions=(corridor,)))
    model = CostModel(airspace, _level_constrained_problem().cost_model.aircraft, CostWeights())
    cost, metrics = model.transition_cost(GridState(0, 1, 0), GridState(0, 2, 0))
    assert not metrics.feasible
    assert cost == math.inf


# ---------------------------------------------------------------------------
# MAJOR: totals of an infeasible trajectory cover only its feasible segments,
# so they are not comparable with a feasible trajectory's totals.
# ---------------------------------------------------------------------------
def test_infeasible_trajectory_reports_dropped_segments_and_infinite_cost():
    built = build_scenario(get_scenario("two-no-fly"))
    from atp.baselines.direct import plan_direct_route

    baseline = plan_direct_route(built.problem)
    evaluation = baseline.evaluation
    assert not evaluation.feasible
    assert evaluation.unpriced_segments > 0
    assert evaluation.unpriced_segments == len(evaluation.hard_violations)
    # The priced total covers only part of the route, so it must not be offered
    # as a comparable cost.
    assert evaluation.comparable_cost == math.inf
    assert evaluation.cost.total < math.inf


def test_feasible_trajectory_comparable_cost_equals_priced_total():
    built = build_scenario(get_scenario("empty-cruise"))
    from atp.baselines.direct import plan_direct_route

    evaluation = plan_direct_route(built.problem).evaluation
    assert evaluation.feasible
    assert evaluation.unpriced_segments == 0
    assert evaluation.comparable_cost == pytest.approx(evaluation.cost.total)


def test_evaluation_of_a_path_through_a_no_fly_zone_is_flagged():
    restrictions = RestrictionSet(
        regions=(
            CircularRestriction(
                region_id="P-AUDIT", centre_nm=Vec2(25.0, 25.0), radius_nm=12.0
            ),
        )
    )
    spec = GridSpec(cells_x=6, cells_y=6, cell_size_nm=10.0, flight_levels=(300,))
    airspace = Airspace(spec=spec, restrictions=restrictions)
    model = CostModel(
        airspace,
        _level_constrained_problem().cost_model.aircraft,
        CostWeights(time_cost_per_hour=1.0),
    )
    straight = [GridState(i, i, 0) for i in range(6)]
    evaluation = evaluate_trajectory(model, straight)
    assert not evaluation.feasible
    assert evaluation.comparable_cost == math.inf


# ---------------------------------------------------------------------------
# MAJOR: non-blocked endpoints do not imply a solvable instance.
# ---------------------------------------------------------------------------
def test_reachability_detects_a_wall_that_endpoint_checks_miss():
    spec = GridSpec(cells_x=9, cells_y=9, cell_size_nm=10.0, flight_levels=(300,))
    # Three circles side by side spanning the full width: neither endpoint is
    # inside any of them, yet no route exists.
    wall = RestrictionSet(
        regions=tuple(
            CircularRestriction(
                region_id=f"W-{k}", centre_nm=Vec2(15.0 + 30.0 * k, 45.0), radius_nm=18.0
            )
            for k in range(3)
        )
    )
    airspace = Airspace(spec=spec, restrictions=wall)
    model = CostModel(
        airspace, _level_constrained_problem().cost_model.aircraft, CostWeights(time_cost_per_hour=1.0)
    )
    problem = TrajectoryPlanningProblem(
        airspace, model, start=GridState(0, 0, 0), goal=GoalSpec(GridState(8, 8, 0))
    )
    assert not problem.airspace.is_blocked(problem.start)
    assert not problem.airspace.is_blocked(problem.goal.state)
    assert not is_goal_reachable(problem)
    assert not dijkstra(problem).solved


@pytest.mark.parametrize("seed", [101, 202, 303])
def test_random_scenarios_are_actually_reachable(seed):
    built = build_scenario(random_scenario(seed))
    assert is_goal_reachable(built.problem)
    assert dijkstra(built.problem).solved


def test_random_scenario_reachability_repair_is_deterministic():
    assert random_scenario(77) == random_scenario(77)


@pytest.mark.parametrize("name", ["empty-cruise", "two-no-fly", "dense-restrictions"])
def test_library_scenarios_are_reachable(name):
    assert is_goal_reachable(build_scenario(get_scenario(name)).problem)
