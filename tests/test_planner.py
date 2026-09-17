from __future__ import annotations

import pytest

from atp.core.geometry import Vec2
from atp.environment.airspace import GridState
from atp.environment.restrictions import (
    CircularRestriction,
    PolygonRestriction,
    RestrictionSet,
)
from atp.environment.wind import LayeredWind, UniformWind
from atp.evaluation.metrics import evaluate_trajectory
from atp.planning.astar import SearchStatus, astar
from atp.planning.cost import CostWeights
from atp.planning.heuristics import OptimisticCostHeuristic
from atp.planning.problem import GoalSpec, TrajectoryPlanningProblem
from conftest import make_problem

TIME_ONLY = CostWeights(time_cost_per_hour=3000.0, fuel_cost_per_kg=0.8)


def solve(problem):
    return astar(problem, OptimisticCostHeuristic(problem))


def test_empty_airspace_path_is_contiguous_and_reaches_the_goal():
    problem = make_problem(cells=15, weights=TIME_ONLY)
    result = solve(problem)
    assert result.status is SearchStatus.SOLVED
    assert result.path[0] == problem.start
    assert problem.is_goal(result.path[-1])
    for a, b in zip(result.path, result.path[1:]):
        assert max(abs(a.ix - b.ix), abs(a.iy - b.iy)) <= 1


def test_planned_trajectory_never_enters_a_prohibited_area():
    restrictions = RestrictionSet(
        regions=(
            CircularRestriction(
                region_id="P-1", centre_nm=Vec2(70.0, 70.0), radius_nm=30.0
            ),
            PolygonRestriction(
                region_id="P-2",
                vertices_nm=(
                    Vec2(20.0, 100.0),
                    Vec2(60.0, 100.0),
                    Vec2(60.0, 130.0),
                    Vec2(20.0, 130.0),
                ),
            ),
        )
    )
    problem = make_problem(cells=15, restrictions=restrictions, weights=TIME_ONLY)
    result = solve(problem)
    assert result.solved

    evaluation = evaluate_trajectory(problem.cost_model, result.path)
    assert evaluation.feasible
    assert evaluation.hard_violations == []
    for state in result.path:
        assert not problem.airspace.is_blocked(state)


def test_fully_enclosed_goal_is_reported_unsolvable():
    wall = PolygonRestriction(
        region_id="WALL",
        vertices_nm=(
            Vec2(0.0, 95.0),
            Vec2(150.0, 95.0),
            Vec2(150.0, 105.0),
            Vec2(0.0, 105.0),
        ),
    )
    problem = make_problem(
        cells=15,
        restrictions=RestrictionSet(regions=(wall,)),
        start=GridState(0, 0, 0),
        goal=GridState(14, 14, 0),
        weights=TIME_ONLY,
    )
    result = solve(problem)
    assert result.status is SearchStatus.UNSOLVABLE


def test_headwind_makes_the_planner_prefer_a_longer_but_faster_route():
    """With a strong crosswind the minimum-time route is not the shortest one."""
    straight = make_problem(cells=15, weights=TIME_ONLY)
    windy = make_problem(
        cells=15, wind=UniformWind(Vec2(-120.0, 120.0)), weights=TIME_ONLY
    )
    calm_result = solve(straight)
    windy_result = solve(windy)
    assert calm_result.solved and windy_result.solved

    calm_eval = evaluate_trajectory(straight.cost_model, calm_result.path)
    windy_eval = evaluate_trajectory(windy.cost_model, windy_result.path)
    # Same geometry, worse wind: the trip must take longer.
    assert windy_eval.time_h > calm_eval.time_h
    # And the cost model must still price it consistently with the search.
    assert windy_eval.cost.total == pytest.approx(windy_result.cost, rel=1e-9)


def test_planner_climbs_into_a_favourable_layer():
    wind = LayeredWind(
        layers=(
            (31000.0, Vec2(-40.0, 0.0)),  # headwind low
            (33000.0, Vec2(120.0, 0.0)),  # strong tailwind at FL320
        ),
        above=Vec2(120.0, 0.0),
    )
    problem = make_problem(
        cells=15,
        cell_size_nm=15.0,
        levels=(280, 300, 320),
        wind=wind,
        start=GridState(0, 7, 0),
        goal=GridState(14, 7, 0),
        weights=TIME_ONLY,
    )
    result = solve(problem)
    assert result.solved
    assert max(s.il for s in result.path) >= 2, "expected a climb into the tailwind"


def test_goal_level_constraint_is_honoured():
    problem = make_problem(
        cells=12,
        cell_size_nm=15.0,
        levels=(280, 300, 320),
        start=GridState(0, 0, 0),
        goal=GridState(11, 11, 2),
        match_level=True,
        weights=TIME_ONLY,
    )
    result = solve(problem)
    assert result.solved
    assert result.path[-1].il == 2


def test_out_of_bounds_endpoints_are_rejected_at_construction():
    problem = make_problem(cells=8)
    with pytest.raises(ValueError):
        TrajectoryPlanningProblem(
            problem.airspace,
            problem.cost_model,
            start=GridState(99, 0, 0),
            goal=GoalSpec(GridState(1, 1, 0)),
        )


def test_evaluation_agrees_with_the_search_cost():
    problem = make_problem(cells=13, weights=TIME_ONLY)
    result = solve(problem)
    evaluation = evaluate_trajectory(problem.cost_model, result.path)
    assert evaluation.cost.total == pytest.approx(result.cost, rel=1e-9)
    assert evaluation.num_waypoints == len(result.path)
