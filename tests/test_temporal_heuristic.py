from __future__ import annotations

import math
from dataclasses import replace

import pytest

from atp.aircraft.envelope import SpeedEnvelope
from atp.aircraft.performance import MEDIUM_TWIN_JET
from atp.core.geometry import Vec2
from atp.core.temporal import TemporalConfig
from atp.environment.airspace import Airspace, GridSpec, GridState
from atp.environment.wind import DynamicWindField, PeriodicWind
from atp.planning.astar import SearchStatus, astar, dijkstra
from atp.planning.cost import CostModel, CostWeights
from atp.planning.heuristics import (
    DynamicOptimisticHeuristic,
    ZeroHeuristic,
)
from atp.planning.problem import GoalSpec, TrajectoryPlanningProblem, Transition


class ZeroWind(DynamicWindField):
    def at_time(self, x_nm: float, y_nm: float, altitude_ft: float, t_h: float) -> Vec2:
        return Vec2(0.0, 0.0)

    def max_magnitude_kt(self) -> float:
        return 0.0


def make_dynamic_problem(
    *,
    connectivity: int = 8,
    turn_model: str = "none",
    speed: bool = False,
    wind: DynamicWindField | None = None,
    start: GridState = GridState(0, 0, 0),
    goal: GridState = GridState(4, 4, 0),
) -> TrajectoryPlanningProblem:
    spec = GridSpec(
        cells_x=5,
        cells_y=5,
        cell_size_nm=10.0,
        flight_levels=(300,),
        connectivity=connectivity,
    )
    airspace = Airspace(spec=spec)
    aircraft = MEDIUM_TWIN_JET
    if speed:
        aircraft = replace(
            aircraft,
            speed_envelope=SpeedEnvelope((300.0, 360.0, 420.0), 360.0),
        )
    cost_model = CostModel(
        airspace,
        aircraft,
        CostWeights(
            time_cost_per_hour=2000.0,
            fuel_cost_per_kg=0.5,
            distance_cost_per_nm=2.0,
        ),
        turn_model=turn_model,
    )
    return TrajectoryPlanningProblem(
        airspace,
        cost_model,
        start,
        GoalSpec(goal),
        temporal_config=TemporalConfig(time_origin_h=1.0, time_step_h=0.5),
        dynamic_wind=wind or ZeroWind(),
    )


class Subproblem:
    def __init__(self, problem, start) -> None:
        self.problem = problem
        self.start = start

    def initial_state(self):
        return self.start

    def is_goal(self, state):
        return self.problem.is_goal(state)

    def successors(self, state):
        return self.problem.successors(state)


def exact_remaining_cost(problem, state) -> float:
    result = dijkstra(Subproblem(problem, state))
    return result.cost


def test_zero_heuristic_remains_the_reference():
    problem = make_dynamic_problem()
    result = astar(problem, ZeroHeuristic())
    assert result.status is SearchStatus.SOLVED
    assert result.cost == dijkstra(make_dynamic_problem()).cost


def test_dynamic_heuristic_goal_is_zero_and_nonnegative():
    problem = make_dynamic_problem()
    heuristic = DynamicOptimisticHeuristic(problem)
    assert heuristic(problem.initial_state()) >= 0.0
    assert heuristic(problem.initial_state()) == pytest.approx(
        heuristic(problem.initial_state())
    )
    assert heuristic(
        problem.initial_state()._replace(
            ix=problem.goal.state.ix, iy=problem.goal.state.iy
        )
    ) == pytest.approx(0.0)


def test_dynamic_heuristic_is_admissible_for_sampled_reachable_states():
    problem = make_dynamic_problem(wind=PeriodicWind(
        centre_nm=Vec2(20.0, 20.0),
        base_vector_kt=Vec2(10.0, 0.0),
        amplitude_kt=20.0,
        period_h=3.0,
        altitude_scale_ft=1000.0,
    ))
    heuristic = DynamicOptimisticHeuristic(problem)
    reference = dijkstra(problem)
    sampled = reference.path[::2] + [reference.path[-1]]
    for state in sampled:
        assert heuristic(state) <= exact_remaining_cost(problem, state) + 1e-8


def test_dynamic_heuristic_values_are_deterministic_across_departure_buckets():
    problem = make_dynamic_problem()
    heuristic = DynamicOptimisticHeuristic(problem)
    start = problem.initial_state()
    later = start._replace(k=start.k + 7)
    assert heuristic(start) == heuristic(later)


@pytest.mark.parametrize(
    ("connectivity", "turn_model", "speed"),
    [(4, "none", False), (8, "gate", False), (8, "none", True)],
)
def test_dynamic_heuristic_matches_zero_heuristic_optimum(
    connectivity, turn_model, speed
):
    problem = make_dynamic_problem(
        connectivity=connectivity,
        turn_model=turn_model,
        speed=speed,
    )
    reference = dijkstra(problem)
    result = astar(problem, DynamicOptimisticHeuristic(problem))
    assert reference.status is result.status is SearchStatus.SOLVED
    assert result.cost == pytest.approx(reference.cost, rel=1e-9)


def test_static_m3_regression_and_registry():
    spec = GridSpec(cells_x=4, cells_y=4, cell_size_nm=10.0, flight_levels=(300,), connectivity=8)
    airspace = Airspace(spec=spec)
    cost_model = CostModel(
        airspace,
        replace(
            MEDIUM_TWIN_JET,
            speed_envelope=SpeedEnvelope((300.0, 360.0, 420.0), 360.0),
        ),
        CostWeights(time_cost_per_hour=1000.0, distance_cost_per_nm=2.0),
    )
    problem = TrajectoryPlanningProblem(
        airspace, cost_model, GridState(0, 0, 0), GoalSpec(GridState(3, 3, 0))
    )
    heuristic = DynamicOptimisticHeuristic(problem)
    assert heuristic.admissible
    assert heuristic(problem.initial_state()) >= 0.0
    assert astar(problem, heuristic).cost == pytest.approx(dijkstra(problem).cost)


@pytest.mark.parametrize("seed", range(5))
def test_multiple_deterministic_small_dynamic_scenarios(seed):
    start = GridState(0, 0, 0)
    goal = GridState(2 + seed % 3, 2 + (seed * 2) % 3, 0)
    problem = make_dynamic_problem(
        connectivity=4 if seed % 2 == 0 else 8,
        start=start,
        goal=goal,
        wind=PeriodicWind(
            centre_nm=Vec2(20.0, 20.0),
            base_vector_kt=Vec2(float(seed), 0.0),
            amplitude_kt=10.0 + seed,
            period_h=2.0 + seed,
            altitude_scale_ft=1000.0,
        ),
    )
    heuristic = DynamicOptimisticHeuristic(problem)
    reference = dijkstra(problem)
    result = astar(problem, heuristic)
    assert reference.solved and result.solved
    assert result.cost == pytest.approx(reference.cost, rel=1e-9)
    assert all(math.isfinite(heuristic(state)) for state in reference.path)
