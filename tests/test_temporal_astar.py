from __future__ import annotations

from atp.aircraft.performance import MEDIUM_TWIN_JET
from atp.core.geometry import Vec2
from atp.core.temporal import TemporalConfig
from atp.environment.airspace import Airspace, GridSpec, GridState
from atp.environment.wind import DynamicWindField
from atp.planning.astar import SearchStatus, astar
from atp.planning.cost import CostModel, CostWeights
from atp.planning.heuristics import ZeroHeuristic
from atp.planning.problem import GoalSpec, TrajectoryPlanningProblem, Transition
from atp.planning.state import FlightState


class ZeroWind(DynamicWindField):
    def at_time(self, x_nm: float, y_nm: float, altitude_ft: float, t_h: float) -> Vec2:
        return Vec2(0.0, 0.0)

    def max_magnitude_kt(self) -> float:
        return 0.0


def make_dynamic_problem() -> TrajectoryPlanningProblem:
    airspace = Airspace(
        GridSpec(
            cells_x=4,
            cells_y=2,
            cell_size_nm=10.0,
            flight_levels=(300,),
            connectivity=4,
        )
    )
    cost_model = CostModel(
        airspace,
        MEDIUM_TWIN_JET,
        CostWeights(time_cost_per_hour=1.0, fuel_cost_per_kg=0.0),
    )
    return TrajectoryPlanningProblem(
        airspace,
        cost_model,
        GridState(0, 0, 0),
        GoalSpec(GridState(3, 0, 0)),
        temporal_config=TemporalConfig(time_step_h=0.5),
        dynamic_wind=ZeroWind(),
    )


def test_dynamic_graph_can_be_searched_with_zero_heuristic():
    problem = make_dynamic_problem()
    result = astar(problem, ZeroHeuristic())

    assert result.status is SearchStatus.SOLVED
    assert problem.is_goal(result.path[-1])
    assert all(isinstance(state, FlightState) for state in result.path)
    assert all(child.k >= parent.k for parent, child in zip(result.path, result.path[1:]))


def test_temporal_identity_is_preserved_by_search():
    start = FlightState(0, 0, 0, -1, 0, 0)
    early = FlightState(1, 0, 0, 0, 0, 1)
    late = FlightState(1, 0, 0, 0, 0, 2)

    class TemporalGraph:
        def initial_state(self):
            return start

        def is_goal(self, state):
            return state.ix == 1

        def successors(self, state):
            if state == start:
                yield Transition(early, 5.0)
                yield Transition(late, 1.0)

    result = astar(TemporalGraph(), lambda _state: 0.0)

    assert result.status is SearchStatus.SOLVED
    assert result.path == [start, late]
    assert result.cost == 1.0
    assert result.statistics.generated == 3


def test_dynamic_search_is_deterministic():
    first = astar(make_dynamic_problem(), lambda _state: 0.0)
    second = astar(make_dynamic_problem(), lambda _state: 0.0)

    assert first.status is second.status is SearchStatus.SOLVED
    assert first.path == second.path
    assert first.cost == second.cost
    assert first.statistics.as_dict() | {"runtime_s": 0.0} == (
        second.statistics.as_dict() | {"runtime_s": 0.0}
    )


def test_zero_heuristic_matches_dijkstra_on_dynamic_graph():
    problem = make_dynamic_problem()
    zero = astar(problem, ZeroHeuristic())
    reference = astar(make_dynamic_problem(), lambda _state: 0.0)

    assert zero.status is reference.status is SearchStatus.SOLVED
    assert zero.path == reference.path
    assert zero.cost == reference.cost
