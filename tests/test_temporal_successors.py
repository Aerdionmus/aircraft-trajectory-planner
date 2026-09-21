from __future__ import annotations

import math

import pytest

from atp.aircraft.performance import MEDIUM_TWIN_JET
from atp.core.geometry import Vec2
from atp.core.temporal import TemporalConfig
from atp.environment.airspace import Airspace, GridSpec, GridState
from atp.environment.wind import DynamicWindField
from atp.planning.cost import CostModel, CostWeights
from atp.planning.problem import GoalSpec, TrajectoryPlanningProblem
from atp.planning.state import FlightState, NO_HEADING


class ZeroDynamicWind(DynamicWindField):
    def at_time(self, x_nm: float, y_nm: float, altitude_ft: float, t_h: float) -> Vec2:
        return Vec2(0.0, 0.0)

    def max_magnitude_kt(self) -> float:
        return 0.0


class OscillatingWind(DynamicWindField):
    def __init__(self, amplitude_kt: float) -> None:
        self.amplitude_kt = amplitude_kt

    def at_time(self, x_nm: float, y_nm: float, altitude_ft: float, t_h: float) -> Vec2:
        return Vec2(self.amplitude_kt * math.sin(t_h), 0.0)

    def max_magnitude_kt(self) -> float:
        return abs(self.amplitude_kt)


def make_problem(*, dt: float = 0.5, levels: tuple[int, ...] = (300,), allow_pure_level_change: bool = False, wind: DynamicWindField | None = None):
    spec = GridSpec(cells_x=8, cells_y=8, cell_size_nm=10.0, flight_levels=levels, connectivity=8)
    airspace = Airspace(spec=spec)
    cost_model = CostModel(
        airspace,
        MEDIUM_TWIN_JET,
        CostWeights(time_cost_per_hour=1.0, fuel_cost_per_kg=0.0),
        turn_model="none",
    )
    return TrajectoryPlanningProblem(
        airspace,
        cost_model,
        GridState(0, 0, 0),
        GoalSpec(GridState(1, 0, 0)),
        allow_pure_level_change=allow_pure_level_change,
        temporal_config=TemporalConfig(time_step_h=dt),
        dynamic_wind=wind or ZeroDynamicWind(),
    )


def test_static_parity_keeps_state_shape_and_bucket_zero():
    spec = GridSpec(cells_x=5, cells_y=5, cell_size_nm=10.0, flight_levels=(300,), connectivity=8)
    airspace = Airspace(spec=spec)
    cost_model = CostModel(airspace, MEDIUM_TWIN_JET, CostWeights(time_cost_per_hour=1.0, fuel_cost_per_kg=0.0), turn_model="none")
    problem = TrajectoryPlanningProblem(
        airspace,
        cost_model,
        GridState(0, 0, 0),
        GoalSpec(GridState(1, 0, 0)),
    )
    state = problem.initial_state()
    successors = list(problem.successors(state))
    assert all(isinstance(s.state, GridState) for s in successors)
    assert all(getattr(s.state, "k", 0) == 0 for s in successors)


def test_constructor_validation_rejects_partial_dynamic_configuration():
    spec = GridSpec(cells_x=4, cells_y=4, cell_size_nm=10.0, flight_levels=(300,), connectivity=8)
    airspace = Airspace(spec=spec)
    cost_model = CostModel(airspace, MEDIUM_TWIN_JET, CostWeights())
    start = GridState(0, 0, 0)
    goal = GoalSpec(GridState(1, 0, 0))

    with pytest.raises(ValueError):
        TrajectoryPlanningProblem(airspace, cost_model, start, goal, temporal_config=TemporalConfig(time_step_h=0.5))
    with pytest.raises(ValueError):
        TrajectoryPlanningProblem(airspace, cost_model, start, goal, dynamic_wind=ZeroDynamicWind())


def test_initial_state_in_dynamic_mode_starts_at_bucket_zero():
    problem = make_problem()
    initial = problem.initial_state()
    assert isinstance(initial, FlightState)
    assert initial.k == 0


def test_departure_time_mapping_uses_bucket_origin():
    cfg = TemporalConfig(time_origin_h=12.5, time_step_h=1.75)
    assert cfg.time_at_bucket(0) == pytest.approx(12.5)
    assert cfg.time_at_bucket(3) == pytest.approx(12.5 + 3.0 * 1.75)


def test_dynamic_successor_advances_time_bucket_by_traversal_count():
    problem = make_problem(dt=0.5)
    start = problem.initial_state()
    successor = next(iter(problem.successors(start)))
    assert successor.state.k > start.k
    assert successor.state.k == start.k + successor.state.k - start.k


def test_zero_wind_bucket_advance_matches_ceil_tau_over_dt():
    cfg = TemporalConfig(time_step_h=0.5)
    wind = ZeroDynamicWind()
    airspace = Airspace(GridSpec(cells_x=8, cells_y=8, cell_size_nm=10.0, flight_levels=(300,), connectivity=8))
    cost_model = CostModel(airspace, MEDIUM_TWIN_JET, CostWeights(time_cost_per_hour=1.0, fuel_cost_per_kg=0.0), turn_model="none")
    problem = TrajectoryPlanningProblem(
        airspace,
        cost_model,
        GridState(0, 0, 0),
        GoalSpec(GridState(1, 0, 0)),
        temporal_config=cfg,
        dynamic_wind=wind,
    )
    start = problem.initial_state()
    child = next(iter(problem.successors(start)))
    tau = 10.0 / MEDIUM_TWIN_JET.cruise_tas_kt
    expected_bucket = math.ceil(tau / cfg.time_step_h)
    assert child.state.k == start.k + expected_bucket


def test_temporal_identity_distinguishes_same_spatial_state_at_different_buckets():
    assert FlightState(5, 5, 2, 3, 1, 10) != FlightState(5, 5, 2, 3, 1, 11)


def test_departure_time_sensitivity_changes_dynamic_sampled_traversal():
    cfg = TemporalConfig(time_step_h=0.5)
    wind = OscillatingWind(30.0)
    airspace = Airspace(GridSpec(cells_x=8, cells_y=8, cell_size_nm=10.0, flight_levels=(300,), connectivity=8))
    cost_model = CostModel(airspace, MEDIUM_TWIN_JET, CostWeights(time_cost_per_hour=1.0, fuel_cost_per_kg=0.0), turn_model="none")
    p0 = TrajectoryPlanningProblem(airspace, cost_model, GridState(0, 0, 0), GoalSpec(GridState(1, 0, 0)), temporal_config=cfg, dynamic_wind=wind)
    p2 = TrajectoryPlanningProblem(airspace, cost_model, GridState(0, 0, 0), GoalSpec(GridState(1, 0, 0)), temporal_config=cfg, dynamic_wind=wind)
    first = next(iter(p0.successors(FlightState(0, 0, 0, NO_HEADING, 0, 0))))
    second = next(iter(p2.successors(FlightState(0, 0, 0, NO_HEADING, 0, 2))))
    assert first.state.k != second.state.k


def test_rejected_dynamic_transition_is_not_yielded():
    class RejectedWind(DynamicWindField):
        def at_time(self, x_nm: float, y_nm: float, altitude_ft: float, t_h: float) -> Vec2:
            return Vec2(-999.0, 0.0)

        def max_magnitude_kt(self) -> float:
            return 999.0

    cfg = TemporalConfig(time_step_h=0.5)
    airspace = Airspace(GridSpec(cells_x=4, cells_y=4, cell_size_nm=10.0, flight_levels=(300,), connectivity=8))
    cost_model = CostModel(airspace, MEDIUM_TWIN_JET, CostWeights(time_cost_per_hour=1.0, fuel_cost_per_kg=0.0), turn_model="none")
    problem = TrajectoryPlanningProblem(airspace, cost_model, GridState(0, 0, 0), GoalSpec(GridState(1, 0, 0)), temporal_config=cfg, dynamic_wind=RejectedWind())
    successors = list(problem.successors(problem.initial_state()))
    assert not successors


def test_pure_level_change_preserves_bucket_in_dynamic_mode():
    airspace = Airspace(GridSpec(cells_x=4, cells_y=4, cell_size_nm=10.0, flight_levels=(300, 350), connectivity=8))
    cost_model = CostModel(airspace, MEDIUM_TWIN_JET, CostWeights(time_cost_per_hour=1.0, fuel_cost_per_kg=0.0), turn_model="none")
    problem = TrajectoryPlanningProblem(
        airspace,
        cost_model,
        GridState(1, 1, 0),
        GoalSpec(GridState(1, 1, 1), match_level=True),
        allow_pure_level_change=True,
        temporal_config=TemporalConfig(time_step_h=0.5),
        dynamic_wind=ZeroDynamicWind(),
    )
    state = problem.initial_state()
    successors = list(problem.successors(state))
    assert any(s.state.il != state.il and s.state.k == state.k for s in successors)


def test_dynamic_successor_generation_is_deterministic():
    problem = make_problem(dt=0.5)
    start = problem.initial_state()
    first = list(problem.successors(start))
    second = list(problem.successors(start))
    assert first == second


def test_k_changes_only_by_traversal_bucket_count():
    problem = make_problem(dt=0.5)
    start = problem.initial_state()
    child = next(iter(problem.successors(start)))
    assert child.state.k == start.k + max(1, math.ceil(10.0 / MEDIUM_TWIN_JET.cruise_tas_kt / 0.5))
