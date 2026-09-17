from __future__ import annotations

from dataclasses import replace

import pytest

from atp.aircraft.performance import MEDIUM_TWIN_JET
from atp.environment.airspace import Airspace, GridSpec, GridState
from atp.planning.cost import CostModel, CostWeights
from atp.planning.problem import GoalSpec, TrajectoryPlanningProblem


@pytest.fixture
def weights() -> CostWeights:
    return CostWeights(
        time_cost_per_hour=3000.0,
        fuel_cost_per_kg=0.8,
        distance_cost_per_nm=0.0,
        risk_cost_per_exposure=5000.0,
    )


def make_problem(
    *,
    cells: int = 12,
    cell_size_nm: float = 10.0,
    levels: tuple[int, ...] = (300,),
    connectivity: int = 8,
    wind=None,
    restrictions=None,
    risk=None,
    weights: CostWeights | None = None,
    start: GridState | None = None,
    goal: GridState | None = None,
    match_level: bool = False,
    aircraft=MEDIUM_TWIN_JET,
    turn_model: str = "none",
    max_turn_deg: float = 180.0,
    max_bank_deg: float | None = None,
    start_heading: int | None = None,
    goal_heading: int | None = None,
) -> TrajectoryPlanningProblem:
    """Build a small planning problem; every component is overridable so that a
    test can isolate exactly one effect."""
    spec = GridSpec(
        cells_x=cells,
        cells_y=cells,
        cell_size_nm=cell_size_nm,
        flight_levels=levels,
        connectivity=connectivity,
    )
    kwargs = {}
    if wind is not None:
        kwargs["wind"] = wind
    if restrictions is not None:
        kwargs["restrictions"] = restrictions
    if risk is not None:
        kwargs["risk"] = risk
    airspace = Airspace(spec=spec, **kwargs)
    if max_bank_deg is not None:
        aircraft = replace(aircraft, max_bank_deg=max_bank_deg)
    model = CostModel(
        airspace,
        aircraft,
        weights or CostWeights(time_cost_per_hour=3000.0, fuel_cost_per_kg=0.8),
        turn_model=turn_model,
        max_turn_deg=max_turn_deg,
    )
    return TrajectoryPlanningProblem(
        airspace,
        model,
        start=start or GridState(0, 0, 0),
        goal=GoalSpec(
            goal or GridState(cells - 1, cells - 1, 0), match_level, goal_heading
        ),
        start_heading_index=start_heading,
    )
