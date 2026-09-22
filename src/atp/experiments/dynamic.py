"""Reproducible static/dynamic planning experiment orchestration.

This module is deliberately downstream of the planning API.  It constructs
explicit dynamic problems for experiments, runs the existing generic search,
and writes one stable, machine-readable record per run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from ..aircraft.envelope import SpeedEnvelope
from ..aircraft.performance import AircraftPerformance, MEDIUM_TWIN_JET
from ..core.geometry import Vec2
from ..core.temporal import TemporalConfig
from ..environment.airspace import Airspace, GridSpec, GridState
from ..environment.wind import DynamicWindField, PeriodicWind
from ..evaluation.metrics import evaluate_trajectory
from ..experiments.runner import run_plan as run_static_plan
from ..planning.astar import SearchResult, SearchStatus, astar, dijkstra
from ..planning.heuristics import DynamicOptimisticHeuristic, ZeroHeuristic
from ..planning.problem import GoalSpec, TrajectoryPlanningProblem
from ..scenarios.library import get_scenario
from ..scenarios.spec import build_scenario


@dataclass(frozen=True)
class DynamicExperimentSpec:
    """One deterministic experiment family definition."""

    name: str = "dynamic-default"
    cells: tuple[int, ...] = (5,)
    time_steps_h: tuple[float, ...] = (0.5,)
    wind_amplitudes_kt: tuple[float, ...] = (0.0,)
    wind_periods_h: tuple[float, ...] = (4.0,)
    speed_sets_kt: tuple[tuple[float, ...], ...] = ((330.0,),)
    turn_models: tuple[str, ...] = ("none",)
    modes: tuple[str, ...] = ("static", "dynamic")
    heuristics: tuple[str, ...] = ("zero", "dynamic-optimistic")
    seed: int = 0
    max_expansions: int | None = None
    family: str = "MATRIX"


class _ZeroDynamicWind(DynamicWindField):
    def at_time(
        self, x_nm: float, y_nm: float, altitude_ft: float, t_h: float
    ) -> Vec2:
        return Vec2(0.0, 0.0)

    def max_magnitude_kt(self) -> float:
        return 0.0


def _wind(amplitude_kt: float, period_h: float) -> DynamicWindField:
    if amplitude_kt == 0.0:
        return _ZeroDynamicWind()
    return PeriodicWind(
        centre_nm=Vec2(20.0, 20.0),
        base_vector_kt=Vec2(0.0, 0.0),
        amplitude_kt=amplitude_kt,
        period_h=period_h,
        altitude_scale_ft=1000.0,
        phase_offset=0.0,
    )


def _build_problem(
    *,
    cells: int,
    time_step_h: float,
    amplitude_kt: float,
    period_h: float,
    speeds_kt: tuple[float, ...],
    turn_model: str,
    dynamic: bool,
    start: tuple[int, int, int] = (0, 0, 0),
    goal: tuple[int, int, int] | None = None,
    aircraft: AircraftPerformance = MEDIUM_TWIN_JET,
    dynamic_wind: DynamicWindField | None = None,
    flight_levels: tuple[int, ...] = (300,),
    cell_size_nm: float = 10.0,
    goal_match_level: bool = False,
) -> TrajectoryPlanningProblem:
    grid = GridSpec(
        cells_x=cells,
        cells_y=cells,
        cell_size_nm=cell_size_nm,
        flight_levels=flight_levels,
        connectivity=8,
    )
    airspace = Airspace(spec=grid)
    aircraft = replace(
        aircraft,
        speed_envelope=SpeedEnvelope(
            planning_tas_kt=tuple(sorted(speeds_kt)),
            cruise_tas_kt=min(speeds_kt, key=lambda speed: (abs(speed - 330.0), speed)),
            min_cas_kt=aircraft.envelope.min_cas_kt,
            max_cas_kt=aircraft.envelope.max_cas_kt,
            max_mach=aircraft.envelope.max_mach,
            name=aircraft.envelope.name,
        ),
    )
    from ..planning.cost import CostModel, CostWeights

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
    kwargs: dict[str, Any] = {}
    if dynamic:
        kwargs = {
            "temporal_config": TemporalConfig(
                time_origin_h=0.0, time_step_h=time_step_h
            ),
            "dynamic_wind": dynamic_wind or _wind(amplitude_kt, period_h),
        }
    goal = goal or (cells - 1, cells - 1, 0)
    return TrajectoryPlanningProblem(
        airspace,
        cost_model,
        GridState(*start),
        GoalSpec(GridState(*goal), match_level=goal_match_level),
        **kwargs,
    )


def _run_static_regression(
    *,
    name: str,
    cells: int,
    speeds_kt: tuple[float, ...],
    turn_model: str,
    seed: int,
    max_expansions: int | None,
    start: tuple[int, int, int] = (0, 0, 0),
    goal: tuple[int, int, int] | None = None,
    include_path: bool = False,
    aircraft: AircraftPerformance = MEDIUM_TWIN_JET,
    dynamic_wind: DynamicWindField | None = None,
    flight_levels: tuple[int, ...] = (300,),
    cell_size_nm: float = 10.0,
) -> dict[str, object]:
    scenario = get_scenario("empty-cruise")
    scenario = replace(
        scenario,
        name=name,
        grid=replace(scenario.grid, cells_x=cells, cells_y=cells),
        start=list(start),
        goal=list(goal or (cells - 1, cells - 1, 0)),
        turn_model=turn_model,
        speed_envelope={
            "planning_tas_kt": list(sorted(speeds_kt)),
            "cruise_tas_kt": min(
                speeds_kt, key=lambda speed: (abs(speed - 330.0), speed)
            ),
        },
    )
    built = build_scenario(scenario)
    report = run_static_plan(
        built,
        heuristic="zero",
        max_expansions=max_expansions,
    )
    solved = report.status == SearchStatus.SOLVED.value
    evaluation = report.evaluation
    path = report.path
    final_state = path[-1] if solved and path else None
    record = {
        "family": "STATIC_REGRESSION",
        "scenario_name": name,
        "mode": "static",
        "heuristic": "zero",
        "algorithm": "m3-runner",
        "temporal_resolution_h": None,
        "wind_amplitude_kt": 0.0,
        "wind_period_h": None,
        "speed_set_kt": list(speeds_kt),
        "turn_model": turn_model,
        "seed": seed,
        "solved": solved,
        "status": report.status,
        "objective_cost": report.search_cost if solved else None,
        "distance_nm": evaluation.distance_nm if solved else None,
        "elapsed_time_h": evaluation.time_h if solved else None,
        "planned_time_h": evaluation.time_h if solved else None,
        "fuel_kg": evaluation.fuel_kg if solved else None,
        "risk_exposure": evaluation.risk_exposure if solved else None,
        "expansions": report.statistics.expansions,
        "generated_states": report.statistics.generated,
        "runtime_s": report.statistics.runtime_s,
        "final_state": tuple(final_state) if final_state is not None else None,
        "final_time_bucket": 0 if final_state is not None else None,
        "path_length": len(path),
        "speed_changes": evaluation.speed_changes,
        "heading_transitions": evaluation.num_turns,
        "grid": {
            "cells_x": scenario.grid.cells_x,
            "cells_y": scenario.grid.cells_y,
            "cell_size_nm": scenario.grid.cell_size_nm,
            "connectivity": scenario.grid.connectivity,
            "flight_levels": list(scenario.grid.flight_levels),
        },
    }
    if include_path:
        record["path"] = [tuple(state) for state in path]
    return record


def _search(
    problem: TrajectoryPlanningProblem,
    *,
    algorithm: str,
    heuristic: str,
    max_expansions: int | None,
) -> SearchResult:
    if algorithm == "dijkstra":
        if heuristic != "zero":
            raise ValueError("dijkstra requires the zero heuristic")
        return dijkstra(problem, max_expansions=max_expansions)
    if algorithm != "astar":
        raise ValueError(f"unknown algorithm {algorithm!r}")
    if heuristic == "zero":
        h = ZeroHeuristic()
    elif heuristic == "dynamic-optimistic":
        h = DynamicOptimisticHeuristic(problem)
    else:
        raise ValueError(f"unknown dynamic heuristic {heuristic!r}")
    return astar(
        problem,
        h,
        max_expansions=max_expansions,
        reopen_closed=not h.consistent,
    )


def run_dynamic_experiment(
    *,
    name: str,
    mode: str,
    cells: int,
    time_step_h: float,
    amplitude_kt: float,
    period_h: float,
    speeds_kt: tuple[float, ...],
    turn_model: str,
    algorithm: str,
    heuristic: str,
    seed: int = 0,
    max_expansions: int | None = None,
    family: str = "MATRIX",
    start: tuple[int, int, int] = (0, 0, 0),
    goal: tuple[int, int, int] | None = None,
    include_path: bool = False,
    aircraft: AircraftPerformance = MEDIUM_TWIN_JET,
    dynamic_wind: DynamicWindField | None = None,
    flight_levels: tuple[int, ...] = (300,),
    cell_size_nm: float = 10.0,
    goal_match_level: bool = False,
) -> dict[str, object]:
    """Run one experiment and return a stable result record."""
    if mode not in {"static", "dynamic"}:
        raise ValueError("mode must be 'static' or 'dynamic'")
    if algorithm == "dijkstra" and heuristic != "zero":
        raise ValueError("dijkstra requires the zero heuristic")
    if mode == "static" and heuristic == "dynamic-optimistic":
        raise ValueError("dynamic-optimistic requires dynamic mode")
    if mode == "static":
        return _run_static_regression(
            name=name,
            cells=cells,
            speeds_kt=speeds_kt,
            turn_model=turn_model,
            seed=seed,
            max_expansions=max_expansions,
            start=start,
            goal=goal,
            include_path=include_path,
        )
    dynamic = mode == "dynamic"
    problem = _build_problem(
        cells=cells,
        time_step_h=time_step_h,
        amplitude_kt=amplitude_kt,
        period_h=period_h,
        speeds_kt=speeds_kt,
        turn_model=turn_model,
        dynamic=dynamic,
        start=start,
        goal=goal,
        aircraft=aircraft,
        dynamic_wind=dynamic_wind,
        flight_levels=flight_levels,
        cell_size_nm=cell_size_nm,
        goal_match_level=goal_match_level,
    )
    result = _search(
        problem,
        algorithm=algorithm,
        heuristic=heuristic,
        max_expansions=max_expansions,
    )
    path = list(result.path)
    solved = result.status is SearchStatus.SOLVED
    evaluation = evaluate_trajectory(problem.cost_model, path)
    final_state = path[-1] if solved and path else None
    final_bucket = getattr(final_state, "k", 0) if final_state is not None else None
    planned_time_h = (
        final_bucket * time_step_h
        if dynamic and final_bucket is not None
        else evaluation.time_h if solved else None
    )
    record = {
        "family": family,
        "scenario_name": name,
        "mode": mode,
        "heuristic": heuristic,
        "algorithm": algorithm,
        "temporal_resolution_h": time_step_h if dynamic else None,
        "wind_amplitude_kt": amplitude_kt,
        "wind_period_h": period_h,
        "speed_set_kt": list(speeds_kt),
        "turn_model": turn_model,
        "seed": seed,
        "solved": solved,
        "status": result.status.value,
        "objective_cost": result.cost if solved else None,
        "distance_nm": evaluation.distance_nm if solved else None,
        "elapsed_time_h": evaluation.time_h if solved else None,
        "planned_time_h": planned_time_h,
        "fuel_kg": evaluation.fuel_kg if solved else None,
        "risk_exposure": evaluation.risk_exposure if solved else None,
        "expansions": result.statistics.expansions,
        "generated_states": result.statistics.generated,
        "runtime_s": result.statistics.runtime_s,
        "final_state": tuple(final_state) if final_state is not None else None,
        "final_time_bucket": final_bucket,
        "path_length": len(path),
        "speed_changes": evaluation.speed_changes,
        "heading_transitions": evaluation.num_turns,
        "grid": {
            "cells_x": problem.airspace.spec.cells_x,
            "cells_y": problem.airspace.spec.cells_y,
            "cell_size_nm": problem.airspace.spec.cell_size_nm,
            "connectivity": problem.airspace.spec.connectivity,
            "flight_levels": list(problem.airspace.spec.flight_levels),
        },
    }
    if include_path:
        record["path"] = [tuple(state) for state in path]
    return record


def iter_dynamic_experiments(spec: DynamicExperimentSpec) -> Iterable[dict[str, object]]:
    """Yield runs in stable lexicographic matrix order."""
    for cells in sorted(spec.cells):
        for mode in spec.modes:
            for heuristic in spec.heuristics:
                for algorithm in ("dijkstra", "astar"):
                    for time_step_h in sorted(spec.time_steps_h):
                        for amplitude in sorted(spec.wind_amplitudes_kt):
                            for period_h in sorted(spec.wind_periods_h):
                                for speeds in sorted(spec.speed_sets_kt):
                                    for turn_model in spec.turn_models:
                                        if mode == "static" and algorithm != "astar":
                                            continue
                                        if algorithm == "dijkstra" and heuristic != "zero":
                                            continue
                                        if mode == "static" and heuristic == "dynamic-optimistic":
                                            continue
                                        yield run_dynamic_experiment(
                                            name=spec.name,
                                            mode=mode,
                                            cells=cells,
                                            time_step_h=time_step_h,
                                            amplitude_kt=amplitude,
                                            period_h=period_h,
                                            speeds_kt=speeds,
                                            turn_model=turn_model,
                                            algorithm=algorithm,
                                            heuristic=heuristic,
                                            seed=spec.seed,
                                            max_expansions=spec.max_expansions,
                                            family=spec.family,
                                        )


def run_dynamic_matrix(spec: DynamicExperimentSpec) -> list[dict[str, object]]:
    return list(iter_dynamic_experiments(spec))


def run_scaling_experiments(spec: DynamicExperimentSpec) -> list[dict[str, object]]:
    """Execute one explicitly labelled scaling family over ``spec.cells``."""
    return run_dynamic_matrix(replace(spec, family="SCALING"))


def write_dynamic_results(
    results: Iterable[dict[str, object]], path: str | Path
) -> Path:
    """Write deterministic JSON results without embedding machine timing metadata."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"results": list(results)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
