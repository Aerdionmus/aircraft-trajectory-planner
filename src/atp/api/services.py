"""Thin orchestration services for the HTTP presentation layer."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..experiments.dynamic import (
    DynamicExperimentSpec,
    run_dynamic_experiment,
    run_dynamic_matrix,
    run_scaling_experiments,
)
from ..evaluation.metrics import evaluate_trajectory
from ..planning.astar import SearchStatus, astar, dijkstra
from ..planning.heuristics import ZeroHeuristic
from ..scenarios.library import SCENARIO_LIBRARY, get_scenario
from ..scenarios.spec import build_scenario
from .models import ExperimentRequest, PlanRequest
from .serialization import _finite, grid_record, trajectory_record


class ConfigurationError(ValueError):
    """Expected user configuration error at the API boundary."""


def list_scenarios() -> list[dict[str, Any]]:
    return [
        {"name": name, "description": get_scenario(name).description}
        for name in sorted(SCENARIO_LIBRARY)
    ]


def scenario_detail(name: str) -> dict[str, Any]:
    spec = get_scenario(name)
    built = build_scenario(spec)
    envelope = built.aircraft.speed_envelope
    return _finite(
        {
            "name": spec.name,
            "description": spec.description,
            "grid": grid_record(built.airspace.spec),
            "start": list(spec.start),
            "goal": list(spec.goal),
            "restrictions": spec.restrictions,
            "aircraft": {
                "name": built.aircraft.name,
                "cruise_tas_kt": built.aircraft.cruise_tas_kt,
            },
            "speed_options_kt": list(envelope.planning_tas_kt) if envelope else [],
            "supported_modes": ["static", "dynamic"],
        }
    )


def _static_plan(request: PlanRequest) -> dict[str, Any]:
    if not request.scenario:
        raise ConfigurationError("static plans require a scenario")
    spec = get_scenario(request.scenario)
    if request.start is not None:
        spec = replace(spec, start=list(request.start))
    if request.goal is not None:
        spec = replace(spec, goal=list(request.goal))
    if request.turn_model != "none":
        spec = replace(spec, turn_model=request.turn_model)
    if request.speed_set_kt is not None:
        if spec.speed_envelope is None:
            raise ConfigurationError("scenario does not declare a speed envelope")
        spec = replace(
            spec,
            speed_envelope={
                **spec.speed_envelope,
                "planning_tas_kt": list(request.speed_set_kt),
                "cruise_tas_kt": min(request.speed_set_kt),
            },
        )
    built = build_scenario(spec)
    if request.algorithm == "dijkstra":
        result = dijkstra(
            built.problem,
            max_expansions=request.max_expansions,
        )
        status = result.status.value
        algorithm = "dijkstra"
        heuristic = "zero"
        path = result.path
        cost = result.cost
        evaluation = evaluate_trajectory(
            built.cost_model, path, start_move=built.problem.start_move
        )
        statistics = result.statistics
    else:
        result = astar(
            built.problem,
            ZeroHeuristic(),
            max_expansions=request.max_expansions,
        )
        status = result.status.value
        algorithm = "astar"
        heuristic = "zero"
        path = result.path
        cost = result.cost
        evaluation = evaluate_trajectory(
            built.cost_model, path, start_move=built.problem.start_move
        )
        statistics = result.statistics
    solved = status == SearchStatus.SOLVED.value
    metrics = {
        "objective_cost": cost if solved else None,
        "distance_nm": evaluation.distance_nm if solved else None,
        "elapsed_time_h": evaluation.time_h if solved else None,
        "planned_time_h": evaluation.time_h if solved else None,
        "fuel_kg": evaluation.fuel_kg if solved else None,
        "risk_exposure": evaluation.risk_exposure if solved else None,
        "expansions": statistics.expansions,
        "generated_states": statistics.generated,
        "runtime_s": statistics.runtime_s,
        "path_length": len(path),
        "speed_changes": evaluation.speed_changes,
        "heading_transitions": evaluation.num_turns,
    }
    speeds = (
        tuple(built.aircraft.speed_envelope.planning_tas_kt)
        if built.aircraft.speed_envelope
        else None
    )
    return _finite(
        {
            "status": status,
            "algorithm": algorithm,
            "heuristic": heuristic,
            "trajectory": trajectory_record(
                path,
                cell_size_nm=built.airspace.spec.cell_size_nm,
                flight_levels=built.airspace.spec.flight_levels,
                time_step_h=None,
                speed_set_kt=speeds,
            ),
            "metrics": metrics,
            "environment": {
                "mode": "static",
                "temporal_resolution_h": None,
                "wind": {"amplitude_kt": None, "period_h": None},
            },
            "grid": grid_record(built.airspace.spec),
        }
    )


def plan(request: PlanRequest) -> dict[str, Any]:
    if request.mode == "static":
        return _static_plan(request)
    result = run_dynamic_experiment(
        name=request.scenario or "api-plan",
        mode="dynamic",
        cells=request.cells,
        time_step_h=request.temporal_resolution_h,
        amplitude_kt=request.wind_amplitude_kt,
        period_h=request.wind_period_h,
        speeds_kt=request.speed_set_kt or (330.0,),
        turn_model=request.turn_model,
        algorithm=request.algorithm,
        heuristic=request.heuristic,
        max_expansions=request.max_expansions,
        start=request.start or (0, 0, 0),
        goal=request.goal,
        include_path=True,
    )
    path = result.pop("path", [])
    grid = result["grid"]
    return _finite(
        {
            "status": result["status"],
            "algorithm": result["algorithm"],
            "heuristic": result["heuristic"],
            "trajectory": trajectory_record(
                [tuple(state) for state in path],
                cell_size_nm=grid["cell_size_nm"],
                flight_levels=tuple(grid["flight_levels"]),
                time_step_h=request.temporal_resolution_h,
                speed_set_kt=tuple(result["speed_set_kt"]),
            ),
            "metrics": {
                key: result[key]
                for key in (
                    "objective_cost",
                    "distance_nm",
                    "elapsed_time_h",
                    "planned_time_h",
                    "fuel_kg",
                    "risk_exposure",
                    "expansions",
                    "generated_states",
                    "runtime_s",
                    "path_length",
                    "speed_changes",
                    "heading_transitions",
                )
            },
            "environment": {
                "mode": "dynamic",
                "temporal_resolution_h": result["temporal_resolution_h"],
                "wind": {
                    "amplitude_kt": result["wind_amplitude_kt"],
                    "period_h": result["wind_period_h"],
                },
            },
            "grid": grid,
        }
    )


def run_experiment(request: ExperimentRequest) -> dict[str, Any]:
    spec = DynamicExperimentSpec(**request.model_dump())
    if request.family.upper() == "SCALING":
        results = run_scaling_experiments(spec)
    else:
        results = run_dynamic_matrix(spec)
    return _finite({"results": results})
