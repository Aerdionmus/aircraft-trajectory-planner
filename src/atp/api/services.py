"""Thin orchestration services for the HTTP presentation layer."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from ..data.airports import Airport, default_airport_database
from ..aircraft.openap import (
    OpenAPAdapterError,
    OpenAPPerformanceProvider,
    SyntheticPerformanceProvider,
)
from ..aircraft.performance import MEDIUM_TWIN_JET
from ..geospatial.transform import GeospatialTransform
from ..experiments.dynamic import (
    DynamicExperimentSpec,
    run_dynamic_experiment,
    run_dynamic_matrix,
    run_scaling_experiments,
)
from ..evaluation.metrics import evaluate_trajectory
from ..planning.astar import SearchStatus, astar, dijkstra
from ..planning.heuristics import ZeroHeuristic
from ..scenarios.geospatial import airport_scenario, resolve_airport_pair
from ..scenarios.library import SCENARIO_LIBRARY, get_scenario
from ..scenarios.spec import build_scenario
from ..environment.snapshot import WeatherSnapshot, default_weather_snapshot
from .models import ExperimentRequest, PlanRequest
from .serialization import _finite, grid_record, trajectory_record


class ConfigurationError(ValueError):
    """Expected user configuration error at the API boundary."""


def _performance_for_request(request: PlanRequest):
    if request.aircraft_source == "openap":
        provider = OpenAPPerformanceProvider(request.aircraft_model or "")
    else:
        provider = SyntheticPerformanceProvider(MEDIUM_TWIN_JET)
    return provider, provider.to_performance()


def _weather_for_request(request: PlanRequest) -> WeatherSnapshot | None:
    return default_weather_snapshot() if request.weather_source == "snapshot" else None


def airports() -> list[dict[str, Any]]:
    db = default_airport_database()
    return [
        {
            "icao": airport.icao,
            "iata": airport.iata,
            "name": airport.name,
            "municipality": airport.municipality,
            "country": airport.country,
            "latitude_deg": airport.latitude_deg,
            "longitude_deg": airport.longitude_deg,
            "elevation_ft": airport.elevation_ft,
            "airport_type": airport.airport_type,
            "source": airport.source,
            "source_url": airport.source_url,
            "source_date": airport.source_date,
        }
        for airport in db
    ]


def airport_detail(query: str) -> dict[str, Any]:
    airport = default_airport_database().lookup(query)
    return {
        "icao": airport.icao,
        "iata": airport.iata,
        "name": airport.name,
        "municipality": airport.municipality,
        "country": airport.country,
        "latitude_deg": airport.latitude_deg,
        "longitude_deg": airport.longitude_deg,
        "elevation_ft": airport.elevation_ft,
        "airport_type": airport.airport_type,
        "source": airport.source,
        "source_url": airport.source_url,
        "source_date": airport.source_date,
    }


def airport_response_context(route: Any | None) -> dict[str, Any]:
    if route is None:
        return {"geographic": False}
    transform = GeospatialTransform(
        route.reference.latitude_deg,
        route.reference.longitude_deg,
    )
    return {
        "geographic": True,
        "reference": {
            "latitude_deg": route.reference.latitude_deg,
            "longitude_deg": route.reference.longitude_deg,
            "name": route.reference.name,
        },
        "airports": {
            "origin": route.origin.as_dict(),
            "destination": route.destination.as_dict(),
        },
        "geography": _airport_snap_errors(route, transform),
    }


def _airport_snap_errors(
    route: Any,
    transform: GeospatialTransform,
) -> dict[str, float]:
    departure_lat, departure_lon = transform.to_geodetic(
        route.grid_origin_local_nm.x + route.start[0] * route.grid.cell_size_nm,
        route.grid_origin_local_nm.y + route.start[1] * route.grid.cell_size_nm,
    )
    arrival_lat, arrival_lon = transform.to_geodetic(
        route.grid_origin_local_nm.x + route.goal[0] * route.grid.cell_size_nm,
        route.grid_origin_local_nm.y + route.goal[1] * route.grid.cell_size_nm,
    )

    def error_nm(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
        local_a = transform.to_local(lat_a, lon_a)
        local_b = transform.to_local(lat_b, lon_b)
        return math.hypot(local_a.x - local_b.x, local_a.y - local_b.y)

    return {
        "departure_snap_error_nm": error_nm(
            route.origin.latitude_deg,
            route.origin.longitude_deg,
            departure_lat,
            departure_lon,
        ),
        "arrival_snap_error_nm": error_nm(
            route.destination.latitude_deg,
            route.destination.longitude_deg,
            arrival_lat,
            arrival_lon,
        ),
    }


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
    route = None
    if request.start_airport is not None and request.goal_airport is not None:
        route = resolve_airport_pair(
            request.start_airport,
            request.goal_airport,
            reference_airport=request.reference_airport,
        )
        spec = airport_scenario(
            request.start_airport,
            request.goal_airport,
            reference_airport=request.reference_airport,
        )
    else:
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
    route_context = (
        resolve_airport_pair(
            request.start_airport,
            request.goal_airport,
            reference_airport=request.reference_airport,
        )
        if request.start_airport is not None and request.goal_airport is not None
        else None
    )
    built = build_scenario(spec)
    transform = (
        GeospatialTransform(
            route_context.reference.latitude_deg,
            route_context.reference.longitude_deg,
        )
        if route_context
        else None
    )
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
        "planner_path_length": len(path),
        "telemetry_sample_count": None,
        "speed_changes": evaluation.speed_changes,
        "heading_transitions": evaluation.num_turns,
    }
    speeds = (
        tuple(built.aircraft.speed_envelope.planning_tas_kt)
        if built.aircraft.speed_envelope
        else None
    )
    telemetry = trajectory_record(
        path,
        cell_size_nm=built.airspace.spec.cell_size_nm,
        flight_levels=built.airspace.spec.flight_levels,
        time_step_h=None,
        speed_set_kt=speeds,
        geospatial_transform=transform,
        grid_origin_local_nm=route_context.grid_origin_local_nm if route_context else None,
    )
    metrics["telemetry_sample_count"] = len(telemetry)
    return _finite(
        {
            "status": status,
            "algorithm": algorithm,
            "heuristic": heuristic,
            "trajectory": telemetry,
            "planner_path_length": len(path),
            "telemetry_sample_count": len(telemetry),
            "metrics": metrics,
            "environment": {
                "mode": "static",
                "temporal_resolution_h": None,
                "wind": {"amplitude_kt": None, "period_h": None},
            },
            "grid": grid_record(built.airspace.spec),
            **airport_response_context(route_context),
        }
    )


def plan(request: PlanRequest) -> dict[str, Any]:
    provider, aircraft = _performance_for_request(request)
    snapshot = _weather_for_request(request)
    route = None
    if request.start_airport is not None and request.goal_airport is not None:
        route = resolve_airport_pair(
            request.start_airport,
            request.goal_airport,
            reference_airport=request.reference_airport,
            # The OpenAP flight-state search spans time and altitude as well as
            # horizontal cells. A coarser airport grid keeps this connected
            # demonstration bounded without changing synthetic scenarios.
            cell_size_nm=60.0 if request.aircraft_source == "openap" else 10.0,
        )
        request_start = tuple(route.start)
        request_goal = tuple(route.goal)
        scenario_name = f"{route.origin.icao}-{route.destination.icao}"
        route_cells = max(route.grid.cells_x, route.grid.cells_y)
    else:
        request_start = request.start or (0, 0, 0)
        request_goal = request.goal
        scenario_name = request.scenario or "api-plan"
        route_cells = request.cells

    if request.mode == "static":
        if request.aircraft_source == "openap":
            raise OpenAPAdapterError(
                "OpenAP performance is connected to dynamic planning only; "
                "use mode='dynamic' or aircraft_source='synthetic'"
            )
        return _static_plan(request)
    dynamic_cells = max(request.cells, route_cells)
    speeds = request.speed_set_kt or (
        aircraft.envelope.planning_tas_kt
        if request.aircraft_source == "openap"
        else (330.0,)
    )
    flight_levels = (0, 100, 200, 300) if request.aircraft_source == "openap" else (300,)
    result = run_dynamic_experiment(
        name=scenario_name,
        mode="dynamic",
        cells=dynamic_cells,
        time_step_h=request.temporal_resolution_h,
        amplitude_kt=request.wind_amplitude_kt,
        period_h=request.wind_period_h,
        speeds_kt=tuple(speeds),
        turn_model=request.turn_model,
        algorithm=request.algorithm,
        heuristic=request.heuristic,
        max_expansions=(
            request.max_expansions
            if request.max_expansions is not None
            else (10_000 if request.aircraft_source == "openap" else None)
        ),
        start=request_start,
        goal=request_goal,
        include_path=True,
        aircraft=aircraft,
        dynamic_wind=snapshot.field if snapshot is not None else None,
        flight_levels=flight_levels,
        cell_size_nm=route.grid.cell_size_nm if route is not None else 10.0,
        goal_match_level=request.aircraft_source == "openap",
    )
    path = result.pop("path", [])
    grid = result["grid"]
    transform = (
        GeospatialTransform(
            route.reference.latitude_deg,
            route.reference.longitude_deg,
        )
        if request.start_airport is not None and request.goal_airport is not None
        else None
    )
    telemetry = trajectory_record(
        [tuple(state) for state in path],
        cell_size_nm=grid["cell_size_nm"],
        flight_levels=tuple(grid["flight_levels"]),
        time_step_h=request.temporal_resolution_h,
        speed_set_kt=tuple(result["speed_set_kt"]),
        geospatial_transform=transform,
        grid_origin_local_nm=route.grid_origin_local_nm if transform else None,
        aircraft=aircraft,
        wind=snapshot.field if snapshot is not None else None,
        sampling_interval_nm=20.0 if request.aircraft_source == "openap" else None,
    )
    return _finite(
        {
            "status": result["status"],
            "algorithm": result["algorithm"],
            "heuristic": result["heuristic"],
            "trajectory": telemetry,
            "planner_path_length": len(path),
            "telemetry_sample_count": len(telemetry),
            "metrics": {
                **{
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
                "planner_path_length": len(path),
                "telemetry_sample_count": len(telemetry),
            },
            "environment": {
                "mode": "dynamic",
                "temporal_resolution_h": result["temporal_resolution_h"],
                "wind": {
                    "amplitude_kt": result["wind_amplitude_kt"],
                    "period_h": result["wind_period_h"],
                    "source": request.weather_source,
                },
            },
            "grid": grid,
            "aircraft": {
                "source": aircraft.performance_source,
                "model": aircraft.performance_model or aircraft.name,
            },
            "weather": (
                {
                    "source": snapshot.source,
                    "valid_time": snapshot.valid_time,
                    "mode": "snapshot",
                    "spatial_resolution": snapshot.spatial_resolution,
                    "vertical_levels": list(snapshot.vertical_levels),
                    "temporal_resolution": snapshot.temporal_resolution,
                    "provenance": snapshot.provenance,
                    "interpolation_method": snapshot.field.interpolation_method,
                }
                if snapshot is not None
                else {
                    "source": "synthetic",
                    "valid_time": None,
                    "mode": "synthetic",
                    "spatial_resolution": None,
                    "vertical_levels": None,
                    "temporal_resolution": None,
                    "provenance": "Deterministic PeriodicWind model",
                }
            ),
            **airport_response_context(route if transform else None),
        }
    )


def run_experiment(request: ExperimentRequest) -> dict[str, Any]:
    spec = DynamicExperimentSpec(**request.model_dump())
    if request.family.upper() == "SCALING":
        results = run_scaling_experiments(spec)
    else:
        results = run_dynamic_matrix(spec)
    return _finite({"results": results})
