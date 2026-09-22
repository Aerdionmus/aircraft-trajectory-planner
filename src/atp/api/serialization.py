"""Conversion of planner domain objects into frontend-safe JSON structures."""

from __future__ import annotations

import math
from typing import Any, Iterable

from ..environment.airspace import GridState
from ..core.geometry import Vec2
from ..geospatial.transform import GeospatialTransform
from ..aircraft.kinematics import solve_ground_speed
from ..aircraft.performance import AircraftPerformance
from ..environment.wind import DynamicWindField, WindField
from ..planning.state import FlightState


def _finite(value: Any) -> Any:
    if isinstance(value, BaseException):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def grid_record(grid: Any) -> dict[str, Any]:
    return {
        "cells_x": grid.cells_x,
        "cells_y": grid.cells_y,
        "cell_size_nm": grid.cell_size_nm,
        "flight_levels": list(grid.flight_levels),
        "connectivity": grid.connectivity,
    }


def trajectory_record(
    path: Iterable[GridState | FlightState],
    *,
    cell_size_nm: float,
    flight_levels: tuple[int, ...],
    time_step_h: float | None,
    speed_set_kt: tuple[float, ...] | None,
    geospatial_transform: GeospatialTransform | None = None,
    grid_origin_local_nm: Vec2 | None = None,
    aircraft: AircraftPerformance | None = None,
    wind: WindField | DynamicWindField | None = None,
    sampling_interval_nm: float | None = None,
) -> list[dict[str, Any]]:
    states = [
        (
            FlightState(*state)
            if isinstance(state, tuple) and len(state) == 6
            else GridState(*state)
            if isinstance(state, tuple) and len(state) == 3
            else state
        )
        for state in path
    ]
    samples = _densify_states(
        states,
        cell_size_nm=cell_size_nm,
        flight_levels=flight_levels,
        time_step_h=time_step_h,
        speed_set_kt=speed_set_kt,
        sampling_interval_nm=sampling_interval_nm,
    )
    records: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        record: dict[str, Any] = {
            "x": sample["x"],
            "y": sample["y"],
            "altitude_ft": sample["altitude_ft"],
            "heading_deg": sample["heading_deg"],
            "speed_tas_kt": sample["speed_tas_kt"],
            "time_h": sample["time_h"],
            "time_bucket": sample["time_bucket"],
            "flight_phase": _phase(samples, index),
        }
        altitude = float(record["altitude_ft"])
        speed = record["speed_tas_kt"]
        if aircraft is not None and isinstance(speed, (int, float)):
            record["fuel_flow_kg_h"] = aircraft.fuel_flow_kg_per_h(altitude, speed)
        else:
            record["fuel_flow_kg_h"] = None
        record["ground_speed_kt"], wind_vector = _ground_speed_at_sample(
            samples, index, speed, wind, flight_levels, cell_size_nm, time_step_h
        )
        if wind_vector is None:
            record["wind_speed_kt"] = None
            record["wind_direction_deg"] = None
        else:
            record["wind_speed_kt"] = wind_vector.norm()
            record["wind_direction_deg"] = (
                math.degrees(math.atan2(wind_vector.x, wind_vector.y)) + 360.0
            ) % 360.0
        record["mass_kg"] = aircraft.mass_kg if aircraft is not None else None
        record["fuel_remaining_kg"] = None
        records.append(
            _geographic_record(
                record,
                x_nm=sample["x"],
                y_nm=sample["y"],
                transform=geospatial_transform,
                grid_origin=grid_origin_local_nm,
            )
        )
    _apply_fuel_remaining(records, aircraft)
    return _finite(records)


def _densify_states(
    states: list[GridState | FlightState],
    *,
    cell_size_nm: float,
    flight_levels: tuple[int, ...],
    time_step_h: float | None,
    speed_set_kt: tuple[float, ...] | None,
    sampling_interval_nm: float | None,
) -> list[dict[str, Any]]:
    if not states:
        return []
    samples: list[dict[str, Any]] = []
    for index in range(len(states) - 1):
        current, following = states[index], states[index + 1]
        x0, y0 = current.ix * cell_size_nm, current.iy * cell_size_nm
        x1, y1 = following.ix * cell_size_nm, following.iy * cell_size_nm
        distance = math.hypot(x1 - x0, y1 - y0)
        steps = (
            max(1, math.ceil(distance / sampling_interval_nm))
            if sampling_interval_nm is not None
            else 1
        )
        for step in range(steps):
            fraction = step / steps
            samples.append(
                _interpolated_sample(
                    current,
                    following,
                    fraction,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    flight_levels=flight_levels,
                    time_step_h=time_step_h,
                    speed_set_kt=speed_set_kt,
                )
            )
    last = states[-1]
    samples.append(
        _node_sample(
            last,
            x=last.ix * cell_size_nm,
            y=last.iy * cell_size_nm,
            flight_levels=flight_levels,
            time_step_h=time_step_h,
            speed_set_kt=speed_set_kt,
            heading=None,
        )
    )
    for index, sample in enumerate(samples):
        if sample["heading_deg"] is None:
            if index + 1 < len(samples):
                sample["heading_deg"] = _bearing(
                    samples[index]["x"],
                    samples[index]["y"],
                    samples[index + 1]["x"],
                    samples[index + 1]["y"],
                )
            elif index:
                sample["heading_deg"] = samples[index - 1]["heading_deg"]
    return samples


def _node_sample(
    state: GridState | FlightState,
    *,
    x: float,
    y: float,
    flight_levels: tuple[int, ...],
    time_step_h: float | None,
    speed_set_kt: tuple[float, ...] | None,
    heading: float | None,
) -> dict[str, Any]:
    speed = (
        speed_set_kt[state.isp]
        if isinstance(state, FlightState)
        and speed_set_kt
        and 0 <= state.isp < len(speed_set_kt)
        else None
    )
    bucket = getattr(state, "k", None)
    return {
        "x": x,
        "y": y,
        "altitude_ft": float(flight_levels[state.il]),
        "speed_tas_kt": speed,
        "time_h": bucket * time_step_h if bucket is not None and time_step_h else None,
        "time_bucket": bucket,
        "heading_deg": heading,
    }


def _interpolated_sample(
    current: GridState | FlightState,
    following: GridState | FlightState,
    fraction: float,
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    flight_levels: tuple[int, ...],
    time_step_h: float | None,
    speed_set_kt: tuple[float, ...] | None,
) -> dict[str, Any]:
    first = _node_sample(
        current,
        x=x0,
        y=y0,
        flight_levels=flight_levels,
        time_step_h=time_step_h,
        speed_set_kt=speed_set_kt,
        heading=_bearing(x0, y0, x1, y1),
    )
    second = _node_sample(
        following,
        x=x1,
        y=y1,
        flight_levels=flight_levels,
        time_step_h=time_step_h,
        speed_set_kt=speed_set_kt,
        heading=_bearing(x0, y0, x1, y1),
    )
    return {
        key: (
            first[key] + (second[key] - first[key]) * fraction
            if isinstance(first[key], (int, float))
            and isinstance(second[key], (int, float))
            else first[key]
        )
        for key in first
    }


def _bearing(x0: float, y0: float, x1: float, y1: float) -> float | None:
    if x0 == x1 and y0 == y1:
        return None
    return (math.degrees(math.atan2(x1 - x0, y1 - y0)) + 360.0) % 360.0


def _phase(samples: list[dict[str, Any]], index: int) -> str:
    altitude = samples[index]["altitude_ft"]
    previous = samples[index - 1]["altitude_ft"] if index else altitude
    following = (
        samples[index + 1]["altitude_ft"]
        if index + 1 < len(samples)
        else altitude
    )
    if following > altitude or altitude > previous:
        return "climb"
    if following < altitude or altitude < previous:
        return "descent"
    return "cruise"


def _ground_speed_at_sample(
    samples: list[dict[str, Any]],
    index: int,
    speed: float | None,
    wind: WindField | DynamicWindField | None,
    flight_levels: tuple[int, ...],
    cell_size_nm: float,
    time_step_h: float | None,
 ) -> tuple[float | None, Vec2 | None]:
    if speed is None or wind is None or len(samples) < 2:
        return None, None
    if index + 1 < len(samples):
        following = samples[index + 1]
    else:
        following = samples[index - 1]
    delta = Vec2(
        following["x"] - samples[index]["x"],
        following["y"] - samples[index]["y"],
    )
    if delta.norm() == 0.0:
        return None, None
    altitude = samples[index]["altitude_ft"]
    time_h = samples[index]["time_h"] or 0.0
    if isinstance(wind, DynamicWindField):
        vector = wind.at_time(
            samples[index]["x"], samples[index]["y"], altitude, time_h
        )
    else:
        vector = wind.at(samples[index]["x"], samples[index]["y"], altitude)
    solution = solve_ground_speed(speed, vector, delta.normalized())
    return (solution.ground_speed_kt if solution.feasible else None), vector


def _apply_fuel_remaining(
    records: list[dict[str, Any]], aircraft: AircraftPerformance | None
) -> None:
    if aircraft is None or aircraft.fuel_capacity_kg is None:
        return
    remaining = aircraft.fuel_capacity_kg
    for index, record in enumerate(records):
        record["fuel_remaining_kg"] = remaining
        if aircraft.mass_kg is not None:
            record["mass_kg"] = max(0.0, aircraft.mass_kg - (aircraft.fuel_capacity_kg - remaining))
        if index == 0:
            continue
        flow = record.get("fuel_flow_kg_h")
        previous_time = records[index - 1].get("time_h")
        current_time = record.get("time_h")
        if isinstance(flow, (int, float)) and isinstance(previous_time, (int, float)) and isinstance(current_time, (int, float)):
            remaining = max(0.0, remaining - max(0.0, current_time - previous_time) * flow)


def _geographic_record(
    record: dict[str, Any],
    *,
    x_nm: float,
    y_nm: float,
    transform: GeospatialTransform | None,
    grid_origin: Vec2 | None,
) -> dict[str, Any]:
    if transform is None or grid_origin is None:
        return record
    latitude_deg, longitude_deg = transform.to_geodetic(
        grid_origin.x + x_nm,
        grid_origin.y + y_nm,
    )
    record["latitude_deg"] = latitude_deg
    record["longitude_deg"] = longitude_deg
    return record
