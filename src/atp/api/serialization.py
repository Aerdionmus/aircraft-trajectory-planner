"""Conversion of planner domain objects into frontend-safe JSON structures."""

from __future__ import annotations

import math
from typing import Any, Iterable

from ..environment.airspace import GridState
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
    records: list[dict[str, Any]] = []
    for index, state in enumerate(states):
        ix, iy, il = state.ix, state.iy, state.il
        heading = None
        if index:
            previous = states[index - 1]
            dx = ix - previous.ix
            dy = iy - previous.iy
            if dx or dy:
                heading = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
        speed = None
        if isinstance(state, FlightState) and state.isp is not None and speed_set_kt:
            if 0 <= state.isp < len(speed_set_kt):
                speed = speed_set_kt[state.isp]
        bucket = getattr(state, "k", None)
        records.append(
            {
                "x": ix * cell_size_nm,
                "y": iy * cell_size_nm,
                "altitude_ft": flight_levels[il],
                "heading_deg": heading,
                "speed_tas_kt": speed,
                "time_h": bucket * time_step_h if bucket is not None and time_step_h else None,
                "time_bucket": bucket,
            }
        )
    return _finite(records)
