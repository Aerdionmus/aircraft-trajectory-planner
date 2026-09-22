"""Pinned weather snapshot used by the deterministic M6.3 path."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.geometry import Vec2
from .weather import ObservedWindField, WindObservation


@dataclass(frozen=True, slots=True)
class WeatherSnapshot:
    source: str
    valid_time: str
    spatial_resolution: str
    vertical_levels: tuple[int, ...]
    temporal_resolution: str
    provenance: str
    field: ObservedWindField
    temperature_observations: tuple[tuple[float, float, float, float], ...]

    def wind(self, x_nm: float, y_nm: float, altitude_ft: float, time_h: float) -> Vec2:
        return self.field.at_time(x_nm, y_nm, altitude_ft, time_h)

    def temperature(
        self, x_nm: float, y_nm: float, altitude_ft: float, time_h: float
    ) -> float:
        nearest = min(
            self.temperature_observations,
            key=lambda item: (
                (item[0] - x_nm) ** 2
                + ((item[1] - altitude_ft) / 1000.0) ** 2
                + (item[2] - time_h) ** 2
            ),
        )
        return nearest[3]


def default_weather_snapshot() -> WeatherSnapshot:
    """Return a checked-in forecast snapshot sampled on 2026-09-22.

    The values are pinned from Open-Meteo's pressure-level forecast for the
    Chennai/Mumbai corridor.  The application never performs a live request.
    """
    observations = tuple(
        WindObservation(x, y, altitude, Vec2(u, v), time)
        for x, y, altitude, u, v, time in (
            (0.0, 0.0, 16400.0, 13.9, -8.4, 0.0),
            (280.0, 0.0, 16400.0, -3.4, -3.4, 0.0),
            (0.0, 0.0, 32800.0, 14.7, -8.5, 0.0),
            (280.0, 0.0, 32800.0, -8.4, -1.6, 0.0),
            (0.0, 0.0, 16400.0, 19.0, -8.0, 4.0),
            (280.0, 0.0, 16400.0, -5.0, -4.5, 4.0),
            (0.0, 0.0, 32800.0, 16.0, -8.0, 4.0),
            (280.0, 0.0, 32800.0, -9.0, -2.0, 4.0),
        )
    )
    return WeatherSnapshot(
        source="Open-Meteo pressure-level forecast snapshot",
        valid_time="2026-09-22T00:00:00Z",
        spatial_resolution="Chennai/Mumbai sample points",
        vertical_levels=(16400, 32800),
        temporal_resolution="4 h sample from hourly forecast",
        provenance="https://api.open-meteo.com/ (forecast snapshot pinned locally)",
        field=ObservedWindField(observations, interpolate=True),
        temperature_observations=(
            (0.0, 16400.0, 0.0, 293.35),
            (280.0, 16400.0, 0.0, 291.75),
            (0.0, 32800.0, 0.0, 284.55),
            (280.0, 32800.0, 0.0, 284.25),
            (0.0, 16400.0, 4.0, 293.55),
            (280.0, 16400.0, 4.0, 292.05),
            (0.0, 32800.0, 4.0, 284.35),
            (280.0, 32800.0, 4.0, 284.35),
        ),
    )
