"""Adapters for observed atmospheric wind data.

The planner consumes wind as a local east/north flow vector.  This module keeps
data acquisition outside the planner: callers provide validated observations,
which makes network-backed weather services optional and reproducible tests
possible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..core.geometry import Vec2
from .wind import DynamicWindField, WindField


@dataclass(frozen=True, slots=True)
class WindObservation:
    """One observed wind vector in the planner's local frame."""

    x_nm: float
    y_nm: float
    altitude_ft: float
    vector_kt: Vec2
    time_h: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.x_nm,
            self.y_nm,
            self.altitude_ft,
            self.time_h,
            self.vector_kt.x,
            self.vector_kt.y,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("weather observations must contain finite values")


class ObservedWindField(WindField, DynamicWindField):
    """Observed wind adapter with optional multilinear interpolation.

    Interpolation is enabled only for a complete rectangular snapshot. Queries
    outside its sampled domain are clamped to the nearest boundary, never
    extrapolated. The default remains nearest-neighbour for synthetic and
    backwards-compatible callers.
    """

    name = "observed"

    def __init__(
        self,
        observations: tuple[WindObservation, ...],
        *,
        interpolate: bool = False,
    ) -> None:
        if not observations:
            raise ValueError("at least one weather observation is required")
        self._observations = observations
        self._interpolate = interpolate

    @property
    def observations(self) -> tuple[WindObservation, ...]:
        return self._observations

    def _nearest(
        self, x_nm: float, y_nm: float, altitude_ft: float, time_h: float
    ) -> WindObservation:
        return min(
            self._observations,
            key=lambda observation: (
                (observation.x_nm - x_nm) ** 2
                + (observation.y_nm - y_nm) ** 2
                + ((observation.altitude_ft - altitude_ft) / 1000.0) ** 2
                + (observation.time_h - time_h) ** 2
            ),
        )

    def at(self, x_nm: float, y_nm: float, altitude_ft: float) -> Vec2:
        return self._lookup(x_nm, y_nm, altitude_ft, 0.0)

    def at_time(
        self, x_nm: float, y_nm: float, altitude_ft: float, t_h: float
    ) -> Vec2:
        return self._lookup(x_nm, y_nm, altitude_ft, t_h)

    def _lookup(
        self, x_nm: float, y_nm: float, altitude_ft: float, time_h: float
    ) -> Vec2:
        if not self._interpolate:
            return self._nearest(x_nm, y_nm, altitude_ft, time_h).vector_kt
        return Vec2(
            self._interpolated_component(
                x_nm, y_nm, altitude_ft, time_h, component="x"
            ),
            self._interpolated_component(
                x_nm, y_nm, altitude_ft, time_h, component="y"
            ),
        )

    def _interpolated_component(
        self,
        x_nm: float,
        y_nm: float,
        altitude_ft: float,
        time_h: float,
        *,
        component: str,
    ) -> float:
        axes = (
            sorted({observation.x_nm for observation in self._observations}),
            sorted({observation.y_nm for observation in self._observations}),
            sorted({observation.altitude_ft for observation in self._observations}),
            sorted({observation.time_h for observation in self._observations}),
        )
        values = (x_nm, y_nm, altitude_ft, time_h)
        brackets = tuple(
            _bracket(value, axis) for value, axis in zip(values, axes, strict=True)
        )
        total = 0.0
        for ix, wx in brackets[0]:
            for iy, wy in brackets[1]:
                for iz, wz in brackets[2]:
                    for it, wt in brackets[3]:
                        observation = next(
                            (
                                item
                                for item in self._observations
                                if item.x_nm == axes[0][ix]
                                and item.y_nm == axes[1][iy]
                                and item.altitude_ft == axes[2][iz]
                                and item.time_h == axes[3][it]
                            ),
                            None,
                        )
                        if observation is None:
                            return self._nearest(
                                x_nm, y_nm, altitude_ft, time_h
                            ).vector_kt.x if component == "x" else self._nearest(
                                x_nm, y_nm, altitude_ft, time_h
                            ).vector_kt.y
                        component_value = (
                            observation.vector_kt.x
                            if component == "x"
                            else observation.vector_kt.y
                        )
                        total += wx * wy * wz * wt * component_value
        return total

    @property
    def interpolation_method(self) -> str:
        return (
            "multilinear with boundary clamping"
            if self._interpolate
            else "nearest observation"
        )

    def max_magnitude_kt(self) -> float:
        return max(observation.vector_kt.norm() for observation in self._observations)


def _bracket(value: float, axis: list[float]) -> tuple[tuple[int, float], ...]:
    """Return lower/upper indices and weights, clamping outside the axis."""
    if len(axis) == 1 or value <= axis[0]:
        return ((0, 1.0),)
    if value >= axis[-1]:
        return ((len(axis) - 1, 1.0),)
    for index in range(len(axis) - 1):
        lower, upper = axis[index], axis[index + 1]
        if lower <= value <= upper:
            fraction = (value - lower) / (upper - lower)
            return ((index, 1.0 - fraction), (index + 1, fraction))
    raise ValueError("value could not be bracketed")
