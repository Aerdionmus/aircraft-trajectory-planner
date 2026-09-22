"""Concrete OpenAP performance provider.

The provider is intentionally small: OpenAP remains the source of aircraft
properties and fuel-flow calculations, while the existing planner still
consumes :class:`AircraftPerformance`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from ..core.units import M_PER_FT, kt_to_mps, mps_to_kt
from .envelope import SpeedEnvelope
from .performance import AircraftPerformance


class OpenAPAdapterError(ValueError):
    """Raised when OpenAP is unavailable or a model is unsupported."""


class AircraftPerformanceProvider(Protocol):
    source: str
    model: str

    def to_performance(self) -> AircraftPerformance:
        ...


class SyntheticPerformanceProvider:
    """Provider wrapper for the existing deterministic aircraft model."""

    source = "synthetic"

    def __init__(self, aircraft: AircraftPerformance) -> None:
        self.aircraft = aircraft
        self.model = aircraft.name

    def to_performance(self) -> AircraftPerformance:
        return self.aircraft


@dataclass(frozen=True, slots=True)
class OpenAPPerformanceProvider:
    """OpenAP-backed A320 performance adapter.

    OpenAP's public API returns SI speeds and fuel flow in kg/s.  Conversion is
    performed at this boundary; the rest of the planner remains in kt, ft, and
    kg/h.
    """

    model: str = "A320"
    source: str = "openap"
    nominal_mass_kg: float | None = None

    def __post_init__(self) -> None:
        try:
            from openap import prop
        except ImportError as error:
            raise OpenAPAdapterError(
                "OpenAP is unavailable; install the 'openap' dependency"
            ) from error
        normalized = self.model.lower()
        if normalized not in prop.available_aircraft():
            raise OpenAPAdapterError(
                f"OpenAP aircraft {self.model!r} is unsupported; "
                f"available models include {prop.available_aircraft()}"
            )
        object.__setattr__(self, "model", normalized.upper())

    @lru_cache(maxsize=1)
    def _modules(self):
        try:
            from openap import Aero, FuelFlow, prop
        except ImportError as error:
            raise OpenAPAdapterError(
                "OpenAP is unavailable; install the 'openap' dependency"
            ) from error
        return prop, Aero(), FuelFlow(self.model.lower())

    @property
    @lru_cache(maxsize=1)
    def properties(self) -> dict:
        prop, _, _ = self._modules()
        return prop.aircraft(self.model.lower())

    @property
    def mass_kg(self) -> float:
        properties = self.properties
        return (
            float(self.nominal_mass_kg)
            if self.nominal_mass_kg is not None
            else (float(properties["oew"]) + float(properties["mtow"])) / 2.0
        )

    @property
    def cruise_tas_kt(self) -> float:
        _, aero, _ = self._modules()
        return mps_to_kt(
            float(aero.mach2tas(float(self.properties["cruise"]["mach"]), float(self.properties["cruise"]["height"])))
        )

    @property
    def service_ceiling_ft(self) -> float:
        return float(self.properties["limits"]["ceiling"]) / M_PER_FT

    @lru_cache(maxsize=4096)
    def fuel_flow_kg_per_h(
        self,
        altitude_ft: float,
        tas_kt: float,
        *,
        mass_kg: float | None = None,
        vertical_rate_fpm: float = 0.0,
    ) -> float:
        _, _, fuel_flow = self._modules()
        value_kg_s = fuel_flow.enroute(
            float(self.mass_kg if mass_kg is None else mass_kg),
            kt_to_mps(tas_kt),
            float(altitude_ft * M_PER_FT),
            vs=float(vertical_rate_fpm * 0.00508),
        )
        value = float(value_kg_s) * 3600.0
        if not math.isfinite(value) or value < 0.0:
            raise OpenAPAdapterError("OpenAP returned an invalid fuel-flow value")
        return value

    def to_performance(self) -> AircraftPerformance:
        properties = self.properties
        cruise = self.cruise_tas_kt
        limits = properties["limits"]
        planning_speeds = tuple(sorted({250.0, 330.0, round(cruise, 3)}))
        envelope = SpeedEnvelope(
            planning_tas_kt=planning_speeds,
            cruise_tas_kt=min(planning_speeds, key=lambda value: abs(value - cruise)),
            max_cas_kt=float(limits["VMO"]) * 1.0,
            max_mach=float(limits["MMO"]),
            name="openap-a320-envelope",
        )
        return AircraftPerformance(
            name=f"openap:{self.model}",
            cruise_tas_kt=envelope.cruise_tas_kt,
            cruise_fuel_flow_kg_per_h=self.fuel_flow_kg_per_h(
                float(properties["cruise"]["height"]) / M_PER_FT, cruise
            ),
            reference_altitude_ft=float(properties["cruise"]["height"]) / M_PER_FT,
            max_climb_rate_fpm=1800.0,
            max_descent_rate_fpm=2200.0,
            service_ceiling_ft=self.service_ceiling_ft,
            speed_envelope=envelope,
            fuel_flow_model=lambda altitude_ft, tas_kt: self.fuel_flow_kg_per_h(
                altitude_ft, tas_kt
            ),
            performance_source=self.source,
            performance_model=self.model,
            mass_kg=self.mass_kg,
            fuel_capacity_kg=float(properties["limits"]["MFC"]),
        )


@dataclass(frozen=True, slots=True)
class OpenAPProfile:
    """Backward-compatible explicit profile value object.

    New code should use :class:`OpenAPPerformanceProvider`; this value object is
    retained for callers that already supply provider-derived values directly.
    """

    aircraft_id: str
    cruise_tas_kt: float
    cruise_fuel_flow_kg_per_h: float
    max_climb_rate_fpm: float
    max_descent_rate_fpm: float
    service_ceiling_ft: float
    max_bank_deg: float = 25.0

    def to_performance(self) -> AircraftPerformance:
        if not self.aircraft_id.strip():
            raise OpenAPAdapterError("aircraft_id must not be empty")
        try:
            return AircraftPerformance(
                name=f"openap:{self.aircraft_id}",
                cruise_tas_kt=self.cruise_tas_kt,
                cruise_fuel_flow_kg_per_h=self.cruise_fuel_flow_kg_per_h,
                max_climb_rate_fpm=self.max_climb_rate_fpm,
                max_descent_rate_fpm=self.max_descent_rate_fpm,
                service_ceiling_ft=self.service_ceiling_ft,
                max_bank_deg=self.max_bank_deg,
                performance_source="openap",
                performance_model=self.aircraft_id,
            )
        except ValueError as error:
            raise OpenAPAdapterError(str(error)) from error
