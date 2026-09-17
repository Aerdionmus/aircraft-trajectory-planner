"""Aircraft performance model.

Scope of the model (all of it is a simplification, stated here rather than
buried in the code):

* Point mass, constant mass.  Fuel burn does **not** reduce weight, so there is
  no weight/fuel-flow feedback and no step-climb optimisation driven by mass.
* A single commanded TAS per level, independent of mass and temperature.  No
  Mach/CAS schedule, no ISA deviation, no buffet or thrust-limited ceiling.
* Cruise fuel flow varies linearly with altitude about a reference level.  Real
  specific fuel consumption is not linear in altitude; the linear term only
  encodes the qualitative "higher is more efficient" trend.
* Climb and descent are charged a fixed fuel delta per 1000 ft on top of the
  cruise burn for the time spent, and are limited by a constant maximum
  vertical rate.  No thrust/drag integration.
* No turn dynamics: heading is not part of the state, so turn radius, bank
  limits and the associated path lengthening are not modelled.

This is adequate for comparing *search strategies*, which is what the project
is about.  It is not adequate for predicting fuel burn of a real aircraft.  If
higher fidelity is ever needed the intended replacement is a BADA-style table
behind the same interface; nothing outside this module depends on the
functional form.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.units import FT_PER_FLIGHT_LEVEL


@dataclass(frozen=True)
class AircraftPerformance:
    name: str
    cruise_tas_kt: float
    cruise_fuel_flow_kg_per_h: float
    reference_altitude_ft: float = 33000.0
    #: Fractional change in fuel flow per 1000 ft above the reference level.
    #: Negative => burning less fuel higher up.
    fuel_flow_gradient_per_1000ft: float = -0.02
    min_fuel_flow_kg_per_h: float = 0.0
    max_climb_rate_fpm: float = 1800.0
    max_descent_rate_fpm: float = 2000.0
    climb_fuel_penalty_kg_per_1000ft: float = 22.0
    descent_fuel_credit_kg_per_1000ft: float = 6.0
    service_ceiling_ft: float = 41000.0

    def __post_init__(self) -> None:
        if self.cruise_tas_kt <= 0:
            raise ValueError("cruise_tas_kt must be positive")
        if self.cruise_fuel_flow_kg_per_h < 0:
            raise ValueError("cruise_fuel_flow_kg_per_h must be non-negative")
        if self.max_climb_rate_fpm <= 0 or self.max_descent_rate_fpm <= 0:
            raise ValueError("vertical rate limits must be positive")

    # -- speed ---------------------------------------------------------------
    def tas_kt(self, altitude_ft: float) -> float:
        """Commanded TAS. Constant in the MVP; kept as a function so that a
        speed schedule can be introduced without touching the cost model."""
        return self.cruise_tas_kt

    @property
    def max_tas_kt(self) -> float:
        return self.cruise_tas_kt

    # -- fuel ----------------------------------------------------------------
    def fuel_flow_kg_per_h(self, altitude_ft: float) -> float:
        delta_1000ft = (altitude_ft - self.reference_altitude_ft) / 1000.0
        flow = self.cruise_fuel_flow_kg_per_h * (
            1.0 + self.fuel_flow_gradient_per_1000ft * delta_1000ft
        )
        return max(self.min_fuel_flow_kg_per_h, flow)

    def min_fuel_flow_over_levels_kg_per_h(
        self, flight_levels: tuple[int, ...]
    ) -> float:
        """Lower bound on fuel flow over the levels actually available.

        Used by the admissible heuristic, so it must never overestimate.
        """
        if not flight_levels:
            return self.min_fuel_flow_kg_per_h
        return min(
            self.fuel_flow_kg_per_h(fl * FT_PER_FLIGHT_LEVEL) for fl in flight_levels
        )

    def vertical_fuel_delta_kg(self, delta_altitude_ft: float) -> float:
        """Fuel delta for a level change, on top of the cruise burn.

        Positive for a climb, negative (a credit) for a descent.  The cost model
        clamps total segment fuel at zero so that a descent credit can never
        produce a negative edge cost, which would invalidate A*.
        """
        if delta_altitude_ft > 0.0:
            return (delta_altitude_ft / 1000.0) * self.climb_fuel_penalty_kg_per_1000ft
        if delta_altitude_ft < 0.0:
            return (delta_altitude_ft / 1000.0) * self.descent_fuel_credit_kg_per_1000ft
        return 0.0

    # -- vertical capability -------------------------------------------------
    def max_vertical_rate_fpm(self, delta_altitude_ft: float) -> float:
        return (
            self.max_climb_rate_fpm if delta_altitude_ft >= 0 else self.max_descent_rate_fpm
        )

    def can_operate_at(self, altitude_ft: float) -> bool:
        return altitude_ft <= self.service_ceiling_ft


#: Representative (not certified, not manufacturer-sourced) parameter sets.
MEDIUM_TWIN_JET = AircraftPerformance(
    name="medium-twin-jet",
    cruise_tas_kt=450.0,
    cruise_fuel_flow_kg_per_h=2400.0,
    reference_altitude_ft=33000.0,
    fuel_flow_gradient_per_1000ft=-0.020,
    max_climb_rate_fpm=1800.0,
    max_descent_rate_fpm=2200.0,
    climb_fuel_penalty_kg_per_1000ft=22.0,
    descent_fuel_credit_kg_per_1000ft=6.0,
    service_ceiling_ft=41000.0,
)

REGIONAL_TURBOPROP = AircraftPerformance(
    name="regional-turboprop",
    cruise_tas_kt=280.0,
    cruise_fuel_flow_kg_per_h=650.0,
    reference_altitude_ft=20000.0,
    fuel_flow_gradient_per_1000ft=-0.015,
    max_climb_rate_fpm=1200.0,
    max_descent_rate_fpm=1500.0,
    climb_fuel_penalty_kg_per_1000ft=9.0,
    descent_fuel_credit_kg_per_1000ft=3.0,
    service_ceiling_ft=25000.0,
)

AIRCRAFT_LIBRARY: dict[str, AircraftPerformance] = {
    MEDIUM_TWIN_JET.name: MEDIUM_TWIN_JET,
    REGIONAL_TURBOPROP.name: REGIONAL_TURBOPROP,
}
