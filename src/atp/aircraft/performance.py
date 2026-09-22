"""Aircraft performance model.

Scope of the model (all of it is a simplification, stated here rather than
buried in the code):

* Point mass, constant mass.  Fuel burn does **not** reduce weight, so there is
  no weight/fuel-flow feedback and no step-climb optimisation driven by mass.
  Milestone 3 does **not** change this.
* **Speed (Milestone 3).**  The aircraft no longer has a single commanded TAS.
  It carries a :class:`~atp.aircraft.envelope.SpeedEnvelope` of discrete
  selectable TAS values, and fuel flow and turn performance are functions of the
  selected speed.  An aircraft that declares no envelope gets the singleton
  ``{cruise_tas_kt}``, which reproduces the Milestone 2 behaviour exactly -- see
  :meth:`AircraftPerformance.envelope`.  There is still no Mach/CAS *schedule*
  (the planner selects a TAS, it does not fly a climb or cruise schedule) and no
  buffet or thrust-limited ceiling beyond ``service_ceiling_ft`` and the
  envelope's own CAS/Mach limits.
* Cruise fuel flow varies linearly with altitude about a reference level.  Real
  specific fuel consumption is not linear in altitude; the linear term only
  encodes the qualitative "higher is more efficient" trend.  Milestone 3 leaves
  that altitude trend untouched and multiplies it by a speed factor; see
  :meth:`AircraftPerformance.fuel_flow_kg_per_h`.
* Climb and descent are charged a fixed fuel delta per 1000 ft on top of the
  cruise burn for the time spent, and are limited by a constant maximum
  vertical rate.  No thrust/drag integration.
* Turn dynamics are limited to a constant maximum bank angle driving a
  coordinated level turn (see :mod:`atp.aircraft.turn`).  There is no roll-in
  or roll-out time, no bank scheduling, no load-factor limit beyond the bank
  limit, and no turn/climb coupling.

This is adequate for comparing *search strategies*, which is what the project
is about.  It is not adequate for predicting fuel burn of a real aircraft.  If
higher fidelity is ever needed the intended replacement is a BADA-style table
behind the same interface; nothing outside this module depends on the
functional form.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

from ..core.units import FT_PER_FLIGHT_LEVEL
from .envelope import SpeedEnvelope
from .turn import turn_radius_nm, turn_rate_rad_per_h


@lru_cache(maxsize=None)
def _fixed_envelope(cruise_tas_kt: float) -> SpeedEnvelope:
    """The implicit singleton envelope of an aircraft that declares none.

    Cached because it is requested on every transition and is a pure function of
    one float.
    """
    return SpeedEnvelope.fixed(cruise_tas_kt)


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
    #: Maximum bank angle for a coordinated level turn, in degrees.  Purely a
    #: representative operating limit, not a certified structural one.
    max_bank_deg: float = 25.0
    #: Multiplier on the cruise fuel flow while turning, standing in for the
    #: extra induced drag at load factor ``1 / cos(bank)``.  Must be >= 1 so a
    #: turn can never be cheaper in fuel than straight flight.
    turn_fuel_factor: float = 1.0
    #: Milestone 3.  The discrete speeds the planner may select between.  When
    #: ``None`` the aircraft behaves exactly as in Milestone 2: the implicit
    #: envelope is the singleton ``{cruise_tas_kt}``, so there is no speed
    #: decision and every quantity is evaluated at the cruise TAS.
    speed_envelope: SpeedEnvelope | None = None
    #: Milestone 3.  Fraction of the power required at the cruise speed that is
    #: *induced* (lift-dependent) rather than parasite.  Sets the shape of the
    #: speed/fuel-flow curve -- see :meth:`fuel_flow_kg_per_h`.  Must lie in
    #: ``[0, 1)``; ``0`` gives a pure parasite ``V^3`` law with no low-speed
    #: rise.  Synthetic, like every other coefficient here.
    induced_power_fraction: float = 0.25
    #: Optional provider-owned fuel model.  Synthetic aircraft leave this unset;
    #: provider-backed aircraft use it instead of the synthetic power curve.
    fuel_flow_model: Callable[[float, float], float] | None = None
    performance_source: str = "synthetic"
    performance_model: str | None = None
    mass_kg: float | None = None
    fuel_capacity_kg: float | None = None

    def __post_init__(self) -> None:
        if self.cruise_tas_kt <= 0:
            raise ValueError("cruise_tas_kt must be positive")
        if self.cruise_fuel_flow_kg_per_h < 0:
            raise ValueError("cruise_fuel_flow_kg_per_h must be non-negative")
        if self.max_climb_rate_fpm <= 0 or self.max_descent_rate_fpm <= 0:
            raise ValueError("vertical rate limits must be positive")
        if not 0.0 < self.max_bank_deg < 90.0:
            raise ValueError("max_bank_deg must lie strictly between 0 and 90")
        if self.turn_fuel_factor < 1.0:
            raise ValueError("turn_fuel_factor must be >= 1")
        if not 0.0 <= self.induced_power_fraction < 1.0:
            raise ValueError("induced_power_fraction must lie in [0, 1)")

    # -- speed ---------------------------------------------------------------
    @property
    def envelope(self) -> SpeedEnvelope:
        """The speed envelope, defaulting to the singleton ``{cruise_tas_kt}``.

        An aircraft that declares no envelope therefore has exactly one
        selectable speed, which is what makes Milestone 2 parity structural: no
        speed decision exists, so nothing downstream can behave differently.
        """
        if self.speed_envelope is not None:
            return self.speed_envelope
        return _fixed_envelope(self.cruise_tas_kt)

    @property
    def reference_tas_kt(self) -> float:
        """The speed the fuel-flow law is normalised at.

        This is ``cruise_tas_kt`` and it **never moves with the envelope**.  The
        distinction matters for controlled experiments: a speed sweep that
        substitutes the singleton envelope ``{330 kt}`` for ``{450 kt}`` must
        change which speed is *flown*, not which speed the performance model is
        defined around.  If the normalisation moved with the envelope, every
        singleton in a sweep would be flying its own differently-scaled
        aircraft and the sweep would compare nothing.
        """
        return self.cruise_tas_kt

    def tas_kt(self, altitude_ft: float) -> float:
        """Default commanded TAS at a level: the envelope's cruise speed.

        This is the speed used wherever none has been selected -- the baselines,
        and any call that does not name one.  With no declared envelope it is
        ``cruise_tas_kt``, exactly as in Milestone 1/2.  With a multi-speed
        envelope the planner selects per segment instead; see
        :mod:`atp.aircraft.envelope`.
        """
        return self.envelope.cruise_tas_kt

    @property
    def max_tas_kt(self) -> float:
        """Fastest TAS the aircraft can be planned at anywhere in the envelope.

        The admissible heuristic uses this to bound ground speed from above, so
        it must never *under*estimate.  With a singleton envelope it is the
        cruise TAS, exactly as in Milestone 1/2.
        """
        return self.envelope.max_planning_tas_kt

    # -- fuel ----------------------------------------------------------------
    def speed_fuel_factor(self, tas_kt: float) -> float:
        """Multiplier on the reference fuel flow for flying at ``tas_kt``.

        Synthetic model, motivated by the classic power-required split.  Fuel
        flow is taken proportional to the power required, which for a point mass
        in level flight at fixed mass is thrust times speed, i.e. drag times
        speed.  Writing drag as a parasite term growing with ``V^2`` and an
        induced term falling as ``V^-2``, the power required is

            P(V) = k_p V^3 + k_i / V

        Normalising at the reference speed (:attr:`reference_tas_kt`, i.e. the
        aircraft's own cruise TAS, independent of any envelope attached to it)
        and writing ``b`` for the induced share
        of the cruise power (``induced_power_fraction``) and ``x = V / V_c``:

            factor(x) = (1 - b) x^3 + b / x

        ``factor(1) = 1`` by construction, so the Milestone 2 fuel flow is
        recovered **exactly** at the cruise speed -- the equality is
        short-circuited rather than left to floating-point luck, which is what
        makes fixed-speed parity bit-exact.

        The curve has a genuine interior minimum at ``x = (b / (3(1-b)))^(1/4)``
        (for ``b > 0``): flying slower than that costs *more* fuel per hour, not
        less, because the induced term takes over.  That is the qualitative
        shape of a real power-required curve and it is what makes the speed
        decision a real trade-off rather than a monotone knob.

        Maximum specific range (NM per kg, i.e. maximising ``x / factor(x)``
        rather than minimising ``factor(x)``) sits at a different speed again:
        differentiating ``x / factor(x)`` and setting the result to zero gives

            x = (b / (1 - b))^(1/4)

        which is **not** the same expression as the fuel-flow minimum above --
        the denominator lacks the minimum's factor of 3, so in general the two
        speeds differ by more than that factor alone would suggest.  For the
        shipped ``b = 0.25`` this happens to reduce to ``x = 3^(-1/4)``, which
        is why a numeric mention of ``3^(-1/4)`` elsewhere in this codebase and
        its documentation is correct for the shipped aircraft specifically, not
        as the general formula.  That the two optimal speeds differ at all is
        why a minimum-fuel plan and a minimum-time plan select different
        speeds.

        **[LIMITATION]** This is a shape, not a propulsion model.  There is no
        thrust/drag integration, no compressibility (drag rise near the Mach
        limit is absent, so the model understates the cost of the fastest
        options), no mass dependence and no altitude/speed cross-coupling: the
        altitude trend is the inherited Milestone 1 linear one and multiplies
        this factor independently.
        """
        if tas_kt <= 0.0:
            raise ValueError("tas_kt must be positive")
        x = tas_kt / self.reference_tas_kt
        if x == 1.0:
            return 1.0
        b = self.induced_power_fraction
        return (1.0 - b) * x * x * x + b / x

    def fuel_flow_kg_per_h(
        self, altitude_ft: float, tas_kt: float | None = None
    ) -> float:
        """Cruise fuel flow at an altitude and (Milestone 3) a speed.

        ``tas_kt=None`` means the aircraft's own cruise TAS, which reproduces the
        Milestone 1/2 value exactly.
        """
        if self.fuel_flow_model is not None:
            return max(
                0.0,
                float(
                    self.fuel_flow_model(
                        altitude_ft, self.tas_kt(altitude_ft) if tas_kt is None else tas_kt
                    )
                ),
            )
        delta_1000ft = (altitude_ft - self.reference_altitude_ft) / 1000.0
        flow = self.cruise_fuel_flow_kg_per_h * (
            1.0 + self.fuel_flow_gradient_per_1000ft * delta_1000ft
        )
        if tas_kt is not None:
            flow *= self.speed_fuel_factor(tas_kt)
        return max(self.min_fuel_flow_kg_per_h, flow)

    def min_fuel_flow_over_levels_kg_per_h(
        self, flight_levels: tuple[int, ...], tas_kt: float | None = None
    ) -> float:
        """Lower bound on fuel flow over the levels actually available, at a
        given speed.

        Used by the admissible heuristic, so it must never overestimate.  Fuel
        flow is linear in altitude before the clamp and the clamp is a maximum
        with a constant, so the minimum over any altitude *between* two levels
        is attained at one of them; evaluating at the enumerated levels is
        therefore sufficient even though segments are priced at mid-altitudes.
        """
        if not flight_levels:
            return self.min_fuel_flow_kg_per_h
        return min(
            self.fuel_flow_kg_per_h(fl * FT_PER_FLIGHT_LEVEL, tas_kt)
            for fl in flight_levels
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

    # -- turn capability -----------------------------------------------------
    def turn_radius_nm(
        self, altitude_ft: float, tas_kt: float | None = None
    ) -> float:
        """Air-mass turn radius [NM], ``V^2 / (g tan phi)``.

        ``tas_kt=None`` uses the default commanded TAS, reproducing Milestone 2.
        Passing a speed is how Milestone 3 makes turn geometry respond to the
        speed decision: the radius grows with the *square* of TAS, so the speed
        choice moves turn feasibility much faster than it moves anything else.
        """
        return turn_radius_nm(
            self.tas_kt(altitude_ft) if tas_kt is None else tas_kt, self.max_bank_deg
        )

    def turn_rate_rad_per_h(
        self, altitude_ft: float, tas_kt: float | None = None
    ) -> float:
        """Turn rate [rad/h], ``g tan(phi) / V``.  Falls with TAS, so a faster
        aircraft takes longer to turn through the same angle."""
        return turn_rate_rad_per_h(
            self.tas_kt(altitude_ft) if tas_kt is None else tas_kt, self.max_bank_deg
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
