"""Aircraft speed envelope: the discrete speeds a planner may choose from.

Provenance, stated once
-----------------------
**Every number in this module and in any envelope built from it is synthetic.**
The limits are representative of a generic transport aircraft to within an order
of magnitude and nothing more.  They are not manufacturer data, not from a
certified aeroplane flight manual, not from BADA, and carry no airworthiness or
performance-guarantee claim.  ``docs/assumptions.md`` records this alongside the
rest of the model's provenance.

What an envelope is
-------------------
Two separate things, deliberately not conflated:

*The operating limits* --- the continuous region of the flight envelope the
aircraft may be operated in.  These are expressed the way real limits are
expressed, and therefore the way they actually behave with altitude:

* a **low-speed limit in CAS** (``min_cas_kt``), because the low-speed end of an
  envelope tracks dynamic pressure;
* a **high-speed limit in CAS** (``max_cas_kt``, the ``Vmo`` analogue), binding
  at low altitude;
* a **high-speed limit in Mach** (``max_mach``, the ``Mmo`` analogue), binding at
  high altitude.

Converted to TAS through :mod:`atp.core.speeds` on the ISA atmosphere, these
produce an altitude-dependent TAS band that narrows with height -- the
qualitative behaviour real envelopes have.  This is what makes the atmosphere
model load-bearing rather than decorative: change the atmosphere and the set of
speeds the planner may select genuinely changes.

*The planning speeds* --- the finite set of TAS values the search may actually
select between, ``planning_tas_kt``.  The planner is a discrete search, so the
speed decision must be discrete.  They are given in **TAS** because TAS is the
only airspeed the wind triangle and the coordinated-turn equations accept; a
planning set expressed in Mach or CAS would have to be converted to TAS at every
single transition and would make exact reproduction of a fixed-TAS operating
point impossible.  A planning speed outside the operating limits *at a given
altitude* is simply unavailable there; it is not clamped and not silently
substituted.

The fixed-speed case
--------------------
``SpeedEnvelope.fixed(V)`` is a singleton envelope with no limits.  It is the
default for every :class:`~atp.aircraft.performance.AircraftPerformance` that
does not declare one, which is what makes Milestone 3 reduce **structurally**
rather than coincidentally to Milestone 2: with one selectable speed there is no
decision to make, the speed dimension of the state has one value, and every
quantity is evaluated at the cruise TAS exactly as before.

This is a property of ``.fixed()`` specifically, not of every singleton
envelope.  Constructing ``SpeedEnvelope((450.0,), 450.0, max_mach=0.60)``
directly still yields a one-option envelope -- :attr:`is_fixed` is true, there
is still no speed *decision* -- but it also declares a limit, and that limit
still has to be enforced: at an altitude where 450 kt exceeds the declared
Mach limit, the single available speed is genuinely infeasible there. Having
one option does not mean having no constraints; see :attr:`has_limits`, kept
deliberately independent of :attr:`is_fixed`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..core.atmosphere import AtmosphereDomainError
from ..core.speeds import SpeedDomainError, cas_from_tas, mach_from_tas


@dataclass(frozen=True)
class SpeedEnvelope:
    """A discrete set of selectable TAS values plus the limits bounding them.

    ``planning_tas_kt`` must be strictly ascending, and ``cruise_tas_kt`` must be
    one of its members: the cruise speed is the operating point the rest of the
    package defaults to, so it has to be selectable.
    """

    planning_tas_kt: tuple[float, ...]
    cruise_tas_kt: float
    #: Low-speed operating limit, in CAS.  ``None`` disables the limit.
    min_cas_kt: float | None = None
    #: High-speed operating limit in CAS (a ``Vmo`` analogue).
    max_cas_kt: float | None = None
    #: High-speed operating limit in Mach (an ``Mmo`` analogue).
    max_mach: float | None = None
    name: str = "synthetic-envelope"
    #: Per-altitude availability cache.  Excluded from equality and hashing so
    #: that two envelopes with the same limits compare equal regardless of what
    #: either has been asked about.
    _availability: dict[float, tuple[int, ...]] = field(
        default_factory=dict, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.planning_tas_kt:
            raise ValueError("a speed envelope needs at least one planning speed")
        if any(not math.isfinite(v) for v in self.planning_tas_kt):
            # A NaN or infinite speed is silently invisible to the ordering and
            # positivity checks below: `nan <= 0.0` and `inf <= 0.0` are both
            # False in Python, so either would otherwise slip past as a
            # nonsensical but "valid" planning speed. Checked first and
            # separately so the failure names the actual problem.
            raise ValueError("planning speeds must be finite")
        if any(v <= 0.0 for v in self.planning_tas_kt):
            raise ValueError("planning speeds must be positive")
        ascending = tuple(sorted(set(self.planning_tas_kt)))
        if ascending != tuple(self.planning_tas_kt):
            raise ValueError(
                "planning_tas_kt must be strictly ascending and free of duplicates"
            )
        if not math.isfinite(self.cruise_tas_kt):
            raise ValueError("cruise_tas_kt must be finite")
        if self.cruise_tas_kt not in self.planning_tas_kt:
            raise ValueError(
                f"cruise_tas_kt {self.cruise_tas_kt} is not one of the planning "
                f"speeds {list(self.planning_tas_kt)}"
            )
        for limit_name in ("min_cas_kt", "max_cas_kt", "max_mach"):
            limit = getattr(self, limit_name)
            if limit is None:
                continue
            if not math.isfinite(limit):
                raise ValueError(f"{limit_name} must be finite when set")
            if limit <= 0.0:
                raise ValueError(f"{limit_name} must be positive when set")
        if self.max_mach is not None and self.max_mach >= 1.0:
            # `atp.core.speeds` implements only the subsonic stagnation
            # relation and raises `SpeedDomainError` at or above Mach 1 (see
            # its module docstring). A `max_mach` at or above 1 would declare
            # an operating limit this envelope can never actually evaluate:
            # `tas_is_within_limits` would report every fast speed
            # "unavailable" not because it violates the limit but because the
            # limit itself is outside the domain the conversion is defined on.
            raise ValueError(
                "max_mach must be strictly below Mach 1: the CAS/TAS/Mach "
                "conversions this envelope's limits are evaluated through "
                "(atp.core.speeds) are explicitly subsonic-only"
            )
        if (
            self.min_cas_kt is not None
            and self.max_cas_kt is not None
            and self.min_cas_kt >= self.max_cas_kt
        ):
            raise ValueError("min_cas_kt must be below max_cas_kt")

    # -- construction --------------------------------------------------------
    @staticmethod
    def fixed(tas_kt: float, *, name: str = "fixed-speed") -> "SpeedEnvelope":
        """The degenerate single-speed envelope: no decision, no limits.

        This is the Milestone 2 operating point expressed in Milestone 3 terms.
        """
        return SpeedEnvelope(
            planning_tas_kt=(float(tas_kt),), cruise_tas_kt=float(tas_kt), name=name
        )

    # -- basic accessors -----------------------------------------------------
    @property
    def num_options(self) -> int:
        return len(self.planning_tas_kt)

    @property
    def is_fixed(self) -> bool:
        """True when there is nothing to decide."""
        return self.num_options == 1

    @property
    def cruise_index(self) -> int:
        return self.planning_tas_kt.index(self.cruise_tas_kt)

    @property
    def min_planning_tas_kt(self) -> float:
        return self.planning_tas_kt[0]

    @property
    def max_planning_tas_kt(self) -> float:
        return self.planning_tas_kt[-1]

    def tas_kt(self, index: int) -> float:
        """TAS of a speed option.  Raises on an out-of-range index rather than
        clamping, so a corrupt state cannot be silently planned with."""
        if not 0 <= index < self.num_options:
            raise IndexError(
                f"speed index {index} is outside the envelope's "
                f"{self.num_options} option(s)"
            )
        return self.planning_tas_kt[index]

    def indices(self) -> tuple[int, ...]:
        return tuple(range(self.num_options))

    # -- operating limits ----------------------------------------------------
    @property
    def has_limits(self) -> bool:
        """True when any operating limit is declared.

        Deliberately independent of :attr:`is_fixed`.  "Has a speed decision"
        (more than one selectable option) and "has operating envelope
        constraints" (a CAS or Mach limit that can make even a single option
        infeasible somewhere) are different facts about an envelope, and
        conflating them was a real gap: a singleton envelope that explicitly
        declares a restrictive limit still has to enforce it, even though it
        offers nothing to choose between. See
        :attr:`atp.planning.cost.CostModel.models_speed` for the "has a
        decision" half of this distinction, kept separate on purpose.
        """
        return (
            self.min_cas_kt is not None
            or self.max_cas_kt is not None
            or self.max_mach is not None
        )

    def tas_is_within_limits(self, tas_kt: float, altitude_ft: float) -> bool:
        """Is ``tas_kt`` inside the operating limits at ``altitude_ft``?

        ``True`` when no limit is declared.  A speed whose CAS or Mach cannot be
        evaluated at all -- outside the ISA range, or at/above Mach 1 where the
        subsonic pitot relations do not hold -- is reported as **not** available.
        Refusing to plan with a speed whose limits cannot be evaluated is the
        conservative direction, matching the package's rule elsewhere that an
        unproven number blocks rather than permits.
        """
        if not self.has_limits:
            return True
        try:
            if self.max_mach is not None:
                if mach_from_tas(tas_kt, altitude_ft) > self.max_mach + 1e-12:
                    return False
            if self.min_cas_kt is not None or self.max_cas_kt is not None:
                cas = cas_from_tas(tas_kt, altitude_ft)
                if self.min_cas_kt is not None and cas < self.min_cas_kt - 1e-12:
                    return False
                if self.max_cas_kt is not None and cas > self.max_cas_kt + 1e-12:
                    return False
        except (SpeedDomainError, AtmosphereDomainError):
            return False
        return True

    def is_available(self, index: int, altitude_ft: float) -> bool:
        return self.tas_is_within_limits(self.tas_kt(index), altitude_ft)

    def available_indices(self, altitude_ft: float) -> tuple[int, ...]:
        """Indices selectable at ``altitude_ft``, ascending.  Cached per
        altitude: the planner asks this once per level, not once per edge."""
        key = float(altitude_ft)
        cached = self._availability.get(key)
        if cached is None:
            cached = tuple(
                i for i in range(self.num_options) if self.is_available(i, key)
            )
            self._availability[key] = cached
        return cached

    def describe(self) -> dict[str, object]:
        """Reporting view; every value is synthetic (see the module docstring)."""
        return {
            "name": self.name,
            "planning_tas_kt": list(self.planning_tas_kt),
            "cruise_tas_kt": self.cruise_tas_kt,
            "min_cas_kt": self.min_cas_kt,
            "max_cas_kt": self.max_cas_kt,
            "max_mach": self.max_mach,
            "synthetic": True,
        }


#: A synthetic envelope for the medium twin jet, centred on the 450 kt cruise
#: TAS Milestone 2 used so that the fixed-speed operating point remains
#: selectable.  Not manufacturer data.
SYNTHETIC_JET_ENVELOPE = SpeedEnvelope(
    planning_tas_kt=(330.0, 360.0, 390.0, 420.0, 450.0, 480.0),
    cruise_tas_kt=450.0,
    min_cas_kt=200.0,
    max_cas_kt=340.0,
    max_mach=0.82,
    name="synthetic-jet-envelope",
)

#: The same for the turboprop, centred on its 280 kt cruise TAS.
SYNTHETIC_TURBOPROP_ENVELOPE = SpeedEnvelope(
    planning_tas_kt=(200.0, 240.0, 280.0, 310.0),
    cruise_tas_kt=280.0,
    min_cas_kt=120.0,
    max_cas_kt=250.0,
    max_mach=0.55,
    name="synthetic-turboprop-envelope",
)

ENVELOPE_LIBRARY: dict[str, SpeedEnvelope] = {
    SYNTHETIC_JET_ENVELOPE.name: SYNTHETIC_JET_ENVELOPE,
    SYNTHETIC_TURBOPROP_ENVELOPE.name: SYNTHETIC_TURBOPROP_ENVELOPE,
}
