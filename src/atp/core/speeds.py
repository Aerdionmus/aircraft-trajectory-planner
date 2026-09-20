"""Airspeed conversions on the ISA atmosphere.

Three airspeeds are used in this package and they must not be conflated:

TAS
    True airspeed: the speed of the aircraft through the air mass.  This is the
    only one the wind triangle (:mod:`atp.aircraft.kinematics`) and the
    coordinated-turn equations (:mod:`atp.aircraft.turn`) accept, because both
    are statements about motion relative to the air.
Mach
    ``M = TAS / a`` with ``a`` the local speed of sound.  Compressibility and
    the high-speed end of an operating envelope are naturally expressed here.
CAS
    Calibrated airspeed: what a (perfect) pitot-static airspeed indicator would
    read.  Low-speed limits and the low-altitude end of an envelope are
    naturally expressed here because they track dynamic pressure.

Every function takes an altitude and evaluates the atmosphere at it, so every
conversion is explicitly tied to a stated atmospheric state.  There is no
"standard day offset" knob: the atmosphere is ISA, full stop.

Derivation
----------
For subsonic compressible flow, isentropic stagnation gives the impact
(differential) pressure

    qc = p * ((1 + (gamma-1)/2 * M^2) ^ (gamma/(gamma-1)) - 1)

With ``gamma = 1.4`` the exponents are ``0.2`` and ``3.5``.  CAS is *defined*
as the speed that produces the same ``qc`` at sea-level standard conditions:

    qc = p0 * ((1 + 0.2 * (CAS/a0)^2) ^ 3.5 - 1)

Both relations are strictly increasing in their speed argument over the
subsonic range, so each inverts uniquely:

    M   = sqrt(5 * ((qc/p  + 1) ^ (2/7) - 1))
    CAS = a0 * sqrt(5 * ((qc/p0 + 1) ^ (2/7) - 1))

Domain and assumptions
----------------------
* **Subsonic only.**  The stagnation relation above is the subsonic branch; at
  and above ``M = 1`` a normal shock forms ahead of the probe and the Rayleigh
  supersonic pitot formula applies instead.  That branch is *not* implemented,
  and every function raises :class:`SpeedDomainError` at ``M >= 1`` rather than
  returning a number from the wrong formula.  Every speed the package plans
  with is well below ``M = 0.9``.
* **Perfect gas, dry air, no instrument error.**  CAS here is the ideal
  calibrated airspeed; position error, compressibility correction tables and
  the CAS/EAS distinction at low dynamic pressure are not modelled.
* Altitude validity is delegated to :mod:`atp.core.atmosphere`, which raises
  outside its two modelled layers.

Every ``x_from_y`` here has an exact inverse ``y_from_x``; the round trips are
asserted to 1e-9 relative in ``tests/test_speeds.py``.
"""

from __future__ import annotations

import math

from .atmosphere import A0_KT, GAMMA, P0_PA, isa

#: ``(gamma - 1) / 2``.
_HALF_GAMMA_MINUS_1: float = 0.5 * (GAMMA - 1.0)
#: ``gamma / (gamma - 1)``.
_STAGNATION_EXPONENT: float = GAMMA / (GAMMA - 1.0)
#: ``(gamma - 1) / gamma``, the inverse exponent.
_INVERSE_EXPONENT: float = 1.0 / _STAGNATION_EXPONENT
#: ``2 / (gamma - 1)``.
_TWO_OVER_GAMMA_MINUS_1: float = 2.0 / (GAMMA - 1.0)

#: Largest Mach number the subsonic relations are applied to.
MAX_SUBSONIC_MACH: float = 1.0


class SpeedDomainError(ValueError):
    """Raised for a speed outside the subsonic domain these relations hold on."""


def _check_mach(mach: float, what: str) -> float:
    if not math.isfinite(mach) or mach < 0.0:
        raise SpeedDomainError(f"{what} must be finite and non-negative")
    if mach >= MAX_SUBSONIC_MACH:
        raise SpeedDomainError(
            f"{what} of Mach {mach:.4f} is at or above Mach 1; the subsonic "
            "pitot relations implemented here do not apply"
        )
    return mach


def _check_speed(speed_kt: float, what: str) -> float:
    if not math.isfinite(speed_kt) or speed_kt < 0.0:
        raise SpeedDomainError(f"{what} must be finite and non-negative")
    return speed_kt


def _impact_pressure_pa(speed_kt: float, sonic_kt: float, pressure_pa: float) -> float:
    """``qc`` from a speed, the local speed of sound and the local pressure."""
    mach = speed_kt / sonic_kt
    return pressure_pa * (
        (1.0 + _HALF_GAMMA_MINUS_1 * mach * mach) ** _STAGNATION_EXPONENT - 1.0
    )


def _mach_from_impact_pressure(qc_pa: float, pressure_pa: float) -> float:
    """Invert :func:`_impact_pressure_pa` for the Mach number."""
    ratio = qc_pa / pressure_pa + 1.0
    return math.sqrt(
        _TWO_OVER_GAMMA_MINUS_1 * (ratio**_INVERSE_EXPONENT - 1.0)
    )


# -- TAS <-> Mach ------------------------------------------------------------
def mach_from_tas(tas_kt: float, altitude_ft: float) -> float:
    """``M = TAS / a(altitude)``."""
    _check_speed(tas_kt, "TAS")
    mach = tas_kt / isa(altitude_ft).speed_of_sound_kt
    return _check_mach(mach, "TAS")


def tas_from_mach(mach: float, altitude_ft: float) -> float:
    """``TAS = M a(altitude)``."""
    _check_mach(mach, "Mach number")
    return mach * isa(altitude_ft).speed_of_sound_kt


# -- CAS <-> TAS -------------------------------------------------------------
def cas_from_tas(tas_kt: float, altitude_ft: float) -> float:
    """Calibrated airspeed equivalent to ``tas_kt`` at ``altitude_ft``."""
    _check_speed(tas_kt, "TAS")
    state = isa(altitude_ft)
    _check_mach(tas_kt / state.speed_of_sound_kt, "TAS")
    qc = _impact_pressure_pa(tas_kt, state.speed_of_sound_kt, state.pressure_pa)
    return A0_KT * _mach_from_impact_pressure(qc, P0_PA)


def tas_from_cas(cas_kt: float, altitude_ft: float) -> float:
    """True airspeed equivalent to ``cas_kt`` at ``altitude_ft``."""
    _check_speed(cas_kt, "CAS")
    _check_mach(cas_kt / A0_KT, "CAS")
    state = isa(altitude_ft)
    qc = _impact_pressure_pa(cas_kt, A0_KT, P0_PA)
    mach = _mach_from_impact_pressure(qc, state.pressure_pa)
    _check_mach(mach, "the TAS implied by this CAS")
    return mach * state.speed_of_sound_kt


# -- CAS <-> Mach ------------------------------------------------------------
def cas_from_mach(mach: float, altitude_ft: float) -> float:
    return cas_from_tas(tas_from_mach(mach, altitude_ft), altitude_ft)


def mach_from_cas(cas_kt: float, altitude_ft: float) -> float:
    return mach_from_tas(tas_from_cas(cas_kt, altitude_ft), altitude_ft)
