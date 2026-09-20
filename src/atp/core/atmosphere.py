"""International Standard Atmosphere (ISA).

Scope, stated once
------------------
This is the *standard* atmosphere: a deterministic, closed-form function of
geopotential altitude alone.  There is no ISA deviation, no temperature offset,
no humidity, no real-weather ingest and no time dimension.  Nothing in this
module reads data from anywhere; every number below is a defining constant of
the standard, not a measurement.  ``docs/assumptions.md`` records what that
excludes.

The model is the two-layer subset the project's altitude range actually needs:

======================  ===============  ==========================
Layer                   Geopotential     Behaviour
======================  ===============  ==========================
Troposphere             -2000..11000 m   linear lapse, ``L = 6.5 K/km``
Lower stratosphere      11000..20000 m   isothermal at 216.65 K
======================  ===============  ==========================

Above 20 km the standard continues with further layers; they are **not**
implemented, and :func:`isa` raises rather than extrapolating.  The shipped
scenarios operate between FL200 and FL410, comfortably inside the modelled
range.

Derivation
----------
Hydrostatic equilibrium with the perfect-gas law, ``dp/dh = -rho g`` and
``p = rho R T``, integrates in a constant-lapse layer to

    T(h) = T_b - L (h - h_b)
    p(h) = p_b (T(h) / T_b) ^ (g0 / (L R))

and in an isothermal layer to

    T(h) = T_b
    p(h) = p_b exp(-g0 (h - h_b) / (R T_b))

Density follows from the gas law and the speed of sound from
``a = sqrt(gamma R T)``.

Units
-----
Altitude is accepted in **feet**, matching the rest of the package.  Returned
quantities are SI (K, Pa, kg/m^3, m/s) because that is the form the defining
constants are quoted in; :func:`speed_of_sound_kt` is provided because the
speed conversions in :mod:`atp.core.speeds` work in knots.  Geopotential and
geometric altitude are **not** distinguished.  The distinction is intentionally
omitted rather than quantified here: over the altitude range this package
operates in, it is below the fidelity already implied by the rest of this
synthetic model -- a constant-mass aircraft, a fixed-lapse-rate atmosphere with
no deviation, and aircraft performance coefficients that are themselves
representative rather than measured -- and treating the input as geopotential
keeps the closed forms above exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .units import M_PER_FT, mps_to_kt

# -- defining constants of the standard --------------------------------------
#: Sea-level standard temperature [K].
T0_K: float = 288.15
#: Sea-level standard pressure [Pa].
P0_PA: float = 101325.0
#: Sea-level standard density [kg/m^3].  Consistent with ``P0/(R T0)``.
RHO0_KG_PER_M3: float = 1.225
#: Tropospheric temperature lapse rate [K/m].
LAPSE_K_PER_M: float = 0.0065
#: Standard gravity [m/s^2].
G0_MPS2: float = 9.80665
#: Specific gas constant for dry air [J/(kg K)].
R_AIR: float = 287.05287
#: Ratio of specific heats for dry air [-].
GAMMA: float = 1.4

#: Geopotential altitude of the tropopause [m].
TROPOPAUSE_M: float = 11000.0
#: Temperature of the isothermal lower stratosphere [K].
T_TROPOPAUSE_K: float = T0_K - LAPSE_K_PER_M * TROPOPAUSE_M  # 216.65 K exactly

#: Lowest altitude the model is defined for [m] / [ft].
ISA_MIN_ALTITUDE_M: float = -2000.0
#: Highest altitude the model is defined for [m] / [ft]: the top of the
#: isothermal layer.  Above this the standard has further layers that are
#: deliberately not implemented.
ISA_MAX_ALTITUDE_M: float = 20000.0
ISA_MIN_ALTITUDE_FT: float = ISA_MIN_ALTITUDE_M / M_PER_FT
ISA_MAX_ALTITUDE_FT: float = ISA_MAX_ALTITUDE_M / M_PER_FT

#: ``g0 / (L R)``, the exponent of the tropospheric pressure law.
_PRESSURE_EXPONENT: float = G0_MPS2 / (LAPSE_K_PER_M * R_AIR)

#: Sea-level standard speed of sound [m/s]: ``sqrt(gamma R T0)``.
A0_MPS: float = math.sqrt(GAMMA * R_AIR * T0_K)
A0_KT: float = mps_to_kt(A0_MPS)

#: Pressure at the tropopause [Pa], the base of the isothermal layer.
_P_TROPOPAUSE_PA: float = P0_PA * (T_TROPOPAUSE_K / T0_K) ** _PRESSURE_EXPONENT


class AtmosphereDomainError(ValueError):
    """Raised for an altitude outside the modelled range.

    A distinct type so that callers can tell "outside the atmosphere model" from
    any other ``ValueError``; the speed envelope relies on this.
    """


@dataclass(frozen=True, slots=True)
class AtmosphereState:
    """The four ISA quantities at one altitude, computed once."""

    altitude_ft: float
    temperature_k: float
    pressure_pa: float
    density_kg_per_m3: float
    speed_of_sound_mps: float

    @property
    def speed_of_sound_kt(self) -> float:
        return mps_to_kt(self.speed_of_sound_mps)

    @property
    def pressure_ratio(self) -> float:
        """``delta = p / p0``."""
        return self.pressure_pa / P0_PA

    @property
    def density_ratio(self) -> float:
        """``sigma = rho / rho0``."""
        return self.density_kg_per_m3 / RHO0_KG_PER_M3

    @property
    def temperature_ratio(self) -> float:
        """``theta = T / T0``."""
        return self.temperature_k / T0_K


def _check_altitude(altitude_ft: float) -> float:
    if not math.isfinite(altitude_ft):
        raise AtmosphereDomainError("altitude must be finite")
    altitude_m = altitude_ft * M_PER_FT
    if altitude_m < ISA_MIN_ALTITUDE_M - 1e-9 or altitude_m > ISA_MAX_ALTITUDE_M + 1e-9:
        raise AtmosphereDomainError(
            f"altitude {altitude_ft:.0f} ft is outside the modelled ISA range "
            f"[{ISA_MIN_ALTITUDE_FT:.0f}, {ISA_MAX_ALTITUDE_FT:.0f}] ft"
        )
    return altitude_m


def isa(altitude_ft: float) -> AtmosphereState:
    """Full ISA state at ``altitude_ft``.

    Raises :class:`AtmosphereDomainError` outside
    ``[ISA_MIN_ALTITUDE_FT, ISA_MAX_ALTITUDE_FT]`` rather than extrapolating a
    law that does not hold there.
    """
    altitude_m = _check_altitude(altitude_ft)
    if altitude_m <= TROPOPAUSE_M:
        temperature = T0_K - LAPSE_K_PER_M * altitude_m
        pressure = P0_PA * (temperature / T0_K) ** _PRESSURE_EXPONENT
    else:
        temperature = T_TROPOPAUSE_K
        pressure = _P_TROPOPAUSE_PA * math.exp(
            -G0_MPS2 * (altitude_m - TROPOPAUSE_M) / (R_AIR * T_TROPOPAUSE_K)
        )
    density = pressure / (R_AIR * temperature)
    speed_of_sound = math.sqrt(GAMMA * R_AIR * temperature)
    return AtmosphereState(
        altitude_ft=altitude_ft,
        temperature_k=temperature,
        pressure_pa=pressure,
        density_kg_per_m3=density,
        speed_of_sound_mps=speed_of_sound,
    )


def temperature_k(altitude_ft: float) -> float:
    return isa(altitude_ft).temperature_k


def pressure_pa(altitude_ft: float) -> float:
    return isa(altitude_ft).pressure_pa


def density_kg_per_m3(altitude_ft: float) -> float:
    return isa(altitude_ft).density_kg_per_m3


def speed_of_sound_mps(altitude_ft: float) -> float:
    return isa(altitude_ft).speed_of_sound_mps


def speed_of_sound_kt(altitude_ft: float) -> float:
    return mps_to_kt(isa(altitude_ft).speed_of_sound_mps)
