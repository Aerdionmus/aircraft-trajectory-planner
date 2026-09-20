"""M3-REQ-002 / M3-REQ-013: the TAS/CAS/Mach conversions.

Every conversion the package supports has an exact inverse, and the round trips
are asserted rather than assumed.  The conversions also have to reproduce the
two identities that pin them to reality: at sea level CAS and TAS coincide, and
Mach is TAS divided by the local speed of sound.
"""

from __future__ import annotations

import math

import pytest

from atp.core.atmosphere import A0_KT, AtmosphereDomainError, speed_of_sound_kt
from atp.core.speeds import (
    SpeedDomainError,
    cas_from_mach,
    cas_from_tas,
    mach_from_cas,
    mach_from_tas,
    tas_from_cas,
    tas_from_mach,
)

ALTITUDES_FT = [0.0, 5000.0, 15000.0, 25000.0, 30000.0, 36089.0, 41000.0]
TAS_KT = [150.0, 250.0, 330.0, 400.0, 450.0, 480.0, 520.0]
#: Deliberately capped at 280 kt.  A higher CAS is genuinely supersonic near
#: the top of the altitude range -- 340 kt CAS at FL410 is about Mach 1.09 --
#: so a round-trip sweep including it would be asserting that the conversions
#: work outside the domain they are derived for.  That case is covered
#: separately by ``test_a_cas_implying_a_supersonic_tas_is_refused``.
CAS_KT = [120.0, 180.0, 240.0, 280.0]
MACHS = [0.2, 0.4, 0.6, 0.75, 0.82, 0.9]


# -- identities that anchor the conversions ----------------------------------
def test_cas_equals_tas_at_sea_level():
    """CAS is defined as the speed giving the same impact pressure at sea-level
    standard conditions, so at sea level the conversion must be the identity."""
    for tas in TAS_KT:
        assert cas_from_tas(tas, 0.0) == pytest.approx(tas, rel=1e-12)
        assert tas_from_cas(tas, 0.0) == pytest.approx(tas, rel=1e-12)


def test_mach_is_tas_over_the_local_speed_of_sound():
    for altitude in ALTITUDES_FT:
        for tas in TAS_KT:
            assert mach_from_tas(tas, altitude) == pytest.approx(
                tas / speed_of_sound_kt(altitude), rel=1e-12
            )


def test_zero_speed_maps_to_zero_everywhere():
    for altitude in ALTITUDES_FT:
        assert cas_from_tas(0.0, altitude) == pytest.approx(0.0, abs=1e-12)
        assert tas_from_cas(0.0, altitude) == pytest.approx(0.0, abs=1e-12)
        assert mach_from_tas(0.0, altitude) == pytest.approx(0.0, abs=1e-12)


# -- round trips -------------------------------------------------------------
@pytest.mark.parametrize("altitude_ft", ALTITUDES_FT)
def test_tas_mach_round_trip(altitude_ft):
    for tas in TAS_KT:
        back = tas_from_mach(mach_from_tas(tas, altitude_ft), altitude_ft)
        assert back == pytest.approx(tas, rel=1e-12)


@pytest.mark.parametrize("altitude_ft", ALTITUDES_FT)
def test_mach_tas_round_trip(altitude_ft):
    for mach in MACHS:
        back = mach_from_tas(tas_from_mach(mach, altitude_ft), altitude_ft)
        assert back == pytest.approx(mach, rel=1e-12)


@pytest.mark.parametrize("altitude_ft", ALTITUDES_FT)
def test_cas_tas_round_trip(altitude_ft):
    for cas in CAS_KT:
        back = cas_from_tas(tas_from_cas(cas, altitude_ft), altitude_ft)
        assert back == pytest.approx(cas, rel=1e-9)


@pytest.mark.parametrize("altitude_ft", ALTITUDES_FT)
def test_tas_cas_round_trip(altitude_ft):
    for tas in TAS_KT:
        back = tas_from_cas(cas_from_tas(tas, altitude_ft), altitude_ft)
        assert back == pytest.approx(tas, rel=1e-9)


@pytest.mark.parametrize("altitude_ft", ALTITUDES_FT)
def test_cas_mach_round_trip(altitude_ft):
    for cas in CAS_KT:
        back = cas_from_mach(mach_from_cas(cas, altitude_ft), altitude_ft)
        assert back == pytest.approx(cas, rel=1e-9)


# -- physical behaviour ------------------------------------------------------
def test_tas_exceeds_cas_above_sea_level_and_the_gap_widens_with_altitude():
    """Air thins with height, so a given CAS corresponds to an increasing TAS.
    This is the behaviour that makes the envelope's limits altitude-dependent."""
    ratios = [tas_from_cas(250.0, a) / 250.0 for a in ALTITUDES_FT]
    assert ratios[0] == pytest.approx(1.0, rel=1e-12)
    for lower, upper in zip(ratios, ratios[1:]):
        assert upper > lower


def test_a_constant_mach_corresponds_to_a_falling_cas_with_altitude():
    cas = [cas_from_mach(0.78, a) for a in ALTITUDES_FT]
    for lower, upper in zip(cas, cas[1:]):
        assert upper < lower


def test_each_conversion_is_strictly_increasing():
    """Monotonicity is what makes each relation uniquely invertible; the
    inverses above would be ill-defined without it."""
    for altitude in ALTITUDES_FT:
        values = [cas_from_tas(v, altitude) for v in TAS_KT]
        for lower, upper in zip(values, values[1:]):
            assert upper > lower


# -- domains -----------------------------------------------------------------
@pytest.mark.parametrize("bad", [-1.0, -250.0, math.nan, math.inf])
def test_negative_and_non_finite_speeds_are_refused(bad):
    with pytest.raises(SpeedDomainError):
        cas_from_tas(bad, 30000.0)
    with pytest.raises(SpeedDomainError):
        tas_from_cas(bad, 30000.0)


def test_supersonic_inputs_are_refused_rather_than_answered_wrongly():
    """The implemented stagnation relation is the subsonic branch.  Returning a
    number from it above Mach 1 would be the wrong formula silently applied."""
    sonic = speed_of_sound_kt(30000.0)
    with pytest.raises(SpeedDomainError, match="Mach"):
        mach_from_tas(sonic * 1.2, 30000.0)
    with pytest.raises(SpeedDomainError, match="Mach"):
        cas_from_tas(sonic * 1.05, 30000.0)
    with pytest.raises(SpeedDomainError, match="Mach"):
        tas_from_mach(1.4, 30000.0)
    # Exactly Mach 1 is outside the domain too, not a boundary case that slips
    # through.
    with pytest.raises(SpeedDomainError):
        tas_from_mach(1.0, 30000.0)


def test_a_cas_implying_a_supersonic_tas_is_refused():
    """At altitude a high CAS can imply a supersonic TAS even though the CAS
    itself is subsonic at sea level; the check has to be on the result."""
    with pytest.raises(SpeedDomainError):
        tas_from_cas(A0_KT * 0.95, 41000.0)


def test_altitudes_outside_the_atmosphere_propagate_the_domain_error():
    with pytest.raises(AtmosphereDomainError):
        cas_from_tas(450.0, 100000.0)
    with pytest.raises(AtmosphereDomainError):
        mach_from_tas(450.0, 100000.0)


def test_conversions_are_deterministic():
    first = [cas_from_tas(450.0, a) for a in ALTITUDES_FT]
    second = [cas_from_tas(450.0, a) for a in ALTITUDES_FT]
    assert first == second
