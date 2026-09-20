"""M3-REQ-001 / M3-REQ-013: the ISA model, checked against the standard itself.

The reference values below are the *defining* values of the International
Standard Atmosphere, not measurements and not outputs of this implementation.
Sea-level conditions are exact by definition; the values at altitude are taken
from the closed-form laws the standard specifies and are reproduced here to the
precision a double-precision evaluation of those laws supports.
"""

from __future__ import annotations

import math

import pytest

from atp.core.atmosphere import (
    A0_KT,
    A0_MPS,
    ISA_MAX_ALTITUDE_FT,
    ISA_MIN_ALTITUDE_FT,
    P0_PA,
    RHO0_KG_PER_M3,
    T0_K,
    T_TROPOPAUSE_K,
    AtmosphereDomainError,
    density_kg_per_m3,
    isa,
    pressure_pa,
    speed_of_sound_kt,
    temperature_k,
)

#: Altitudes the shipped scenarios actually use, plus the layer boundary.
SAMPLE_FT = [0.0, 10000.0, 20000.0, 30000.0, 36089.24, 41000.0, 60000.0]


def test_sea_level_matches_the_defining_reference_values():
    state = isa(0.0)
    assert state.temperature_k == pytest.approx(288.15, abs=1e-12)
    assert state.pressure_pa == pytest.approx(101325.0, abs=1e-9)
    # Density is derived from p and T rather than asserted, so agreement with
    # the standard's quoted 1.225 kg/m^3 is a consistency check on the gas
    # constant, not a tautology.
    assert state.density_kg_per_m3 == pytest.approx(1.225, rel=1e-7)
    assert state.speed_of_sound_mps == pytest.approx(340.294, abs=1e-3)
    assert state.speed_of_sound_kt == pytest.approx(661.4786, abs=1e-3)
    assert state.pressure_ratio == pytest.approx(1.0)
    assert state.density_ratio == pytest.approx(1.0, rel=1e-7)
    assert state.temperature_ratio == pytest.approx(1.0)


def test_module_constants_agree_with_the_sea_level_state():
    assert T0_K == 288.15
    assert P0_PA == 101325.0
    assert RHO0_KG_PER_M3 == 1.225
    assert A0_MPS == pytest.approx(isa(0.0).speed_of_sound_mps, rel=1e-15)
    assert A0_KT == pytest.approx(isa(0.0).speed_of_sound_kt, rel=1e-15)


def test_the_tropopause_temperature_is_exactly_the_standard_value():
    """216.65 K is a defining value; it must fall out of the lapse rate rather
    than be hard-coded independently of it."""
    assert T_TROPOPAUSE_K == pytest.approx(216.65, abs=1e-9)


def test_known_altitudes_reproduce_standard_table_values():
    """Spot values from the standard's own laws, to table precision."""
    assert temperature_k(30000.0) == pytest.approx(228.714, abs=1e-3)
    assert pressure_pa(30000.0) == pytest.approx(30089.6, rel=1e-4)
    assert density_kg_per_m3(30000.0) == pytest.approx(0.45831, rel=1e-4)
    assert speed_of_sound_kt(30000.0) == pytest.approx(589.322, abs=1e-2)
    # Above the tropopause the temperature stops falling.
    assert temperature_k(41000.0) == pytest.approx(216.65, abs=1e-9)
    assert speed_of_sound_kt(41000.0) == pytest.approx(
        speed_of_sound_kt(38000.0), rel=1e-12
    )


def test_density_and_pressure_decrease_strictly_with_altitude():
    """Monotonicity is what the admissible bounds and the envelope both lean
    on, so it is asserted across the whole modelled range, not spot-checked."""
    altitudes = [
        ISA_MIN_ALTITUDE_FT + 1.0 + i * (ISA_MAX_ALTITUDE_FT - ISA_MIN_ALTITUDE_FT - 2.0) / 200.0
        for i in range(201)
    ]
    densities = [density_kg_per_m3(a) for a in altitudes]
    pressures = [pressure_pa(a) for a in altitudes]
    for lower, upper in zip(densities, densities[1:]):
        assert upper < lower
    for lower, upper in zip(pressures, pressures[1:]):
        assert upper < lower
    assert all(value > 0.0 for value in densities + pressures)


def test_temperature_falls_then_holds_constant():
    tropopause_ft = 11000.0 / 0.3048
    below = [temperature_k(a) for a in (0.0, 10000.0, 20000.0, 30000.0)]
    for lower, upper in zip(below, below[1:]):
        assert upper < lower
    above = [temperature_k(tropopause_ft + d) for d in (100.0, 5000.0, 10000.0)]
    assert all(t == pytest.approx(T_TROPOPAUSE_K, abs=1e-9) for t in above)


def test_the_two_layers_join_continuously():
    """A discontinuity at the tropopause would show up as a jump in any
    quantity derived from pressure, including the speed envelope's limits."""
    tropopause_ft = 11000.0 / 0.3048
    below = isa(tropopause_ft - 1e-4)
    above = isa(tropopause_ft + 1e-4)
    # The tolerance is the genuine hydrostatic gradient across the 2e-4 ft gap
    # (about 3.6 Pa/m, i.e. 2e-4 Pa here), not slack: a real discontinuity at a
    # layer join would be orders of magnitude larger than this.
    assert below.pressure_pa == pytest.approx(above.pressure_pa, rel=1e-7)
    assert below.temperature_k == pytest.approx(above.temperature_k, rel=1e-7)
    assert below.density_kg_per_m3 == pytest.approx(above.density_kg_per_m3, rel=1e-7)


def test_speed_of_sound_tracks_the_square_root_of_temperature():
    for altitude in SAMPLE_FT:
        state = isa(altitude)
        assert state.speed_of_sound_mps == pytest.approx(
            math.sqrt(1.4 * 287.05287 * state.temperature_k), rel=1e-12
        )


@pytest.mark.parametrize(
    "altitude_ft",
    [ISA_MIN_ALTITUDE_FT - 1.0, ISA_MAX_ALTITUDE_FT + 1.0, 100000.0, -50000.0],
)
def test_altitudes_outside_the_modelled_range_are_refused(altitude_ft):
    """Extrapolating a law outside the layer it was derived for is exactly the
    kind of silent unsoundness this package refuses elsewhere."""
    with pytest.raises(AtmosphereDomainError):
        isa(altitude_ft)


@pytest.mark.parametrize("altitude_ft", [math.nan, math.inf, -math.inf])
def test_non_finite_altitudes_are_refused(altitude_ft):
    with pytest.raises(AtmosphereDomainError):
        isa(altitude_ft)


def test_the_range_boundaries_themselves_are_inside_the_domain():
    assert isa(ISA_MIN_ALTITUDE_FT).temperature_k > 0.0
    assert isa(ISA_MAX_ALTITUDE_FT).temperature_k > 0.0


def test_the_service_ceiling_of_every_shipped_aircraft_is_inside_the_domain():
    from atp.aircraft.performance import AIRCRAFT_LIBRARY

    for aircraft in AIRCRAFT_LIBRARY.values():
        assert aircraft.service_ceiling_ft <= ISA_MAX_ALTITUDE_FT
        isa(aircraft.service_ceiling_ft)


def test_the_atmosphere_is_deterministic():
    first = [isa(a) for a in SAMPLE_FT]
    second = [isa(a) for a in SAMPLE_FT]
    assert first == second
