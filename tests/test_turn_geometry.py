"""T-6 to T-9: the coordinated-turn equations, checked against hand values.

These are pure-physics tests: no grid, no airspace, no search.
"""

from __future__ import annotations

import math

import pytest

from atp.aircraft.performance import MEDIUM_TWIN_JET, REGIONAL_TURBOPROP
from atp.aircraft.turn import (
    corner_cut_nm,
    tangent_length_nm,
    turn_fits,
    turn_radius_nm,
    turn_rate_rad_per_h,
    turn_time_h,
)
from atp.core.units import G_NM_PER_H2, S_PER_H

#: (TAS kt, bank deg, radius NM, rate deg/s, 45-degree tangent NM), computed by
#: hand from R = V^2 / (g tan phi) and omega = g tan phi / V.
HAND_VALUES = [
    (450.0, 15.0, 11.01, 0.65, 4.56),
    (450.0, 20.0, 8.11, 0.88, 3.36),
    (450.0, 25.0, 6.33, 1.13, 2.62),
    (450.0, 30.0, 5.11, 1.40, 2.12),
    (450.0, 35.0, 4.21, 1.70, 1.75),
    (280.0, 25.0, 2.45, 1.82, 1.01),
]


def test_gravity_constant_round_trips_to_si():
    """T-7: the one new conversion factor, checked both ways."""
    metres_per_second_squared = G_NM_PER_H2 * 1852.0 / (S_PER_H * S_PER_H)
    assert metres_per_second_squared == pytest.approx(9.80665, rel=1e-12)
    assert G_NM_PER_H2 == pytest.approx(68625.369, abs=1e-3)


@pytest.mark.parametrize("tas,bank,radius,rate_deg_s,tangent", HAND_VALUES)
def test_turn_radius_and_rate_match_hand_computed_values(
    tas, bank, radius, rate_deg_s, tangent
):
    """T-6."""
    assert turn_radius_nm(tas, bank) == pytest.approx(radius, abs=0.01)
    omega = turn_rate_rad_per_h(tas, bank)
    assert math.degrees(omega) / S_PER_H == pytest.approx(rate_deg_s, abs=0.01)
    assert tangent_length_nm(
        turn_radius_nm(tas, bank), math.radians(45.0)
    ) == pytest.approx(tangent, abs=0.01)


@pytest.mark.parametrize("tas,bank,_r,_w,_t", HAND_VALUES)
def test_rate_times_radius_recovers_airspeed(tas, bank, _r, _w, _t):
    """T-6: omega * R == V is the dimensional round trip."""
    assert turn_rate_rad_per_h(tas, bank) * turn_radius_nm(tas, bank) == pytest.approx(
        tas, rel=1e-12
    )


def test_a_fast_jet_cannot_achieve_rate_one_within_its_bank_limit():
    """Sanity: 3 deg/s is a low-speed construct.  A 450 kt jet at 25 deg banks
    at about 1.1 deg/s; a 280 kt turboprop gets closer but still falls short."""
    jet = math.degrees(turn_rate_rad_per_h(450.0, 25.0)) / S_PER_H
    prop = math.degrees(turn_rate_rad_per_h(280.0, 25.0)) / S_PER_H
    assert jet < prop < 3.0


def test_radius_and_rate_are_monotone_in_bank():
    """T-8."""
    radii = [turn_radius_nm(450.0, b) for b in (15.0, 20.0, 25.0, 30.0, 35.0)]
    rates = [turn_rate_rad_per_h(450.0, b) for b in (15.0, 20.0, 25.0, 30.0, 35.0)]
    assert radii == sorted(radii, reverse=True)
    assert rates == sorted(rates)


def test_invalid_bank_and_airspeed_are_rejected():
    for bad_bank in (0.0, -5.0, 90.0, 120.0):
        with pytest.raises(ValueError):
            turn_radius_nm(450.0, bad_bank)
        with pytest.raises(ValueError):
            turn_rate_rad_per_h(450.0, bad_bank)
    with pytest.raises(ValueError):
        turn_radius_nm(0.0, 25.0)


def test_tangent_length_vanishes_for_a_null_turn_and_diverges_at_reversal():
    """T-9."""
    radius = turn_radius_nm(450.0, 25.0)
    assert tangent_length_nm(radius, 0.0) == 0.0
    assert tangent_length_nm(radius, 1e-9) == pytest.approx(0.0, abs=1e-8)
    assert math.isinf(tangent_length_nm(radius, math.pi))
    assert tangent_length_nm(radius, math.radians(179.9)) > 1000.0


def test_a_course_reversal_never_fits_any_leg():
    """T-9: tan(90 deg) must be handled, not overflow."""
    radius = turn_radius_nm(450.0, 25.0)
    assert not turn_fits(radius, math.pi, 1e6, 1e6)


def test_turn_fits_is_exactly_the_closed_form():
    """T-23 (geometry half): the gate is its own equation, not an approximation."""
    for bank in (15.0, 25.0, 35.0):
        radius = turn_radius_nm(450.0, bank)
        for degrees in (10.0, 45.0, 90.0, 135.0):
            dpsi = math.radians(degrees)
            needed = radius * math.tan(dpsi / 2.0)
            assert turn_fits(radius, dpsi, 2 * needed * 1.001, 1e6)
            assert not turn_fits(radius, dpsi, 2 * needed * 0.999, 1e6)


def test_turn_time_is_proportional_and_non_negative():
    omega = turn_rate_rad_per_h(450.0, 25.0)
    assert turn_time_h(0.0, omega) == 0.0
    assert turn_time_h(math.radians(45.0), omega) * S_PER_H == pytest.approx(
        39.8, abs=0.1
    )
    assert turn_time_h(math.radians(90.0), omega) == pytest.approx(
        2 * turn_time_h(math.radians(45.0), omega)
    )
    assert turn_time_h(-math.radians(45.0), omega) > 0.0


def test_corner_cut_is_non_negative_and_grows_with_the_turn():
    """The fly-by shortening is measured; ``docs/turn_model.md`` records why it
    is never credited."""
    radius = turn_radius_nm(450.0, 25.0)
    assert corner_cut_nm(radius, 0.0) == 0.0
    cuts = [corner_cut_nm(radius, math.radians(d)) for d in (10, 45, 90, 135)]
    assert all(c >= 0.0 for c in cuts)
    assert cuts == sorted(cuts)


def test_aircraft_expose_turn_capability_consistently():
    for aircraft in (MEDIUM_TWIN_JET, REGIONAL_TURBOPROP):
        alt = 25000.0
        assert aircraft.turn_radius_nm(alt) == pytest.approx(
            turn_radius_nm(aircraft.tas_kt(alt), aircraft.max_bank_deg)
        )
        assert aircraft.turn_rate_rad_per_h(alt) == pytest.approx(
            turn_rate_rad_per_h(aircraft.tas_kt(alt), aircraft.max_bank_deg)
        )


def test_aircraft_rejects_an_impossible_bank_or_fuel_factor():
    from dataclasses import replace

    with pytest.raises(ValueError):
        replace(MEDIUM_TWIN_JET, max_bank_deg=95.0)
    with pytest.raises(ValueError):
        replace(MEDIUM_TWIN_JET, max_bank_deg=0.0)
    with pytest.raises(ValueError):
        replace(MEDIUM_TWIN_JET, turn_fuel_factor=0.5)
