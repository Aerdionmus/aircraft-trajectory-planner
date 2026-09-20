"""M3-REQ-005 / M3-REQ-006: performance and turn geometry respond to speed.

Two claims are checked here and they are separate:

* **fuel flow varies with speed**, with the shape the power-required split
  implies -- an interior minimum, and a *different* speed for best specific
  range, which is what makes minimum-time and minimum-fuel plans differ;
* **turn geometry varies with speed**, through the same coordinated-turn
  relations Milestone 2 established, now evaluated at the selected TAS.

Both must reduce exactly to the Milestone 2 values at the reference speed.  That
reduction is asserted at the tightest tolerance available, because fixed-speed
parity depends on it being exact rather than close.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from atp.aircraft.envelope import SpeedEnvelope
from atp.aircraft.performance import MEDIUM_TWIN_JET, REGIONAL_TURBOPROP
from atp.aircraft.turn import turn_radius_nm, turn_rate_rad_per_h
from atp.core.units import G_NM_PER_H2

ALTITUDES = [28000.0, 30000.0, 33000.0, 37000.0]
SPEEDS = [330.0, 360.0, 390.0, 420.0, 450.0, 480.0]


# -- reduction to Milestone 2 ------------------------------------------------
def test_the_speed_factor_is_exactly_one_at_the_reference_speed():
    """Not 'approximately one'.  Fixed-speed parity is bit-exact and rests on
    this multiplication being the identity."""
    for aircraft in (MEDIUM_TWIN_JET, REGIONAL_TURBOPROP):
        assert aircraft.speed_fuel_factor(aircraft.reference_tas_kt) == 1.0


def test_fuel_flow_at_the_reference_speed_is_bit_identical_to_the_unspeeded_call():
    for altitude in ALTITUDES:
        unspeeded = MEDIUM_TWIN_JET.fuel_flow_kg_per_h(altitude)
        speeded = MEDIUM_TWIN_JET.fuel_flow_kg_per_h(
            altitude, MEDIUM_TWIN_JET.reference_tas_kt
        )
        assert speeded == unspeeded


def test_turn_geometry_at_the_default_speed_is_unchanged():
    for altitude in ALTITUDES:
        assert MEDIUM_TWIN_JET.turn_radius_nm(altitude) == turn_radius_nm(
            MEDIUM_TWIN_JET.tas_kt(altitude), MEDIUM_TWIN_JET.max_bank_deg
        )
        assert MEDIUM_TWIN_JET.turn_rate_rad_per_h(altitude) == turn_rate_rad_per_h(
            MEDIUM_TWIN_JET.tas_kt(altitude), MEDIUM_TWIN_JET.max_bank_deg
        )


def test_the_reference_speed_does_not_move_when_an_envelope_is_attached():
    """A speed sweep substitutes envelopes.  If the fuel law re-normalised each
    time, every arm of the sweep would be a different aircraft and the sweep
    would compare nothing."""
    slow = replace(MEDIUM_TWIN_JET, speed_envelope=SpeedEnvelope.fixed(330.0))
    multi = replace(
        MEDIUM_TWIN_JET, speed_envelope=SpeedEnvelope((330.0, 450.0), 450.0)
    )
    assert slow.reference_tas_kt == multi.reference_tas_kt == 450.0
    assert slow.fuel_flow_kg_per_h(30000.0, 330.0) == multi.fuel_flow_kg_per_h(
        30000.0, 330.0
    )
    # The *default commanded* speed does follow the envelope, though.
    assert slow.tas_kt(30000.0) == 330.0
    assert multi.tas_kt(30000.0) == 450.0


# -- fuel flow versus speed --------------------------------------------------
def test_fuel_flow_follows_the_documented_power_law():
    b = MEDIUM_TWIN_JET.induced_power_fraction
    for tas in SPEEDS:
        x = tas / MEDIUM_TWIN_JET.reference_tas_kt
        expected = (1.0 - b) * x**3 + b / x
        assert MEDIUM_TWIN_JET.speed_fuel_factor(tas) == pytest.approx(
            expected, rel=1e-12
        )


def test_fuel_flow_actually_varies_with_speed():
    """The milestone's minimum requirement: speed must move fuel flow."""
    flows = [MEDIUM_TWIN_JET.fuel_flow_kg_per_h(30000.0, v) for v in SPEEDS]
    assert len(set(flows)) == len(flows)
    assert max(flows) / min(flows) > 1.5


def test_the_fuel_flow_curve_has_an_interior_minimum_where_the_derivation_says():
    """``d/dx [(1-b)x^3 + b/x] = 0`` at ``x = (b / (3(1-b)))^(1/4)``.  The
    minimum is interior, so flying slower is not unconditionally cheaper -- the
    induced term takes over.  A monotone curve would make the speed decision a
    one-way knob rather than a trade-off."""
    b = MEDIUM_TWIN_JET.induced_power_fraction
    x_star = (b / (3.0 * (1.0 - b))) ** 0.25
    assert 0.0 < x_star < 1.0
    factor = MEDIUM_TWIN_JET.speed_fuel_factor
    reference = MEDIUM_TWIN_JET.reference_tas_kt
    best = factor(x_star * reference)
    for delta in (-0.08, -0.03, 0.03, 0.08):
        assert factor((x_star + delta) * reference) > best


def test_best_specific_range_sits_at_a_different_speed_than_least_fuel_flow():
    """Specific range is ``V / ff``; maximising ``x / factor(x)`` gives
    ``x = (b / (1-b))^(1/4)`` -- the same shape as the fuel-flow minimum but
    *without* its factor of 3 in the denominator, so in general the two optima
    are not related by that factor alone. For the shipped ``b = 0.25`` this
    happens to reduce to ``3^(-1/4)``, which is why that specific numeral
    appears elsewhere in this codebase's documentation, but the formula
    asserted here is the general one, not the coincidence."""
    b = MEDIUM_TWIN_JET.induced_power_fraction
    reference = MEDIUM_TWIN_JET.reference_tas_kt
    least_flow_x = (b / (3.0 * (1.0 - b))) ** 0.25
    best_range_x = (b / (1.0 - b)) ** 0.25
    assert best_range_x == pytest.approx(3.0**-0.25, rel=1e-9), (
        "sanity check: for the shipped b=0.25 the general formula must "
        "coincide with the numeral quoted in the documentation"
    )
    assert best_range_x != pytest.approx(least_flow_x, rel=1e-3)

    def specific_range(x: float) -> float:
        return x / MEDIUM_TWIN_JET.speed_fuel_factor(x * reference)

    best = specific_range(best_range_x)
    for delta in (-0.08, -0.03, 0.03, 0.08):
        assert specific_range(best_range_x + delta) < best


def test_the_specific_range_formula_is_not_just_the_b_equals_quarter_coincidence():
    """At ``b = 0.25`` the correct formula ``(b/(1-b))^(1/4)`` and the
    superficially similar ``3^(-1/4)`` happen to coincide, which is exactly
    why a documentation error here could survive unnoticed. This checks a
    *different* induced fraction, where the two numerically diverge, so the
    formula under test is pinned rather than merely consistent with one lucky
    value of ``b``."""
    b = 0.4  # deliberately not 0.25
    aircraft = replace(MEDIUM_TWIN_JET, induced_power_fraction=b)
    reference = aircraft.reference_tas_kt
    correct_x = (b / (1.0 - b)) ** 0.25
    coincidental_x = 3.0**-0.25
    assert correct_x != pytest.approx(coincidental_x, rel=1e-3), (
        "the test needs b=0.4 to actually distinguish the two formulas"
    )

    def specific_range(x: float) -> float:
        return x / aircraft.speed_fuel_factor(x * reference)

    best = specific_range(correct_x)
    worse = specific_range(coincidental_x)
    assert best > worse, (
        "the true optimum (b/(1-b))^(1/4) must beat the superficially "
        "similar but wrong 3^(-1/4) once they numerically diverge"
    )


def test_the_altitude_trend_is_unchanged_and_separable():
    """Milestone 3 multiplies the inherited altitude trend by a speed factor; it
    does not reshape it.  The ratio between two altitudes must therefore be the
    same at every speed."""
    for tas in SPEEDS:
        low = MEDIUM_TWIN_JET.fuel_flow_kg_per_h(28000.0, tas)
        high = MEDIUM_TWIN_JET.fuel_flow_kg_per_h(37000.0, tas)
        baseline = MEDIUM_TWIN_JET.fuel_flow_kg_per_h(
            28000.0
        ) / MEDIUM_TWIN_JET.fuel_flow_kg_per_h(37000.0)
        assert low / high == pytest.approx(baseline, rel=1e-12)


def test_fuel_flow_is_never_negative_and_respects_the_floor():
    floored = replace(MEDIUM_TWIN_JET, min_fuel_flow_kg_per_h=2000.0)
    for tas in SPEEDS:
        assert floored.fuel_flow_kg_per_h(41000.0, tas) >= 2000.0
        assert MEDIUM_TWIN_JET.fuel_flow_kg_per_h(41000.0, tas) >= 0.0


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_non_positive_speed_is_refused_by_the_fuel_model(bad):
    with pytest.raises(ValueError):
        MEDIUM_TWIN_JET.speed_fuel_factor(bad)


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_an_out_of_range_induced_fraction_is_refused(bad):
    with pytest.raises(ValueError):
        replace(MEDIUM_TWIN_JET, induced_power_fraction=bad)


def test_a_zero_induced_fraction_gives_a_monotone_cube_law():
    pure = replace(MEDIUM_TWIN_JET, induced_power_fraction=0.0)
    flows = [pure.fuel_flow_kg_per_h(30000.0, v) for v in SPEEDS]
    for lower, upper in zip(flows, flows[1:]):
        assert upper > lower


# -- turn geometry versus speed ----------------------------------------------
def test_turn_radius_grows_with_the_square_of_speed():
    bank = MEDIUM_TWIN_JET.max_bank_deg
    for tas in SPEEDS:
        radius = MEDIUM_TWIN_JET.turn_radius_nm(30000.0, tas)
        assert radius == pytest.approx(
            tas * tas / (G_NM_PER_H2 * math.tan(math.radians(bank))), rel=1e-12
        )
    ratio = MEDIUM_TWIN_JET.turn_radius_nm(
        30000.0, 480.0
    ) / MEDIUM_TWIN_JET.turn_radius_nm(30000.0, 330.0)
    assert ratio == pytest.approx((480.0 / 330.0) ** 2, rel=1e-12)


def test_turn_rate_falls_inversely_with_speed():
    rates = [MEDIUM_TWIN_JET.turn_rate_rad_per_h(30000.0, v) for v in SPEEDS]
    for lower, upper in zip(rates, rates[1:]):
        assert upper < lower
    assert MEDIUM_TWIN_JET.turn_rate_rad_per_h(
        30000.0, 330.0
    ) * 330.0 == pytest.approx(
        MEDIUM_TWIN_JET.turn_rate_rad_per_h(30000.0, 480.0) * 480.0, rel=1e-12
    )


def test_the_product_of_rate_and_radius_is_the_speed_at_every_speed():
    """``omega R = V`` is the defining consistency of the two relations."""
    for tas in SPEEDS:
        omega = MEDIUM_TWIN_JET.turn_rate_rad_per_h(30000.0, tas)
        radius = MEDIUM_TWIN_JET.turn_radius_nm(30000.0, tas)
        assert omega * radius == pytest.approx(tas, rel=1e-12)


def test_a_slower_speed_takes_less_time_to_turn_through_the_same_angle():
    angle = math.radians(45.0)
    times = [
        angle / MEDIUM_TWIN_JET.turn_rate_rad_per_h(30000.0, v) for v in SPEEDS
    ]
    for lower, upper in zip(times, times[1:]):
        assert upper > lower
