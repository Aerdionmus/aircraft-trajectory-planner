"""Regression tests for the ground-path turn-radius bound.

The previous implementation used ``radius = max(GS_in, GS_out) / omega`` as the
radius for the fly-by tangent-fit gate, justified by a docstring claim that "the
ground-frame turn rate equals the air-frame turn rate" in uniform wind. That
claim is false, and the formula it justified was not conservative: it could
under-estimate the true ground-path radius of curvature reached during a turn,
which is the wrong direction for a gate documented elsewhere as over-blocking
rather than under-blocking (see ``docs/turn_model.md`` section 3 for the full
derivation and correction).

These tests check the *fix*, ``ground_curvature_radius_bound_nm``, against:

- an independent, from-scratch finite-difference computation of the true
  instantaneous curvature (not the closed form the fix itself uses -- the point
  is to check the fix against raw kinematics, not against its own derivation);
- the exact still-air case, where the fix must reproduce the original formula
  bit for bit (a genuine regression check: this is the only turn-active library
  scenario's wind condition, so no existing test would have caught a regression
  here without this file);
- a constructed wind/turn configuration where the *old* formula would have
  accepted a corner that the true physics cannot fit, demonstrating the bug was
  live and is now closed;
- the declared precondition (wind at or above TAS), where the function must
  return an unbounded radius rather than an unsound finite one;
- the monotone-refinement invariant, since the fix can only make the gate more
  conservative, never less.
"""

from __future__ import annotations

import math

import pytest

from atp.aircraft.kinematics import solve_ground_speed
from atp.aircraft.performance import MEDIUM_TWIN_JET, REGIONAL_TURBOPROP
from atp.aircraft.turn import (
    ground_curvature_radius_bound_nm,
    tangent_length_nm,
    turn_radius_nm,
    turn_rate_rad_per_h,
)
from atp.core.geometry import Vec2
from atp.environment.airspace import GridState
from atp.environment.wind import UniformWind
from conftest import make_problem


# -- independent finite-difference check -------------------------------------
def _exact_instantaneous_radius(tas_kt, wind_vec, heading_rad, omega, h=1e-6):
    """R = |v|^3 / |v x a| for v(t) = V*psi_hat(t) + w, computed by central
    finite difference -- deliberately *not* the closed form the fix itself
    derives, so this is an independent check against raw kinematics.

    ``v`` below is parametrised by *heading* psi, not time, so a central
    difference over psi gives dv/dpsi; the chain rule ``dpsi/dt = omega``
    (the defining property of a constant-rate turn) is applied explicitly to
    recover the true acceleration ``a = dv/dt = omega * dv/dpsi``. Omitting
    that factor is a distinct bug from the one under test here and was caught
    by this file's own extra alignment-point check before being fixed.
    """

    def psi_hat(psi):
        return Vec2(math.sin(psi), math.cos(psi))

    def v(psi):
        u = psi_hat(psi)
        return Vec2(tas_kt * u.x + wind_vec.x, tas_kt * u.y + wind_vec.y)

    v_minus = v(heading_rad - h)
    v_plus = v(heading_rad + h)
    v0 = v(heading_rad)
    dv_dpsi = Vec2((v_plus.x - v_minus.x) / (2 * h), (v_plus.y - v_minus.y) / (2 * h))
    a = Vec2(dv_dpsi.x * omega, dv_dpsi.y * omega)
    speed = v0.norm()
    cross = v0.cross(a)
    if abs(cross) < 1e-9:
        return None  # locally straight; not the tightest point of the turn
    return speed**3 / abs(cross)


@pytest.mark.parametrize("tas", [280.0, 450.0])
@pytest.mark.parametrize("wind_mag", [0.0, 30.0, 100.0, 200.0])
@pytest.mark.parametrize("bank", [15.0, 25.0, 35.0])
def test_bound_dominates_the_true_instantaneous_radius_everywhere(
    tas, wind_mag, bank
):
    """The bound must never be exceeded by the true curvature radius at *any*
    heading the aircraft could point during a turn -- not just at the two leg
    endpoints, which is the specific failure mode being fixed."""
    if wind_mag >= tas:
        pytest.skip("outside the bound's declared precondition")
    omega = turn_rate_rad_per_h(tas, bank)
    bound = ground_curvature_radius_bound_nm(tas, wind_mag, bank)
    assert math.isfinite(bound)

    worst_ratio = 0.0
    for wind_deg in range(0, 360, 10):
        wind = Vec2(
            wind_mag * math.sin(math.radians(wind_deg)),
            wind_mag * math.cos(math.radians(wind_deg)),
        )
        for heading_deg in range(0, 360, 5):
            radius = _exact_instantaneous_radius(
                tas, wind, math.radians(heading_deg), omega
            )
            if radius is None:
                continue
            assert radius <= bound * 1.001, (
                f"tas={tas} wind={wind_mag}@{wind_deg} bank={bank} "
                f"heading={heading_deg}: true radius {radius:.3f} exceeds "
                f"bound {bound:.3f}"
            )
            worst_ratio = max(worst_ratio, radius / bound)

    # And the bound should not be absurdly loose: the true worst case (heading
    # aligned with the wind) must come within numerical tolerance of it, or the
    # closed form and the finite-difference check disagree about where the
    # maximum is.
    if wind_mag > 0.0:
        assert worst_ratio > 0.999


def test_the_bound_is_attained_exactly_when_heading_aligns_with_the_wind():
    """Pins the closed-form maximum, ``(V + |w|)^2 / (V omega)``, against the
    independent finite-difference computation at the specific heading the
    derivation identifies as the worst case."""
    tas, wind_mag, bank = 450.0, 100.0, 25.0
    omega = turn_rate_rad_per_h(tas, bank)
    bound = ground_curvature_radius_bound_nm(tas, wind_mag, bank)
    # Wind blowing due north; heading due north aligns with it (x = cos 0 = 1).
    wind = Vec2(0.0, wind_mag)
    at_alignment = _exact_instantaneous_radius(tas, wind, 0.0, omega)
    assert at_alignment == pytest.approx(bound, rel=1e-4)


# -- zero-wind regression: no change to the only turn-active scenario --------
def test_zero_wind_reproduces_the_original_still_air_radius_exactly():
    """The fix must be a no-op in still air: this is the only wind condition
    the ``turn-limited`` library scenario exercises, so it is what protects
    Milestone 2's one turn-active scenario from an unintended behaviour change."""
    for tas, bank in [(280.0, 25.0), (450.0, 15.0), (450.0, 25.0), (450.0, 35.0)]:
        assert ground_curvature_radius_bound_nm(tas, 0.0, bank) == pytest.approx(
            turn_radius_nm(tas, bank), rel=1e-12
        )


@pytest.mark.parametrize("aircraft", [MEDIUM_TWIN_JET, REGIONAL_TURBOPROP])
def test_shipped_aircraft_are_unaffected_by_the_fix_in_still_air(aircraft):
    alt = 30000.0
    bound = ground_curvature_radius_bound_nm(
        aircraft.tas_kt(alt), 0.0, aircraft.max_bank_deg
    )
    assert bound == pytest.approx(aircraft.turn_radius_nm(alt), rel=1e-12)


# -- the bug, demonstrated concretely -----------------------------------------
def test_old_endpoint_formula_would_have_under_blocked_a_tailwind_turn():
    """Reconstructs the retired ``max(GS_in, GS_out) / omega`` formula directly
    and shows it under-estimates the sound bound for a turn with a tailwind
    component -- i.e. it would have accepted corners the physics cannot
    support. This is the failure mode the fix closes."""
    tas, bank = 450.0, 25.0
    wind_mag = 100.0
    omega = turn_rate_rad_per_h(tas, bank)

    # A 90-degree turn: incoming track due north, outgoing due east, with the
    # wind blowing towards the north-east -- roughly bisecting the turn, so it
    # has a tailwind component on both legs without being extreme on either.
    wind = Vec2(
        wind_mag * math.sin(math.radians(45.0)), wind_mag * math.cos(math.radians(45.0))
    )
    incoming = solve_ground_speed(tas, wind, Vec2(0.0, 1.0))
    outgoing = solve_ground_speed(tas, wind, Vec2(1.0, 0.0))
    assert incoming.feasible and outgoing.feasible

    old_formula_radius = max(incoming.ground_speed_kt, outgoing.ground_speed_kt) / omega
    sound_bound = ground_curvature_radius_bound_nm(tas, wind_mag, bank)
    assert sound_bound > old_formula_radius * 1.05, (
        "the wind/heading geometry in this test no longer demonstrates the gap; "
        "it must be re-chosen to keep exercising the fixed defect"
    )

    dpsi = math.radians(90.0)
    old_tangent = tangent_length_nm(old_formula_radius, dpsi)
    new_tangent = tangent_length_nm(sound_bound, dpsi)
    assert new_tangent > old_tangent

    # Pick a leg length between the two tangent requirements: the retired
    # formula would have called this turn legal; the sound bound correctly
    # calls it illegal.
    half_leg = (old_tangent + new_tangent) / 2.0
    assert old_tangent <= half_leg
    assert new_tangent > half_leg


def test_the_planner_now_rejects_a_turn_it_previously_would_have_accepted():
    """End-to-end version of the previous test, through ``turn_metrics`` and a
    real (if synthetic) grid, rather than the bare geometry functions."""
    tas, bank = 450.0, 25.0
    wind_mag = 100.0
    omega = turn_rate_rad_per_h(tas, bank)
    wind = Vec2(
        wind_mag * math.sin(math.radians(45.0)), wind_mag * math.cos(math.radians(45.0))
    )

    # Old formula's tangent requirement for a 90 degree turn, using the larger
    # endpoint ground speed -- reconstructed independently of the fixed code.
    incoming = solve_ground_speed(tas, wind, Vec2(0.0, 1.0))
    outgoing = solve_ground_speed(tas, wind, Vec2(1.0, 0.0))
    old_radius = max(incoming.ground_speed_kt, outgoing.ground_speed_kt) / omega
    old_tangent = tangent_length_nm(old_radius, math.radians(90.0))

    # A cell size whose half-leg sits just above the old (buggy) requirement --
    # the retired formula would have accepted this turn -- but below the sound
    # bound's requirement.
    cell_size = 2.0 * old_tangent / 0.5 * 1.02 / 2.0  # half-leg ~= old_tangent*1.02
    problem = make_problem(
        cells=9,
        cell_size_nm=cell_size,
        turn_model="gate+cost",
        max_bank_deg=bank,
        wind=UniformWind(wind),
    )
    table = problem.turn_table
    north = table.index_of((0, 1))
    east = table.index_of((1, 0))
    turn = problem.cost_model.turn_metrics(
        GridState(4, 4, 0),
        table.track_unit(north),
        table.track_unit(east),
        table.length_nm[north],
        table.length_nm[east],
    )
    assert not turn.feasible, (
        "the sound radius bound should reject this turn; if it now accepts it, "
        "either the fix regressed or this test's geometry needs re-deriving"
    )
    assert "NM of leg" in turn.infeasible_reason


# -- the declared precondition ------------------------------------------------
def test_wind_at_or_above_tas_returns_an_unbounded_radius():
    """Outside |w| <= V no sound finite bound is established; the function must
    say so explicitly (infinity) rather than silently extrapolate the formula,
    and the planner must turn that into a clean rejection, not a crash."""
    assert math.isinf(ground_curvature_radius_bound_nm(450.0, 450.0, 25.0))
    assert math.isinf(ground_curvature_radius_bound_nm(450.0, 900.0, 25.0))
    assert ground_curvature_radius_bound_nm(450.0, 449.999, 25.0) < math.inf


def test_precondition_violation_is_handled_cleanly_by_the_gate():
    """A wind whose magnitude exceeds TAS but that bisects a 90 degree turn
    passes each leg's individual crosswind check (crosswind on each leg is
    only ``|w| sin(45 deg)``) yet puts ``wind.norm()`` outside the radius
    bound's precondition; the corner must be rejected on the *tangent length*
    (the infinite-radius path), not crash and not be accepted on an unsound
    number."""
    tas = 450.0
    wind_mag = 1.1 * tas  # > TAS overall, but sin(45) * wind_mag < TAS
    assert wind_mag * math.sin(math.radians(45.0)) < tas  # premise of the setup
    wind_vec = Vec2(
        wind_mag * math.sin(math.radians(45.0)), wind_mag * math.cos(math.radians(45.0))
    )
    problem = make_problem(
        cells=9,
        cell_size_nm=200.0,
        turn_model="gate+cost",
        max_bank_deg=25.0,
        wind=UniformWind(wind_vec),
    )
    table = problem.turn_table
    north, east = table.index_of((0, 1)), table.index_of((1, 0))
    turn = problem.cost_model.turn_metrics(
        GridState(4, 4, 0),
        table.track_unit(north),
        table.track_unit(east),
        table.length_nm[north],
        table.length_nm[east],
    )
    assert not turn.feasible
    assert "NM of leg" in turn.infeasible_reason
    assert "needs inf NM" in turn.infeasible_reason  # {tangent:.2f} of math.inf


def test_negative_or_zero_inputs_are_rejected():
    with pytest.raises(ValueError):
        ground_curvature_radius_bound_nm(0.0, 0.0, 25.0)
    with pytest.raises(ValueError):
        ground_curvature_radius_bound_nm(-450.0, 0.0, 25.0)
    with pytest.raises(ValueError):
        ground_curvature_radius_bound_nm(450.0, -1.0, 25.0)


# -- monotone refinement: the fix can only make the gate stricter ------------
@pytest.mark.parametrize("wind_mag", [0.0, 30.0, 76.16, 130.0])
def test_the_fixed_bound_never_undercuts_the_retired_endpoint_formula(wind_mag):
    """The fix is a strictly more conservative gate: for any endpoint speeds a
    real corner could have, the sound bound is at least as large as the
    retired ``max(GS_in, GS_out) / omega`` value. Turns that were previously
    illegal remain illegal; turns that were previously legal may now become
    illegal, but never the reverse -- so no edge the fixed gate accepts is
    cheaper than what the retired gate would have charged for it, preserving
    the monotone-refinement invariant this milestone depends on."""
    tas, bank = 450.0, 25.0
    omega = turn_rate_rad_per_h(tas, bank)
    bound = ground_curvature_radius_bound_nm(tas, wind_mag, bank)
    for heading_deg in range(0, 360, 15):
        wind = Vec2(
            wind_mag * math.sin(math.radians(heading_deg)),
            wind_mag * math.cos(math.radians(heading_deg)),
        )
        for track_deg in (0.0, 45.0, 90.0, 135.0, 180.0):
            track = Vec2(
                math.sin(math.radians(track_deg)), math.cos(math.radians(track_deg))
            )
            solution = solve_ground_speed(tas, wind, track)
            if not solution.feasible:
                continue
            old_style_radius = solution.ground_speed_kt / omega
            assert bound >= old_style_radius - 1e-9


def test_no_library_scenario_cost_decreased_relative_to_its_pre_fix_value():
    """Direct regression guard on the one scenario that could possibly be
    affected: turn-limited's cost and feasibility must be unchanged, because it
    uses zero wind (where the fix is exact-identical to the retired formula),
    confirming the fix did not silently alter Milestone 2's shipped behaviour."""
    from atp.experiments.runner import run_plan
    from atp.scenarios.library import get_scenario
    from atp.scenarios.spec import build_scenario

    built = build_scenario(get_scenario("turn-limited"))
    report = run_plan(built, heuristic="optimistic")
    assert report.status == "solved"
    assert report.evaluation.comparable_cost == pytest.approx(6493.054358075926, rel=1e-9)
    assert report.evaluation.num_turns == 3
