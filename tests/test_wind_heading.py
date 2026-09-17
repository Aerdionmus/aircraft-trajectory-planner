"""T-17 to T-21: air heading recovered from the wind triangle.

Milestone 1 solved for ground speed and threw the heading away.  Milestone 2
needs it, because a bank limit constrains the change in *air heading*, not the
change in ground track.
"""

from __future__ import annotations

import math
import random

import pytest

from atp.aircraft.kinematics import solve_ground_speed
from atp.core.geometry import Vec2, angle_between, signed_angle_between

EAST = Vec2(1.0, 0.0)
NORTH = Vec2(0.0, 1.0)


def _unit(angle_rad: float) -> Vec2:
    return Vec2(math.cos(angle_rad), math.sin(angle_rad))


def test_air_heading_is_a_unit_vector_over_a_wide_random_sweep():
    """T-17.  |psi_hat| = 1 is exact algebra, not a numerical accident:
    |GS t - w|^2 = (GS - a)^2 + c^2 = TAS^2."""
    rng = random.Random(20260917)
    checked = 0
    worst = 0.0
    for _ in range(4000):
        tas = rng.uniform(80.0, 520.0)
        wind_magnitude = rng.uniform(0.0, 700.0)  # deliberately includes |w| > TAS
        track = _unit(rng.uniform(0.0, 2 * math.pi))
        wind = _unit(rng.uniform(0.0, 2 * math.pi)) * wind_magnitude
        solution = solve_ground_speed(tas, wind, track)
        if not solution.feasible:
            assert solution.air_heading_unit.norm() == 0.0
            continue
        checked += 1
        worst = max(worst, abs(solution.air_heading_unit.norm() - 1.0))
        # The triangle itself must close: TAS * psi_hat + w == GS * t_hat.
        closed = solution.air_heading_unit * tas + wind
        assert closed.x == pytest.approx(track.x * solution.ground_speed_kt, abs=1e-9)
        assert closed.y == pytest.approx(track.y * solution.ground_speed_kt, abs=1e-9)
    assert checked > 1000
    assert worst < 1e-12


def test_air_heading_at_the_crosswind_limit():
    """T-17 edge case: c -> TAS, where the aircraft points straight into the
    crosswind and the ground speed collapses."""
    tas = 200.0
    solution = solve_ground_speed(tas, Vec2(0.0, tas - 1e-6), EAST)
    assert solution.feasible
    assert solution.air_heading_unit.norm() == pytest.approx(1.0, abs=1e-9)
    assert solution.air_heading_unit.y == pytest.approx(-1.0, abs=1e-3)


def test_zero_wind_makes_heading_equal_track_exactly():
    """T-18."""
    for degrees in range(0, 360, 15):
        track = _unit(math.radians(degrees))
        solution = solve_ground_speed(450.0, Vec2(0.0, 0.0), track)
        assert solution.feasible
        assert solution.air_heading_unit.x == pytest.approx(track.x, abs=1e-12)
        assert solution.air_heading_unit.y == pytest.approx(track.y, abs=1e-12)
        assert solution.drift_angle_deg == pytest.approx(0.0, abs=1e-12)


def test_pure_tail_and_headwind_leave_the_heading_on_track():
    solution = solve_ground_speed(450.0, EAST * 50.0, EAST)
    assert solution.ground_speed_kt == pytest.approx(500.0)
    assert solution.drift_angle_deg == pytest.approx(0.0, abs=1e-9)
    assert solution.air_heading_unit.x == pytest.approx(1.0)


def test_drift_is_signed_and_opposite_for_mirrored_crosswinds():
    """T-19.  Milestone 1 reported ``asin(c / TAS)``, which is never negative and
    therefore cannot tell a left crab from a right one."""
    left = solve_ground_speed(450.0, NORTH * 90.0, EAST)
    right = solve_ground_speed(450.0, NORTH * -90.0, EAST)
    assert left.feasible and right.feasible
    assert left.drift_angle_deg == pytest.approx(-right.drift_angle_deg)
    assert left.drift_angle_deg > 0.0 > right.drift_angle_deg
    # Magnitude is unchanged from Milestone 1.
    assert abs(left.drift_angle_deg) == pytest.approx(math.degrees(math.asin(90 / 450)))
    # A wind blowing north pushes the aircraft left of an easterly track, so it
    # must point south of east to hold that track.
    assert left.air_heading_unit.y < 0.0
    assert right.air_heading_unit.y > 0.0


def test_track_equals_heading_plus_drift():
    """T-19: the sign convention is the aviation one."""
    rng = random.Random(5)
    for _ in range(200):
        tas = rng.uniform(200.0, 500.0)
        track = _unit(rng.uniform(0.0, 2 * math.pi))
        wind = _unit(rng.uniform(0.0, 2 * math.pi)) * rng.uniform(0.0, 150.0)
        solution = solve_ground_speed(tas, wind, track)
        if not solution.feasible:
            continue
        drift = signed_angle_between(solution.air_heading_unit, track)
        assert math.degrees(drift) == pytest.approx(solution.drift_angle_deg, abs=1e-9)


def test_heading_change_equals_track_change_only_in_still_air():
    """T-20."""
    tracks = [_unit(math.radians(d)) for d in (0.0, 45.0, 90.0, 135.0)]
    still = Vec2(0.0, 0.0)
    for a, b in zip(tracks, tracks[1:]):
        ha = solve_ground_speed(450.0, still, a).air_heading_unit
        hb = solve_ground_speed(450.0, still, b).air_heading_unit
        assert angle_between(ha, hb) == pytest.approx(angle_between(a, b), abs=1e-12)

    wind = Vec2(0.0, 120.0)
    differed = False
    for a, b in zip(tracks, tracks[1:]):
        sa, sb = solve_ground_speed(450.0, wind, a), solve_ground_speed(450.0, wind, b)
        assert sa.feasible and sb.feasible
        heading_change = angle_between(sa.air_heading_unit, sb.air_heading_unit)
        if abs(heading_change - angle_between(a, b)) > 1e-6:
            differed = True
    assert differed, "in wind, the heading change must not equal the track change"


def test_unflyable_tracks_are_reported_not_clamped():
    """T-21."""
    too_much_crosswind = solve_ground_speed(200.0, Vec2(0.0, 260.0), EAST)
    assert not too_much_crosswind.feasible
    assert "crosswind" in too_much_crosswind.reason
    assert too_much_crosswind.air_heading_unit.norm() == 0.0

    blown_backwards = solve_ground_speed(200.0, EAST * -260.0, EAST)
    assert not blown_backwards.feasible
    assert blown_backwards.air_heading_unit.norm() == 0.0
