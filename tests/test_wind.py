from __future__ import annotations

import math
import random

import pytest

from atp.aircraft.kinematics import max_possible_ground_speed_kt, solve_ground_speed
from atp.core.geometry import Vec2
from atp.environment.wind import (
    CompositeWind,
    LayeredWind,
    UniformWind,
    VortexWind,
    ZeroWind,
)

FIELDS = [
    ZeroWind(),
    UniformWind(Vec2(40.0, -20.0)),
    LayeredWind(
        layers=((31000.0, Vec2(-25.0, 0.0)), (33000.0, Vec2(90.0, 10.0))),
        above=Vec2(70.0, 0.0),
    ),
    VortexWind(Vec2(150.0, 150.0), peak_speed_kt=110.0, core_radius_nm=60.0),
    CompositeWind(
        fields=(
            UniformWind(Vec2(10.0, 0.0)),
            VortexWind(Vec2(80.0, 80.0), 60.0, 30.0, clockwise=True),
        )
    ),
]


@pytest.mark.parametrize("field", FIELDS, ids=[f.name for f in FIELDS])
def test_max_magnitude_is_a_sound_upper_bound(field):
    """The admissible heuristic depends on this bound; sample the field densely
    and assert nothing exceeds the advertised maximum."""
    rng = random.Random(12345)
    bound = field.max_magnitude_kt()
    for _ in range(3000):
        x = rng.uniform(0.0, 300.0)
        y = rng.uniform(0.0, 300.0)
        alt = rng.uniform(20000.0, 40000.0)
        assert field.at(x, y, alt).norm() <= bound + 1e-9


def test_layered_wind_selects_band():
    field = LayeredWind(
        layers=((31000.0, Vec2(-25.0, 0.0)), (33000.0, Vec2(90.0, 0.0))),
        above=Vec2(70.0, 0.0),
    )
    assert field.at(0, 0, 28000.0).x == -25.0
    assert field.at(0, 0, 32000.0).x == 90.0
    assert field.at(0, 0, 36000.0).x == 70.0


def test_vortex_is_tangential_and_peaks_at_core_radius():
    field = VortexWind(Vec2(0.0, 0.0), peak_speed_kt=100.0, core_radius_nm=10.0)
    at_core = field.at(10.0, 0.0, 30000.0)
    assert at_core.norm() == pytest.approx(100.0)
    # Tangential: perpendicular to the radius vector.
    assert at_core.dot(Vec2(1.0, 0.0)) == pytest.approx(0.0, abs=1e-9)
    assert field.at(20.0, 0.0, 30000.0).norm() == pytest.approx(50.0)
    assert field.at(5.0, 0.0, 30000.0).norm() == pytest.approx(50.0)


def test_wind_triangle_pure_tailwind_and_headwind():
    east = Vec2(1.0, 0.0)
    tail = solve_ground_speed(450.0, Vec2(50.0, 0.0), east)
    head = solve_ground_speed(450.0, Vec2(-50.0, 0.0), east)
    assert tail.feasible and tail.ground_speed_kt == pytest.approx(500.0)
    assert head.feasible and head.ground_speed_kt == pytest.approx(400.0)
    assert tail.drift_angle_deg == pytest.approx(0.0, abs=1e-9)


def test_wind_triangle_pure_crosswind_reduces_ground_speed():
    solution = solve_ground_speed(450.0, Vec2(0.0, 90.0), Vec2(1.0, 0.0))
    assert solution.feasible
    assert solution.ground_speed_kt == pytest.approx(math.sqrt(450**2 - 90**2))
    assert solution.drift_angle_deg == pytest.approx(
        math.degrees(math.asin(90 / 450))
    )


def test_wind_triangle_rejects_unflyable_track():
    too_much_crosswind = solve_ground_speed(200.0, Vec2(0.0, 260.0), Vec2(1.0, 0.0))
    assert not too_much_crosswind.feasible
    assert "crosswind" in too_much_crosswind.reason

    blown_backwards = solve_ground_speed(200.0, Vec2(-260.0, 0.0), Vec2(1.0, 0.0))
    assert not blown_backwards.feasible


def test_ground_speed_bound_dominates_any_track():
    rng = random.Random(7)
    tas, wind_mag = 450.0, 110.0
    bound = max_possible_ground_speed_kt(tas, wind_mag)
    for _ in range(2000):
        angle = rng.uniform(0, 2 * math.pi)
        wind_angle = rng.uniform(0, 2 * math.pi)
        track = Vec2(math.cos(angle), math.sin(angle))
        wind = Vec2(
            wind_mag * math.cos(wind_angle), wind_mag * math.sin(wind_angle)
        )
        solution = solve_ground_speed(tas, wind, track)
        if solution.feasible:
            assert solution.ground_speed_kt <= bound + 1e-9
