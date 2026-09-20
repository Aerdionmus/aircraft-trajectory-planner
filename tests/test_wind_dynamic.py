from __future__ import annotations

import math

import pytest

from atp.core.geometry import Vec2
from atp.environment.wind import PeriodicWind, UniformWind, ZeroWind


def test_periodic_wind_matches_analytic_point_and_time():
    field = PeriodicWind(
        centre_nm=Vec2(50.0, 50.0),
        base_vector_kt=Vec2(20.0, -10.0),
        amplitude_kt=30.0,
        period_h=4.0,
        altitude_scale_ft=20000.0,
        phase_offset=0.0,
    )

    x = 60.0
    y = 50.0
    alt = 30000.0
    t = 1.0
    offset = Vec2(x - 50.0, y - 50.0)
    radial = offset.normalized()
    phase = (
        2.0 * math.pi * (t / 4.0)
        + offset.norm() / max(1.0, 20000.0 / 1000.0)
        + alt / 20000.0
    )
    expected = Vec2(20.0, -10.0) + radial * (math.sin(phase) * 30.0)
    assert field.at_time(x, y, alt, t) == pytest.approx(expected)


def test_periodic_wind_changes_with_time():
    field = PeriodicWind(
        centre_nm=Vec2(0.0, 0.0),
        base_vector_kt=Vec2(0.0, 0.0),
        amplitude_kt=50.0,
        period_h=2.0,
        altitude_scale_ft=10000.0,
    )

    t0 = field.at_time(10.0, 0.0, 10000.0, 0.0)
    t_half = field.at_time(10.0, 0.0, 10000.0, 1.0)
    assert t0 != pytest.approx(t_half)
    assert t0 == pytest.approx(Vec2(-t_half.x, -t_half.y))
    assert t0.x != pytest.approx(t_half.x)
    assert t0.y == pytest.approx(t_half.y)


def test_periodic_wind_changes_with_altitude():
    field = PeriodicWind(
        centre_nm=Vec2(0.0, 0.0),
        base_vector_kt=Vec2(0.0, 0.0),
        amplitude_kt=45.0,
        period_h=1.0,
        altitude_scale_ft=10000.0,
    )

    low = field.at_time(10.0, 0.0, 0.0, 0.5)
    high = field.at_time(10.0, 0.0, 50000.0, 0.5)
    assert low != pytest.approx(high)


def test_periodic_wind_respects_declared_bound():
    field = PeriodicWind(
        centre_nm=Vec2(0.0, 0.0),
        base_vector_kt=Vec2(25.0, -5.0),
        amplitude_kt=40.0,
        period_h=3.0,
        altitude_scale_ft=15000.0,
    )
    bound = field.max_magnitude_kt()
    for x in (-100.0, -10.0, 0.0, 10.0, 100.0):
        for y in (-100.0, 0.0, 100.0):
            for alt in (0.0, 15000.0, 50000.0):
                for t in (0.0, 0.5, 1.5, 3.0):
                    assert field.at_time(x, y, alt, t).norm() <= bound + 1e-9


def test_dynamic_wind_has_no_static_snapshot_api():
    field = PeriodicWind(
        centre_nm=Vec2(5.0, -2.0),
        base_vector_kt=Vec2(12.0, 6.0),
        amplitude_kt=18.0,
        period_h=2.0,
        altitude_scale_ft=20000.0,
    )
    assert not hasattr(field, "at")


def test_static_wind_implementations_are_unchanged():
    zero = ZeroWind().at(0.0, 0.0, 30000.0)
    uniform = UniformWind(Vec2(30.0, -20.0)).at(10.0, 20.0, 10000.0)
    assert zero == pytest.approx(Vec2(0.0, 0.0))
    assert uniform == pytest.approx(Vec2(30.0, -20.0))
