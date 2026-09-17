from __future__ import annotations

import math

import pytest

from atp.core.geometry import (
    Vec2,
    bearing_deg,
    met_wind_to_vector,
    point_in_polygon,
    point_segment_distance,
    segment_circle_intersects,
    segments_intersect,
)
from atp.core.units import ft_to_nm, flight_level_to_ft, nm_to_ft


def test_vector_algebra():
    a, b = Vec2(3.0, 4.0), Vec2(1.0, -1.0)
    assert a.norm() == pytest.approx(5.0)
    assert (a + b).as_tuple() == (4.0, 3.0)
    assert a.dot(b) == pytest.approx(-1.0)
    assert a.normalized().norm() == pytest.approx(1.0)
    assert Vec2(0.0, 0.0).normalized().norm() == 0.0


def test_bearing_convention_is_north_zero_clockwise():
    origin = Vec2(0.0, 0.0)
    assert bearing_deg(origin, Vec2(0.0, 1.0)) == pytest.approx(0.0)
    assert bearing_deg(origin, Vec2(1.0, 0.0)) == pytest.approx(90.0)
    assert bearing_deg(origin, Vec2(0.0, -1.0)) == pytest.approx(180.0)


def test_met_wind_conversion_points_downwind():
    # A wind *from* 270 (a westerly) pushes the air toward the east (+x).
    wind = met_wind_to_vector(270.0, 50.0)
    assert wind.x == pytest.approx(50.0, abs=1e-9)
    assert wind.y == pytest.approx(0.0, abs=1e-9)
    # A wind from 180 (a southerly) pushes toward the north (+y).
    wind = met_wind_to_vector(180.0, 30.0)
    assert wind.y == pytest.approx(30.0, abs=1e-9)


def test_point_segment_distance_clamps_to_endpoints():
    a, b = Vec2(0.0, 0.0), Vec2(10.0, 0.0)
    assert point_segment_distance(Vec2(5.0, 3.0), a, b) == pytest.approx(3.0)
    assert point_segment_distance(Vec2(-4.0, 0.0), a, b) == pytest.approx(4.0)
    assert point_segment_distance(Vec2(14.0, 0.0), a, b) == pytest.approx(4.0)


def test_segment_intersection_cases():
    assert segments_intersect(
        Vec2(0, 0), Vec2(10, 10), Vec2(0, 10), Vec2(10, 0)
    )
    assert not segments_intersect(
        Vec2(0, 0), Vec2(1, 1), Vec2(5, 5), Vec2(6, 6)
    )
    # Touching endpoints count as intersecting (conservative for no-fly zones).
    assert segments_intersect(Vec2(0, 0), Vec2(5, 0), Vec2(5, 0), Vec2(5, 5))


def test_point_in_polygon_including_boundary():
    square = (Vec2(0, 0), Vec2(10, 0), Vec2(10, 10), Vec2(0, 10))
    assert point_in_polygon(Vec2(5, 5), square)
    assert not point_in_polygon(Vec2(15, 5), square)
    assert point_in_polygon(Vec2(0, 5), square)


def test_point_in_polygon_non_convex():
    l_shape = (
        Vec2(0, 0),
        Vec2(10, 0),
        Vec2(10, 4),
        Vec2(4, 4),
        Vec2(4, 10),
        Vec2(0, 10),
    )
    assert point_in_polygon(Vec2(2, 8), l_shape)
    assert not point_in_polygon(Vec2(8, 8), l_shape)


def test_segment_circle_intersection():
    assert segment_circle_intersects(Vec2(0, 0), Vec2(10, 0), Vec2(5, 2), 3.0)
    assert not segment_circle_intersects(Vec2(0, 0), Vec2(10, 0), Vec2(5, 9), 3.0)


def test_unit_round_trips():
    assert ft_to_nm(nm_to_ft(3.0)) == pytest.approx(3.0)
    assert flight_level_to_ft(350) == pytest.approx(35000.0)
    assert math.isclose(nm_to_ft(1.0), 6076.115485564304)
