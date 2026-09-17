from __future__ import annotations

import pytest

from atp.core.geometry import Vec2
from atp.environment.restrictions import (
    CircularRestriction,
    CorridorRestriction,
    PolygonRestriction,
    RestrictionSet,
)


def circle(**kwargs):
    defaults = dict(
        region_id="P-1", centre_nm=Vec2(50.0, 50.0), radius_nm=10.0
    )
    defaults.update(kwargs)
    return CircularRestriction(**defaults)


def test_circle_containment_and_segment_test():
    region = circle()
    assert region.contains_point(Vec2(55.0, 50.0))
    assert not region.contains_point(Vec2(70.0, 50.0))
    assert region.intersects_segment(Vec2(0, 50), Vec2(100, 50), 30000, 30000)
    assert not region.intersects_segment(Vec2(0, 90), Vec2(100, 90), 30000, 30000)


def test_altitude_band_is_respected():
    region = circle(lower_ft=30000.0, upper_ft=34000.0)
    crossing = (Vec2(0, 50), Vec2(100, 50))
    assert region.intersects_segment(*crossing, 31000, 31000)
    assert not region.intersects_segment(*crossing, 20000, 24000)
    # A climb that enters the band is caught.
    assert region.intersects_segment(*crossing, 29000, 31000)


def test_polygon_segment_intersection_without_vertex_inside():
    square = PolygonRestriction(
        region_id="P-2",
        vertices_nm=(Vec2(0, 0), Vec2(10, 0), Vec2(10, 10), Vec2(0, 10)),
    )
    # Segment passes straight through; neither endpoint is inside.
    assert square.intersects_segment_2d(Vec2(-5, 5), Vec2(15, 5))
    assert not square.intersects_segment_2d(Vec2(-5, 20), Vec2(15, 20))


def test_corridor_penalty_fraction():
    corridor = CorridorRestriction(
        region_id="C-1",
        start_nm=Vec2(0.0, 0.0),
        end_nm=Vec2(100.0, 0.0),
        half_width_nm=5.0,
        hard=False,
        penalty_per_nm=10.0,
    )
    # Fully inside.
    assert corridor.inside_fraction(
        Vec2(10, 0), Vec2(20, 0), 30000, 30000, samples=10
    ) == pytest.approx(1.0)
    # Fully outside.
    assert corridor.inside_fraction(
        Vec2(10, 50), Vec2(20, 50), 30000, 30000, samples=10
    ) == pytest.approx(0.0)
    # Half in, half out (crossing the edge at y = 5).
    fraction = corridor.inside_fraction(
        Vec2(10, 0), Vec2(10, 10), 30000, 30000, samples=100
    )
    assert fraction == pytest.approx(0.5, abs=0.02)


def test_restriction_set_separates_hard_and_soft():
    hard = circle(region_id="H")
    soft = CorridorRestriction(
        region_id="S",
        start_nm=Vec2(0, 200),
        end_nm=Vec2(200, 200),
        half_width_nm=5.0,
        hard=False,
        penalty_per_nm=20.0,
    )
    restrictions = RestrictionSet(regions=(hard, soft))

    assert restrictions.hard_regions == (hard,)
    assert restrictions.soft_regions == (soft,)

    blocking = restrictions.blocking_region(
        Vec2(0, 50), Vec2(100, 50), 30000, 30000
    )
    assert blocking is not None and blocking.region_id == "H"

    # Soft regions never block.
    assert (
        restrictions.blocking_region(Vec2(0, 200), Vec2(100, 200), 30000, 30000)
        is None
    )
    penalty = restrictions.soft_penalty(
        Vec2(0, 200), Vec2(100, 200), 30000, 30000, length_nm=100.0, samples=20
    )
    assert penalty == pytest.approx(100.0 * 20.0)


def test_point_blocked_only_inside_hard_regions():
    restrictions = RestrictionSet(
        regions=(circle(lower_ft=28000.0, upper_ft=32000.0),)
    )
    assert restrictions.point_blocked(Vec2(50, 50), 30000.0)
    assert not restrictions.point_blocked(Vec2(50, 50), 36000.0)
    assert not restrictions.point_blocked(Vec2(150, 50), 30000.0)
