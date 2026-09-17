"""Restricted / prohibited airspace regions.

Two enforcement modes are supported and are a deliberate modelling choice:

``hard``
    Any transition whose swept segment touches the region is infeasible and is
    never generated as a successor.  Models prohibited airspace.

``soft``
    The transition is allowed but accrues ``penalty_per_nm`` cost units per NM
    flown inside the region.  Models "avoid unless the detour is
    disproportionate" areas such as temporary restrictions or overflight
    charges.

Vertical extent is a simple ``[lower_ft, upper_ft]`` band.  A transition that
changes level is tested against the band spanned by its two endpoints, which is
conservative (it can flag a climb that only clips the band's corner).

Exactness, precisely
--------------------
The *horizontal* intersection tests (:meth:`intersects_segment_2d`) are exact
for all three region types: circle by point-to-segment distance, polygon by
containment plus edge intersection, corridor by segment-to-segment distance.
The *vertical* test is the conservative band-overlap above, and the soft-penalty
overlap fraction is sampled.  So a hard constraint can over-block a climbing
transition but can never under-block one, which is the direction that matters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..core.geometry import (
    Vec2,
    lerp,
    point_in_polygon,
    point_segment_distance,
    segment_circle_intersects,
    segment_segment_distance,
    segments_intersect,
)


class RestrictedRegion(ABC):
    """Interface for all restricted regions."""

    region_id: str
    lower_ft: float
    upper_ft: float
    hard: bool
    penalty_per_nm: float

    @abstractmethod
    def contains_point(self, point_nm: Vec2) -> bool:
        """Horizontal containment only (altitude handled by the band test)."""

    @abstractmethod
    def intersects_segment_2d(self, a_nm: Vec2, b_nm: Vec2) -> bool:
        """Exact horizontal segment/region intersection."""

    def overlaps_altitude(self, alt_a_ft: float, alt_b_ft: float) -> bool:
        lo, hi = min(alt_a_ft, alt_b_ft), max(alt_a_ft, alt_b_ft)
        return not (hi < self.lower_ft or lo > self.upper_ft)

    def intersects_segment(
        self, a_nm: Vec2, b_nm: Vec2, alt_a_ft: float, alt_b_ft: float
    ) -> bool:
        if not self.overlaps_altitude(alt_a_ft, alt_b_ft):
            return False
        return self.intersects_segment_2d(a_nm, b_nm)

    def inside_fraction(
        self,
        a_nm: Vec2,
        b_nm: Vec2,
        alt_a_ft: float,
        alt_b_ft: float,
        samples: int = 8,
    ) -> float:
        """Approximate fraction of the segment lying inside the region.

        Midpoint sampling with ``samples`` sub-intervals.  Used **only** for
        soft penalties, where an O(1/samples) error is acceptable.  Hard
        constraints never go through this method; they use the exact horizontal
        test in :meth:`intersects_segment_2d`.
        """
        if samples <= 0 or not self.overlaps_altitude(alt_a_ft, alt_b_ft):
            return 0.0
        inside = 0
        for i in range(samples):
            t = (i + 0.5) / samples
            alt = alt_a_ft + (alt_b_ft - alt_a_ft) * t
            if self.lower_ft <= alt <= self.upper_ft and self.contains_point(
                lerp(a_nm, b_nm, t)
            ):
                inside += 1
        return inside / samples


@dataclass(frozen=True)
class CircularRestriction(RestrictedRegion):
    region_id: str
    centre_nm: Vec2
    radius_nm: float
    lower_ft: float = 0.0
    upper_ft: float = 60000.0
    hard: bool = True
    penalty_per_nm: float = 0.0

    def contains_point(self, point_nm: Vec2) -> bool:
        return (point_nm - self.centre_nm).norm() <= self.radius_nm

    def intersects_segment_2d(self, a_nm: Vec2, b_nm: Vec2) -> bool:
        return segment_circle_intersects(a_nm, b_nm, self.centre_nm, self.radius_nm)


@dataclass(frozen=True)
class PolygonRestriction(RestrictedRegion):
    region_id: str
    vertices_nm: tuple[Vec2, ...]
    lower_ft: float = 0.0
    upper_ft: float = 60000.0
    hard: bool = True
    penalty_per_nm: float = 0.0

    def contains_point(self, point_nm: Vec2) -> bool:
        return point_in_polygon(point_nm, self.vertices_nm)

    def intersects_segment_2d(self, a_nm: Vec2, b_nm: Vec2) -> bool:
        if self.contains_point(a_nm) or self.contains_point(b_nm):
            return True
        n = len(self.vertices_nm)
        for i in range(n):
            p, q = self.vertices_nm[i], self.vertices_nm[(i + 1) % n]
            if segments_intersect(a_nm, b_nm, p, q):
                return True
        return False


@dataclass(frozen=True)
class CorridorRestriction(RestrictedRegion):
    """A capsule around a line segment: models an airway closure or a fixed
    buffer around a hazardous route."""

    region_id: str
    start_nm: Vec2
    end_nm: Vec2
    half_width_nm: float
    lower_ft: float = 0.0
    upper_ft: float = 60000.0
    hard: bool = False
    penalty_per_nm: float = 0.0

    def contains_point(self, point_nm: Vec2) -> bool:
        return (
            point_segment_distance(point_nm, self.start_nm, self.end_nm)
            <= self.half_width_nm
        )

    def intersects_segment_2d(self, a_nm: Vec2, b_nm: Vec2) -> bool:
        # A capsule is the set of points within ``half_width_nm`` of the spine,
        # so a segment meets it exactly when its minimum distance to the spine
        # is within that width. Exact, with no sampling: an earlier
        # 16-sample version missed segments that clipped the capsule between
        # samples (see tests/test_audit_regressions.py).
        return (
            segment_segment_distance(a_nm, b_nm, self.start_nm, self.end_nm)
            <= self.half_width_nm + 1e-9
        )


@dataclass(frozen=True)
class RestrictionSet:
    """Collection of regions with the queries the planner actually needs."""

    regions: tuple[RestrictedRegion, ...] = ()

    @property
    def hard_regions(self) -> tuple[RestrictedRegion, ...]:
        return tuple(r for r in self.regions if r.hard)

    @property
    def soft_regions(self) -> tuple[RestrictedRegion, ...]:
        return tuple(r for r in self.regions if not r.hard)

    def blocking_region(
        self, a_nm: Vec2, b_nm: Vec2, alt_a_ft: float, alt_b_ft: float
    ) -> RestrictedRegion | None:
        for region in self.hard_regions:
            if region.intersects_segment(a_nm, b_nm, alt_a_ft, alt_b_ft):
                return region
        return None

    def point_blocked(self, point_nm: Vec2, altitude_ft: float) -> bool:
        for region in self.hard_regions:
            if (
                region.lower_ft <= altitude_ft <= region.upper_ft
                and region.contains_point(point_nm)
            ):
                return True
        return False

    def soft_penalty(
        self,
        a_nm: Vec2,
        b_nm: Vec2,
        alt_a_ft: float,
        alt_b_ft: float,
        length_nm: float,
        samples: int = 8,
    ) -> float:
        """Total soft penalty (cost units) accrued over the segment."""
        total = 0.0
        for region in self.soft_regions:
            if region.penalty_per_nm == 0.0:
                continue
            fraction = region.inside_fraction(a_nm, b_nm, alt_a_ft, alt_b_ft, samples)
            total += fraction * length_nm * region.penalty_per_nm
        return total
