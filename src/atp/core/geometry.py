"""Minimal 2D geometry used by the airspace, wind and restriction models.

Deliberately dependency-free and immutable so that every downstream module can
be unit tested without numerical libraries or global state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

EPS: float = 1e-12


@dataclass(frozen=True, slots=True)
class Vec2:
    """A planar vector. Interpretation (position in NM, velocity in kt) is the
    caller's responsibility and is always encoded in the variable name."""

    x: float
    y: float

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    __rmul__ = __mul__

    def dot(self, other: "Vec2") -> float:
        return self.x * other.x + self.y * other.y

    def cross(self, other: "Vec2") -> float:
        return self.x * other.y - self.y * other.x

    def norm(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> "Vec2":
        n = self.norm()
        if n < EPS:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / n, self.y / n)

    def rotated(self, angle_rad: float) -> "Vec2":
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        return Vec2(self.x * c - self.y * s, self.x * s + self.y * c)

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


ZERO = Vec2(0.0, 0.0)


def lerp(a: Vec2, b: Vec2, t: float) -> Vec2:
    return Vec2(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)


def distance(a: Vec2, b: Vec2) -> float:
    return math.hypot(b.x - a.x, b.y - a.y)


def bearing_deg(a: Vec2, b: Vec2) -> float:
    """True bearing from ``a`` to ``b`` in degrees, 0 = north, clockwise."""
    dx, dy = b.x - a.x, b.y - a.y
    return math.degrees(math.atan2(dx, dy)) % 360.0


def met_wind_to_vector(direction_from_deg: float, speed_kt: float) -> Vec2:
    """Convert a meteorological wind report to a flow vector.

    Aviation convention reports the direction the wind blows *from*.  The
    returned vector points where the air is *going*, which is what the wind
    triangle needs.
    """
    heading_to_rad = math.radians((direction_from_deg + 180.0) % 360.0)
    return Vec2(speed_kt * math.sin(heading_to_rad), speed_kt * math.cos(heading_to_rad))


def point_segment_distance(p: Vec2, a: Vec2, b: Vec2) -> float:
    """Shortest distance from point ``p`` to segment ``ab``."""
    ab = b - a
    denom = ab.dot(ab)
    if denom < EPS:
        return distance(p, a)
    t = max(0.0, min(1.0, (p - a).dot(ab) / denom))
    return distance(p, a + ab * t)


def segments_intersect(p1: Vec2, p2: Vec2, q1: Vec2, q2: Vec2) -> bool:
    """Proper/improper intersection test for two closed segments."""

    def orient(a: Vec2, b: Vec2, c: Vec2) -> float:
        return (b - a).cross(c - a)

    def on_segment(a: Vec2, b: Vec2, c: Vec2) -> bool:
        return (
            min(a.x, b.x) - EPS <= c.x <= max(a.x, b.x) + EPS
            and min(a.y, b.y) - EPS <= c.y <= max(a.y, b.y) + EPS
        )

    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    if ((d1 > EPS and d2 < -EPS) or (d1 < -EPS and d2 > EPS)) and (
        (d3 > EPS and d4 < -EPS) or (d3 < -EPS and d4 > EPS)
    ):
        return True
    if abs(d1) <= EPS and on_segment(q1, q2, p1):
        return True
    if abs(d2) <= EPS and on_segment(q1, q2, p2):
        return True
    if abs(d3) <= EPS and on_segment(p1, p2, q1):
        return True
    if abs(d4) <= EPS and on_segment(p1, p2, q2):
        return True
    return False


def point_in_polygon(p: Vec2, vertices: tuple[Vec2, ...]) -> bool:
    """Ray-casting test for a simple (not necessarily convex) polygon.

    Points exactly on the boundary are reported as inside, which is the
    conservative choice for a no-fly zone.
    """
    n = len(vertices)
    if n < 3:
        return False
    for i in range(n):
        a, b = vertices[i], vertices[(i + 1) % n]
        if point_segment_distance(p, a, b) < 1e-9:
            return True
    inside = False
    for i in range(n):
        a, b = vertices[i], vertices[(i + 1) % n]
        if (a.y > p.y) != (b.y > p.y):
            x_at = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y)
            if x_at > p.x:
                inside = not inside
    return inside


def segment_circle_intersects(a: Vec2, b: Vec2, centre: Vec2, radius_nm: float) -> bool:
    return point_segment_distance(centre, a, b) <= radius_nm + 1e-9
