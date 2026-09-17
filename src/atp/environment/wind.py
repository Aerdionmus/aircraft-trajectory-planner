"""Wind field models.

A wind field maps a point in the airspace to a horizontal flow vector in knots.
The vector points in the direction the air is *moving toward* (not the
meteorological "from" convention -- use
:func:`atp.core.geometry.met_wind_to_vector` to convert).

Every field must expose :meth:`WindField.max_magnitude_kt`, a **sound upper
bound** over the whole airspace.  The admissible heuristic in
:mod:`atp.planning.heuristics` relies on that bound; if an implementation
returns a value that is too small, A* loses its optimality guarantee.  Tests in
``tests/test_wind.py`` sample each field to check the bound empirically.

Vertical wind (updraughts/downdraughts) is deliberately not modelled.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..core.geometry import Vec2, ZERO


class WindField(ABC):
    """Interface for all wind models."""

    name: str = "wind"

    @abstractmethod
    def at(self, x_nm: float, y_nm: float, altitude_ft: float) -> Vec2:
        """Wind vector in knots at the given point."""

    @abstractmethod
    def max_magnitude_kt(self) -> float:
        """Sound upper bound on ``|at(...)|`` anywhere in the airspace."""


@dataclass(frozen=True)
class ZeroWind(WindField):
    name: str = "zero"

    def at(self, x_nm: float, y_nm: float, altitude_ft: float) -> Vec2:
        return ZERO

    def max_magnitude_kt(self) -> float:
        return 0.0


@dataclass(frozen=True)
class UniformWind(WindField):
    """Constant wind everywhere."""

    vector_kt: Vec2
    name: str = "uniform"

    def at(self, x_nm: float, y_nm: float, altitude_ft: float) -> Vec2:
        return self.vector_kt

    def max_magnitude_kt(self) -> float:
        return self.vector_kt.norm()


@dataclass(frozen=True)
class LayeredWind(WindField):
    """Piecewise-constant wind per altitude band.

    ``layers`` is an ordered tuple of ``(upper_bound_ft, vector_kt)``; the first
    layer whose bound is >= the query altitude wins.  ``above`` applies beyond
    the last bound.  This is the cheapest way to model a jet-stream core that
    only exists at cruise levels and therefore makes level choice meaningful.
    """

    layers: tuple[tuple[float, Vec2], ...]
    above: Vec2 = ZERO
    name: str = "layered"

    def at(self, x_nm: float, y_nm: float, altitude_ft: float) -> Vec2:
        for bound_ft, vector in self.layers:
            if altitude_ft <= bound_ft:
                return vector
        return self.above

    def max_magnitude_kt(self) -> float:
        return max(
            [v.norm() for _, v in self.layers] + [self.above.norm()], default=0.0
        )


@dataclass(frozen=True)
class VortexWind(WindField):
    """Rankine-style vortex: solid-body rotation inside the core, ``1/r`` decay
    outside.  Used to create a locally strong, spatially varying field so that
    wind-aware planning differs visibly from pure distance minimisation."""

    centre_nm: Vec2
    peak_speed_kt: float
    core_radius_nm: float
    clockwise: bool = False
    name: str = "vortex"

    def at(self, x_nm: float, y_nm: float, altitude_ft: float) -> Vec2:
        offset = Vec2(x_nm, y_nm) - self.centre_nm
        r = offset.norm()
        if r < 1e-9:
            return ZERO
        if r <= self.core_radius_nm:
            speed = self.peak_speed_kt * (r / self.core_radius_nm)
        else:
            speed = self.peak_speed_kt * (self.core_radius_nm / r)
        tangential = Vec2(-offset.y, offset.x).normalized()
        if self.clockwise:
            tangential = tangential * -1.0
        return tangential * speed

    def max_magnitude_kt(self) -> float:
        return abs(self.peak_speed_kt)


@dataclass(frozen=True)
class CompositeWind(WindField):
    """Vector sum of several fields. The bound is the sum of the bounds, which
    is sound but loose."""

    fields: tuple[WindField, ...] = field(default_factory=tuple)
    name: str = "composite"

    def at(self, x_nm: float, y_nm: float, altitude_ft: float) -> Vec2:
        total = ZERO
        for f in self.fields:
            total = total + f.at(x_nm, y_nm, altitude_ft)
        return total

    def max_magnitude_kt(self) -> float:
        return math.fsum(f.max_magnitude_kt() for f in self.fields)
