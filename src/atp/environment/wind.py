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
    """Static wind field interface."""

    name: str = "wind"

    @abstractmethod
    def at(self, x_nm: float, y_nm: float, altitude_ft: float) -> Vec2:
        """Wind vector in knots at the given point."""

    @abstractmethod
    def max_magnitude_kt(self) -> float:
        """Sound upper bound on ``|at(...)|`` anywhere in the airspace."""


class DynamicWindField(ABC):
    """Time-dependent wind field interface evaluated at continuous mission time.

    Dynamic fields intentionally do not inherit from :class:`WindField`; they are
    not interchangeable with static wind models because a dynamic observer must
    always choose a mission time explicitly.
    """

    @abstractmethod
    def at_time(
        self, x_nm: float, y_nm: float, altitude_ft: float, t_h: float
    ) -> Vec2:
        """Wind vector in knots at the given point and mission time ``t_h``."""

    @abstractmethod
    def max_magnitude_kt(self) -> float:
        """Sound upper bound on ``|at_time(...)|`` anywhere in the airspace."""


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


@dataclass(frozen=True)
class PeriodicWind(DynamicWindField):
    """Deterministic synthetic periodic wind field.

    The field is a superposition of:
      - a constant vector background, and
      - a spatially localised periodic modulation in both x/y and altitude.

    The model is intentionally simple and analytically bounded:

        W(x, y, h, t) = base + amp * sin(2*pi*t/T + phase(x, y, h)) * dir

    where the modulation is deterministic, continuous in time, and bounded by
    ``amp`` in each component.  The vector direction is chosen from the local
    displacement relative to a centre, giving a predictable rotating/oscillating
    jet-like flow without randomness or external data.
    """

    centre_nm: Vec2
    base_vector_kt: Vec2
    amplitude_kt: float
    period_h: float
    altitude_scale_ft: float
    phase_offset: float = 0.0
    name: str = "periodic"

    def __post_init__(self) -> None:
        if not math.isfinite(self.period_h) or self.period_h <= 0.0:
            raise ValueError("period_h must be finite and > 0")
        if not math.isfinite(self.altitude_scale_ft) or self.altitude_scale_ft <= 0.0:
            raise ValueError("altitude_scale_ft must be finite and > 0")
        if not math.isfinite(self.amplitude_kt):
            raise ValueError("amplitude_kt must be finite")
        if not math.isfinite(self.phase_offset):
            raise ValueError("phase_offset must be finite")
        if not math.isfinite(self.base_vector_kt.x) or not math.isfinite(self.base_vector_kt.y):
            raise ValueError("base_vector_kt components must be finite")
        if not math.isfinite(self.centre_nm.x) or not math.isfinite(self.centre_nm.y):
            raise ValueError("centre_nm coordinates must be finite")

    def at_time(
        self, x_nm: float, y_nm: float, altitude_ft: float, t_h: float
    ) -> Vec2:
        offset = Vec2(x_nm, y_nm) - self.centre_nm
        radial = offset.normalized() if offset.norm() > 1e-9 else Vec2(0.0, 0.0)
        phase = (
            2.0
            * math.pi
            * (t_h / self.period_h)
            + self.phase_offset
            + offset.norm() / max(1.0, self.altitude_scale_ft / 1000.0)
            + altitude_ft / self.altitude_scale_ft
        )
        modulation = math.sin(phase)
        return self.base_vector_kt + radial * (modulation * self.amplitude_kt)

    def max_magnitude_kt(self) -> float:
        """A sound bound for the deterministic periodic flow.

        The field is a base vector plus a bounded modulation with magnitude at most
        ``amplitude_kt`` in the local radial direction, so the total magnitude is
        bounded by ``|base| + amplitude_kt``.
        """
        return self.base_vector_kt.norm() + abs(self.amplitude_kt)
