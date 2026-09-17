"""Continuous risk field.

Risk is modelled as a scalar *exposure rate* with units of ``risk-units per
hour``.  A trajectory segment accrues ``exposure = mean_density * time_h`` which
the cost model converts to cost.  Using a rate (rather than a per-NM density)
means that slowing down in a hazardous area is correctly penalised.

The numbers are synthetic.  Nothing here is calibrated against any real
convective-weather, turbulence or conflict-probability product; the field is a
configurable scalar surface whose only purpose is to make risk-aware planning
observable.  See ``docs/assumptions.md``.

:meth:`RiskField.min_density` must be a sound *lower* bound over the airspace --
the admissible heuristic uses it.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..core.geometry import Vec2, lerp
from .restrictions import RestrictedRegion


class RiskField(ABC):
    name: str = "risk"

    @abstractmethod
    def density_at(self, x_nm: float, y_nm: float, altitude_ft: float) -> float:
        """Risk exposure rate [risk-units/h]. Must be >= 0."""

    @abstractmethod
    def min_density(self) -> float:
        """Sound lower bound over the airspace (used by the heuristic)."""

    @abstractmethod
    def max_density(self) -> float:
        """Sound upper bound over the airspace (used for reporting/plots)."""

    def mean_along_segment(
        self,
        a_nm: Vec2,
        b_nm: Vec2,
        alt_a_ft: float,
        alt_b_ft: float,
        samples: int = 8,
    ) -> float:
        """Midpoint-rule average of the density along a segment."""
        if samples <= 0:
            return 0.0
        total = 0.0
        for i in range(samples):
            t = (i + 0.5) / samples
            p = lerp(a_nm, b_nm, t)
            alt = alt_a_ft + (alt_b_ft - alt_a_ft) * t
            total += self.density_at(p.x, p.y, alt)
        return total / samples


@dataclass(frozen=True)
class ConstantRisk(RiskField):
    density: float = 0.0
    name: str = "constant"

    def density_at(self, x_nm: float, y_nm: float, altitude_ft: float) -> float:
        return self.density

    def min_density(self) -> float:
        return self.density

    def max_density(self) -> float:
        return self.density


@dataclass(frozen=True)
class GaussianHazard(RiskField):
    """Isotropic Gaussian bump -- a convective cell or a turbulence pocket.

    Optionally limited to an altitude band; outside the band the density is 0.
    """

    centre_nm: Vec2
    peak: float
    sigma_nm: float
    lower_ft: float = 0.0
    upper_ft: float = 60000.0
    name: str = "gaussian"

    def density_at(self, x_nm: float, y_nm: float, altitude_ft: float) -> float:
        if not (self.lower_ft <= altitude_ft <= self.upper_ft):
            return 0.0
        d2 = (x_nm - self.centre_nm.x) ** 2 + (y_nm - self.centre_nm.y) ** 2
        return self.peak * math.exp(-d2 / (2.0 * self.sigma_nm**2))

    def min_density(self) -> float:
        return 0.0

    def max_density(self) -> float:
        return max(0.0, self.peak)


@dataclass(frozen=True)
class ProximityRisk(RiskField):
    """Risk that decays linearly with distance from a restricted region.

    Encodes "legal but uncomfortable": flying 2 NM outside a prohibited zone is
    penalised without being forbidden.
    """

    region: RestrictedRegion
    peak: float
    buffer_nm: float
    name: str = "proximity"

    def density_at(self, x_nm: float, y_nm: float, altitude_ft: float) -> float:
        if not (self.region.lower_ft <= altitude_ft <= self.region.upper_ft):
            return 0.0
        point = Vec2(x_nm, y_nm)
        if self.region.contains_point(point):
            return self.peak
        # Cheap outward scan: distance is approximated by bisection on the ray
        # from the query point toward the region's containment test.
        lo, hi = 0.0, self.buffer_nm
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            if self._within(point, mid):
                hi = mid
            else:
                lo = mid
        if hi >= self.buffer_nm - 1e-9 and not self._within(point, self.buffer_nm):
            return 0.0
        return self.peak * max(0.0, 1.0 - hi / self.buffer_nm)

    def _within(self, point: Vec2, radius_nm: float) -> bool:
        if radius_nm <= 0.0:
            return self.region.contains_point(point)
        for k in range(8):
            angle = 2.0 * math.pi * k / 8.0
            probe = Vec2(
                point.x + radius_nm * math.cos(angle),
                point.y + radius_nm * math.sin(angle),
            )
            if self.region.contains_point(probe):
                return True
        return False

    def min_density(self) -> float:
        return 0.0

    def max_density(self) -> float:
        return max(0.0, self.peak)


@dataclass(frozen=True)
class CompositeRisk(RiskField):
    """Sum of several fields (risk rates add)."""

    fields: tuple[RiskField, ...] = field(default_factory=tuple)
    name: str = "composite"

    def density_at(self, x_nm: float, y_nm: float, altitude_ft: float) -> float:
        return math.fsum(f.density_at(x_nm, y_nm, altitude_ft) for f in self.fields)

    def min_density(self) -> float:
        return math.fsum(f.min_density() for f in self.fields)

    def max_density(self) -> float:
        return math.fsum(f.max_density() for f in self.fields)
