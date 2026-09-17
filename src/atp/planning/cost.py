"""Transition cost model.

Cost aggregation
----------------
Time, fuel, distance and risk are physically incommensurable, so they are not
"weighted" with dimensionless factors.  Instead each weight is an explicit
*unit price* and the sum is in abstract cost units::

    cost = c_time [cu/h]   * time_h
         + c_fuel [cu/kg]  * fuel_kg
         + c_dist [cu/NM]  * distance_nm
         + c_risk [cu/exp] * risk_exposure
         + soft_restriction_penalty [cu]

This is the same idea as an airline Cost Index (time price / fuel price) but
kept general.  Setting every price but one to zero recovers the single-objective
planners (shortest path, minimum time, minimum fuel), which is what the
experiment harness does.

Invariants the search depends on
--------------------------------
1. Every edge cost is finite and **non-negative** (required by A*/Dijkstra).
   Descent fuel credits are clamped so this cannot be violated.
2. ``lower_bound_cost_per_nm`` must never exceed the true cost per NM of any
   feasible transition.  The admissible heuristic is built from it.
3. Infeasible transitions are reported as such, never as "very expensive".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ..aircraft.kinematics import max_possible_ground_speed_kt, solve_ground_speed
from ..aircraft.performance import AircraftPerformance
from ..core.geometry import Vec2
from ..core.units import MIN_PER_H, ft_to_nm
from ..environment.airspace import Airspace, GridState


@dataclass(frozen=True)
class CostWeights:
    """Unit prices. All must be non-negative."""

    time_cost_per_hour: float = 1.0
    fuel_cost_per_kg: float = 0.0
    distance_cost_per_nm: float = 0.0
    risk_cost_per_exposure: float = 0.0
    #: Multiplies the soft-restriction penalties declared on each region.
    restriction_penalty_scale: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "time_cost_per_hour",
            "fuel_cost_per_kg",
            "distance_cost_per_nm",
            "risk_cost_per_exposure",
            "restriction_penalty_scale",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")

    def is_degenerate(self) -> bool:
        """True if no priced term can grow with path length (every path costs
        the same), which makes the search result meaningless."""
        return (
            self.time_cost_per_hour == 0.0
            and self.fuel_cost_per_kg == 0.0
            and self.distance_cost_per_nm == 0.0
        )


@dataclass(frozen=True)
class SegmentMetrics:
    """Everything one transition contributes, before pricing."""

    feasible: bool
    ground_distance_nm: float = 0.0
    vertical_ft: float = 0.0
    time_h: float = 0.0
    fuel_kg: float = 0.0
    risk_exposure: float = 0.0
    restriction_penalty: float = 0.0
    ground_speed_kt: float = 0.0
    mean_risk_density: float = 0.0
    infeasible_reason: str = ""

    @staticmethod
    def infeasible(reason: str) -> "SegmentMetrics":
        return SegmentMetrics(feasible=False, infeasible_reason=reason)

    def __add__(self, other: "SegmentMetrics") -> "SegmentMetrics":
        """Accumulate along a trajectory (ground speed becomes distance-weighted
        mean; feasibility is the conjunction)."""
        total_d = self.ground_distance_nm + other.ground_distance_nm
        if total_d > 0:
            gs = (
                self.ground_speed_kt * self.ground_distance_nm
                + other.ground_speed_kt * other.ground_distance_nm
            ) / total_d
        else:
            gs = 0.0
        return SegmentMetrics(
            feasible=self.feasible and other.feasible,
            ground_distance_nm=total_d,
            vertical_ft=self.vertical_ft + other.vertical_ft,
            time_h=self.time_h + other.time_h,
            fuel_kg=self.fuel_kg + other.fuel_kg,
            risk_exposure=self.risk_exposure + other.risk_exposure,
            restriction_penalty=self.restriction_penalty + other.restriction_penalty,
            ground_speed_kt=gs,
            mean_risk_density=max(self.mean_risk_density, other.mean_risk_density),
            infeasible_reason=self.infeasible_reason or other.infeasible_reason,
        )


@dataclass(frozen=True)
class CostBreakdown:
    time: float = 0.0
    fuel: float = 0.0
    distance: float = 0.0
    risk: float = 0.0
    restriction: float = 0.0

    @property
    def total(self) -> float:
        return self.time + self.fuel + self.distance + self.risk + self.restriction

    def __add__(self, other: "CostBreakdown") -> "CostBreakdown":
        return CostBreakdown(
            self.time + other.time,
            self.fuel + other.fuel,
            self.distance + other.distance,
            self.risk + other.risk,
            self.restriction + other.restriction,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "time": self.time,
            "fuel": self.fuel,
            "distance": self.distance,
            "risk": self.risk,
            "restriction": self.restriction,
            "total": self.total,
        }


ZERO_BREAKDOWN = CostBreakdown()


class CostModel:
    """Evaluates and prices a transition between two adjacent grid states."""

    def __init__(
        self,
        airspace: Airspace,
        aircraft: AircraftPerformance,
        weights: CostWeights,
        *,
        integration_samples: int = 6,
        enforce_vertical_rate: bool = True,
    ) -> None:
        if integration_samples < 1:
            raise ValueError("integration_samples must be >= 1")
        self.airspace = airspace
        self.aircraft = aircraft
        self.weights = weights
        self.integration_samples = integration_samples
        self.enforce_vertical_rate = enforce_vertical_rate

    # -- physical evaluation -------------------------------------------------
    def evaluate(self, a: GridState, b: GridState) -> SegmentMetrics:
        spec = self.airspace.spec
        pa, pb = spec.centre_nm(a), spec.centre_nm(b)
        alt_a, alt_b = spec.altitude_ft(a.il), spec.altitude_ft(b.il)
        delta_alt = alt_b - alt_a

        if not self.aircraft.can_operate_at(max(alt_a, alt_b)):
            return SegmentMetrics.infeasible("above service ceiling")

        blocking = self.airspace.restrictions.blocking_region(pa, pb, alt_a, alt_b)
        if blocking is not None:
            return SegmentMetrics.infeasible(f"hard restriction {blocking.region_id}")

        delta = pb - pa
        horizontal_nm = delta.norm()

        if horizontal_nm < 1e-9:
            return self._pure_vertical(a, b, pa, alt_a, alt_b, delta_alt)

        track = delta.normalized()
        # Mid-segment wind sample: one evaluation per edge keeps the search loop
        # cheap; the error is O(h^2) in cell size for smooth fields.
        mid = pa + delta * 0.5
        mid_alt = 0.5 * (alt_a + alt_b)
        wind = self.airspace.wind.at(mid.x, mid.y, mid_alt)
        solution = solve_ground_speed(self.aircraft.tas_kt(mid_alt), wind, track)
        if not solution.feasible:
            return SegmentMetrics.infeasible(solution.reason)

        time_h = horizontal_nm / solution.ground_speed_kt

        if self.enforce_vertical_rate and delta_alt != 0.0:
            required_fpm = abs(delta_alt) / (time_h * MIN_PER_H)
            limit = self.aircraft.max_vertical_rate_fpm(delta_alt)
            if required_fpm > limit + 1e-6:
                return SegmentMetrics.infeasible(
                    f"requires {required_fpm:.0f} fpm > {limit:.0f} fpm"
                )

        return self._finish(
            pa, pb, alt_a, alt_b, horizontal_nm, delta_alt, time_h, solution.ground_speed_kt
        )

    def _pure_vertical(
        self,
        a: GridState,
        b: GridState,
        pa: Vec2,
        alt_a: float,
        alt_b: float,
        delta_alt: float,
    ) -> SegmentMetrics:
        """Level change with no horizontal displacement.

        Charged at the maximum vertical rate.  Note the aircraft does in reality
        travel forward during the manoeuvre; that displacement is ignored, which
        makes pure level changes slightly optimistic in distance terms.
        """
        if delta_alt == 0.0:
            return SegmentMetrics.infeasible("null transition")
        rate_fpm = self.aircraft.max_vertical_rate_fpm(delta_alt)
        time_h = abs(delta_alt) / (rate_fpm * MIN_PER_H)
        return self._finish(pa, pa, alt_a, alt_b, 0.0, delta_alt, time_h, 0.0)

    def _finish(
        self,
        pa: Vec2,
        pb: Vec2,
        alt_a: float,
        alt_b: float,
        horizontal_nm: float,
        delta_alt: float,
        time_h: float,
        ground_speed_kt: float,
    ) -> SegmentMetrics:
        mid_alt = 0.5 * (alt_a + alt_b)
        fuel = self.aircraft.fuel_flow_kg_per_h(mid_alt) * time_h
        fuel += self.aircraft.vertical_fuel_delta_kg(delta_alt)
        fuel = max(0.0, fuel)  # invariant 1: no negative edge costs

        mean_risk = self.airspace.risk.mean_along_segment(
            pa, pb, alt_a, alt_b, self.integration_samples
        )
        exposure = mean_risk * time_h

        penalty = (
            self.airspace.restrictions.soft_penalty(
                pa, pb, alt_a, alt_b, horizontal_nm, self.integration_samples
            )
            * self.weights.restriction_penalty_scale
        )

        # 3D path length, so that a level change is not free in distance terms.
        distance_nm = math.hypot(horizontal_nm, ft_to_nm(delta_alt))

        return SegmentMetrics(
            feasible=True,
            ground_distance_nm=distance_nm,
            vertical_ft=abs(delta_alt),
            time_h=time_h,
            fuel_kg=fuel,
            risk_exposure=exposure,
            restriction_penalty=penalty,
            ground_speed_kt=ground_speed_kt,
            mean_risk_density=mean_risk,
        )

    # -- pricing -------------------------------------------------------------
    def price(self, metrics: SegmentMetrics) -> CostBreakdown:
        if not metrics.feasible:
            return ZERO_BREAKDOWN
        w = self.weights
        return CostBreakdown(
            time=w.time_cost_per_hour * metrics.time_h,
            fuel=w.fuel_cost_per_kg * metrics.fuel_kg,
            distance=w.distance_cost_per_nm * metrics.ground_distance_nm,
            risk=w.risk_cost_per_exposure * metrics.risk_exposure,
            restriction=metrics.restriction_penalty,
        )

    def transition_cost(self, a: GridState, b: GridState) -> tuple[float, SegmentMetrics]:
        metrics = self.evaluate(a, b)
        if not metrics.feasible:
            return math.inf, metrics
        return self.price(metrics).total, metrics

    # -- heuristic support ---------------------------------------------------
    def lower_bound_cost_per_nm(self) -> float:
        """Lower bound on cost per NM of *horizontal* progress.

        Derivation.  Over any feasible transition the ground speed is at most
        ``GS_max = TAS_max + |w|_max``, so 1 NM takes at least ``1/GS_max``
        hours and burns at least ``ff_min / GS_max`` kg.  Risk density is at
        least ``risk_min`` and soft penalties are non-negative.  Hence

            cost/NM >= c_time/GS_max + c_fuel*ff_min/GS_max
                     + c_dist + c_risk*risk_min/GS_max

        Climb fuel and the 3D-length correction are dropped, both of which only
        make the bound looser (still admissible).
        """
        spec = self.airspace.spec
        gs_max = max_possible_ground_speed_kt(
            self.aircraft.max_tas_kt, self.airspace.wind.max_magnitude_kt()
        )
        if gs_max <= 0.0:
            return 0.0
        ff_min = self.aircraft.min_fuel_flow_over_levels_kg_per_h(spec.flight_levels)
        risk_min = max(0.0, self.airspace.risk.min_density())
        w = self.weights
        return (
            w.time_cost_per_hour / gs_max
            + w.fuel_cost_per_kg * ff_min / gs_max
            + w.distance_cost_per_nm
            + w.risk_cost_per_exposure * risk_min / gs_max
        )

    def with_weights(self, weights: CostWeights) -> "CostModel":
        return CostModel(
            self.airspace,
            self.aircraft,
            weights,
            integration_samples=self.integration_samples,
            enforce_vertical_rate=self.enforce_vertical_rate,
        )


def scale_weights(weights: CostWeights, **overrides: float) -> CostWeights:
    return replace(weights, **overrides)
