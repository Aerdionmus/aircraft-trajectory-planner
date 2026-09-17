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
2. ``speed_cost_lower_bound_per_nm`` must never exceed the time+fuel+risk cost
   per NM of *horizontal* progress on any feasible transition.  The admissible
   heuristic is built from it, together with the distance price applied to 3D
   length.
3. Infeasible transitions are reported as such, never as "very expensive".
4. **Monotone refinement.**  No Milestone 2 refinement may lower the cost of a
   transition relative to its Milestone 1 value.  Turn modelling may only add
   cost (time, fuel, risk) or remove edges; it never adds or subtracts
   distance, and in particular the geometric shortening a fly-by arc produces
   is measured (``corner_cut_nm``) but never credited.  This invariant is what
   makes the inherited-admissibility argument in :mod:`atp.planning.heuristics`
   hold, and it is checked directly by ``tests/test_turn_cost.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ..aircraft.kinematics import max_possible_ground_speed_kt, solve_ground_speed
from ..aircraft.performance import AircraftPerformance
from ..aircraft.turn import (
    NEGLIGIBLE_TURN_RAD,
    corner_cut_nm,
    ground_curvature_radius_bound_nm,
    tangent_length_nm,
    turn_time_h,
)
from ..core.geometry import Vec2, angle_between
from ..core.units import MIN_PER_H, ft_to_nm
from ..environment.airspace import Airspace, GridState

#: Accepted turn models.  ``none`` reproduces Milestone 1 exactly: heading never
#: enters the state and no turn is evaluated.  ``gate`` applies the legality
#: tests but charges nothing, isolating the effect of pruning.  ``gate+cost``
#: is the full model.
TURN_MODELS: tuple[str, ...] = ("none", "gate", "gate+cost")


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
class TurnMetrics:
    """What one corner contributes, before pricing.

    ``delta_heading_rad`` is the change in *air* heading, which in wind differs
    from ``delta_track_rad``, the change in ground track: the aircraft rolls
    about its own axis, so a bank limit constrains the former.
    ``corner_cut_nm`` is a diagnostic only -- see
    :func:`atp.aircraft.turn.corner_cut_nm`.
    """

    feasible: bool
    delta_heading_rad: float = 0.0
    delta_track_rad: float = 0.0
    time_h: float = 0.0
    fuel_kg: float = 0.0
    risk_exposure: float = 0.0
    corner_cut_nm: float = 0.0
    radius_nm: float = 0.0
    infeasible_reason: str = ""

    @staticmethod
    def infeasible(reason: str) -> "TurnMetrics":
        return TurnMetrics(feasible=False, infeasible_reason=reason)


#: The "no corner here" turn: start state, pure level change, or turn_model none.
NO_TURN = TurnMetrics(feasible=True)


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
    #: Turn contributions, already included in ``time_h`` / ``fuel_kg`` /
    #: ``risk_exposure`` above and repeated here for reporting.
    turn_time_h: float = 0.0
    turn_fuel_kg: float = 0.0
    heading_change_rad: float = 0.0
    track_change_rad: float = 0.0
    corner_cut_nm: float = 0.0
    num_turns: int = 0
    max_heading_change_rad: float = 0.0

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
            turn_time_h=self.turn_time_h + other.turn_time_h,
            turn_fuel_kg=self.turn_fuel_kg + other.turn_fuel_kg,
            heading_change_rad=self.heading_change_rad + other.heading_change_rad,
            track_change_rad=self.track_change_rad + other.track_change_rad,
            corner_cut_nm=self.corner_cut_nm + other.corner_cut_nm,
            num_turns=self.num_turns + other.num_turns,
            max_heading_change_rad=max(
                self.max_heading_change_rad, other.max_heading_change_rad
            ),
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
        turn_model: str = "none",
        max_turn_deg: float = 180.0,
    ) -> None:
        if integration_samples < 1:
            raise ValueError("integration_samples must be >= 1")
        if turn_model not in TURN_MODELS:
            raise ValueError(f"turn_model must be one of {list(TURN_MODELS)}")
        if not 0.0 < max_turn_deg <= 180.0:
            raise ValueError("max_turn_deg must lie in (0, 180]")
        self.airspace = airspace
        self.aircraft = aircraft
        self.weights = weights
        self.integration_samples = integration_samples
        self.enforce_vertical_rate = enforce_vertical_rate
        self.turn_model = turn_model
        self.max_turn_deg = max_turn_deg

    @property
    def models_turns(self) -> bool:
        return self.turn_model != "none"

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

        # Hard restrictions were already rejected exactly in `evaluate`; this
        # is the sampled soft-penalty overlap only.
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

    # -- turns ---------------------------------------------------------------
    def turn_metrics(
        self,
        node: GridState,
        in_track_unit: Vec2 | None,
        out_track_unit: Vec2,
        leg_in_nm: float,
        leg_out_nm: float,
    ) -> TurnMetrics:
        """Evaluate the corner flown at ``node`` between two ground tracks.

        ``in_track_unit is None`` means there is no incoming leg (the start
        state, or a preceding pure level change): no corner, no charge, no gate.

        Both legs are evaluated with the wind sampled at the **node centre**, not
        at their own midpoints.  That is an approximation of the same order as
        the mid-segment sampling already used for a leg, and it is what makes the
        incoming move index a sufficient statistic: without it the air heading
        arriving at a node would depend on the previous leg's midpoint wind,
        which is not in the state.  ``docs/assumptions.md`` records it.
        """
        if not self.models_turns or in_track_unit is None:
            return NO_TURN
        if leg_in_nm <= 0.0 or leg_out_nm <= 0.0:
            return NO_TURN

        spec = self.airspace.spec
        p = spec.centre_nm(node)
        alt = spec.altitude_ft(node.il)
        tas = self.aircraft.tas_kt(alt)
        wind = self.airspace.wind.at(p.x, p.y, alt)

        incoming = solve_ground_speed(tas, wind, in_track_unit)
        if not incoming.feasible:
            return TurnMetrics.infeasible(
                f"incoming track unflyable at node: {incoming.reason}"
            )
        outgoing = solve_ground_speed(tas, wind, out_track_unit)
        if not outgoing.feasible:
            return TurnMetrics.infeasible(
                f"outgoing track unflyable at node: {outgoing.reason}"
            )

        delta_heading = angle_between(
            incoming.air_heading_unit, outgoing.air_heading_unit
        )
        delta_track = angle_between(in_track_unit, out_track_unit)

        if delta_heading < NEGLIGIBLE_TURN_RAD:
            # Straight flight: legal by definition and free, but still reported.
            return TurnMetrics(
                feasible=True,
                delta_heading_rad=delta_heading,
                delta_track_rad=delta_track,
            )

        if math.degrees(delta_heading) > self.max_turn_deg + 1e-9:
            return TurnMetrics.infeasible(
                f"heading change {math.degrees(delta_heading):.1f} deg "
                f"exceeds cap {self.max_turn_deg:.1f} deg"
            )

        omega = self.aircraft.turn_rate_rad_per_h(alt)
        # A sound upper bound on the ground-path radius of curvature reached
        # at ANY instant of the turn, not just at the two leg endpoints -- see
        # ground_curvature_radius_bound_nm for the derivation.  ``GS / omega``
        # at an endpoint speed is not sound: in a tailwind component it can
        # under-estimate the true radius, the wrong direction for a gate that
        # is documented elsewhere to over-block rather than under-block.
        radius = ground_curvature_radius_bound_nm(
            tas, wind.norm(), self.aircraft.max_bank_deg
        )
        available = 0.5 * min(leg_in_nm, leg_out_nm)
        tangent = tangent_length_nm(radius, delta_heading)
        if tangent > available + 1e-9:
            return TurnMetrics.infeasible(
                f"turn of {math.degrees(delta_heading):.1f} deg needs "
                f"{tangent:.2f} NM of leg, {available:.2f} NM available"
            )

        cut = corner_cut_nm(radius, delta_heading)
        if self.turn_model == "gate":
            return TurnMetrics(
                feasible=True,
                delta_heading_rad=delta_heading,
                delta_track_rad=delta_track,
                corner_cut_nm=cut,
                radius_nm=radius,
            )

        time_h = turn_time_h(delta_heading, omega)
        fuel = (
            self.aircraft.fuel_flow_kg_per_h(alt)
            * time_h
            * self.aircraft.turn_fuel_factor
        )
        exposure = max(0.0, self.airspace.risk.density_at(p.x, p.y, alt)) * time_h
        return TurnMetrics(
            feasible=True,
            delta_heading_rad=delta_heading,
            delta_track_rad=delta_track,
            time_h=time_h,
            fuel_kg=max(0.0, fuel),
            risk_exposure=exposure,
            corner_cut_nm=cut,
            radius_nm=radius,
        )

    @staticmethod
    def with_turn(metrics: SegmentMetrics, turn: TurnMetrics) -> SegmentMetrics:
        """Fold a corner into the following leg's metrics.

        Turn time, fuel and risk are added to the leg totals so that the existing
        pricing path handles them with no new weight; distance is untouched, so
        invariant 4 holds by construction.
        """
        if not metrics.feasible:
            return metrics
        if not turn.feasible:
            return SegmentMetrics.infeasible(turn.infeasible_reason)
        charged = turn.delta_heading_rad >= NEGLIGIBLE_TURN_RAD
        return replace(
            metrics,
            time_h=metrics.time_h + turn.time_h,
            fuel_kg=metrics.fuel_kg + turn.fuel_kg,
            risk_exposure=metrics.risk_exposure + turn.risk_exposure,
            turn_time_h=metrics.turn_time_h + turn.time_h,
            turn_fuel_kg=metrics.turn_fuel_kg + turn.fuel_kg,
            heading_change_rad=metrics.heading_change_rad + turn.delta_heading_rad,
            track_change_rad=metrics.track_change_rad + turn.delta_track_rad,
            corner_cut_nm=metrics.corner_cut_nm + turn.corner_cut_nm,
            num_turns=metrics.num_turns + (1 if charged else 0),
            max_heading_change_rad=max(
                metrics.max_heading_change_rad, turn.delta_heading_rad
            ),
        )

    def transition_cost_with_turn(
        self, a: GridState, b: GridState, turn: TurnMetrics
    ) -> tuple[float, SegmentMetrics]:
        metrics = self.with_turn(self.evaluate(a, b), turn)
        if not metrics.feasible:
            return math.inf, metrics
        return self.price(metrics).total, metrics

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
    def speed_cost_lower_bound_per_nm(self) -> float:
        """Lower bound on the *speed-derived* cost per NM of **horizontal**
        progress, i.e. the time, fuel and risk terms only.

        Derivation.  Over any feasible transition the ground speed is at most
        ``GS_max = TAS_max + |w|_max``, so 1 NM of horizontal travel takes at
        least ``1/GS_max`` hours and burns at least ``ff_min / GS_max`` kg, and
        accrues at least ``risk_min / GS_max`` exposure.  Hence for any
        trajectory

            time/fuel/risk cost >= A * (horizontal path length)

        with ``A`` as returned here.  Climb fuel and soft penalties are dropped;
        both are non-negative, so the bound only gets looser.

        The distance term is deliberately **not** included: it is priced against
        3D length, not horizontal length, and folding the two together produced
        an inadmissible heuristic whenever the goal flight level was
        constrained.  See :mod:`atp.planning.heuristics`.

        The fuel term is dropped entirely in the pathological case where a
        descent credits more fuel per 1000 ft than a climb costs, because then
        a climb/descent cycle would be fuel-positive and no per-NM fuel floor
        exists.  Otherwise the (bounded) credit is handled by
        :meth:`constant_cost_offset`.
        """
        spec = self.airspace.spec
        gs_max = max_possible_ground_speed_kt(
            self.aircraft.max_tas_kt, self.airspace.wind.max_magnitude_kt()
        )
        if gs_max <= 0.0:
            return 0.0
        risk_min = max(0.0, self.airspace.risk.min_density())
        w = self.weights
        ff_min = (
            self.aircraft.min_fuel_flow_over_levels_kg_per_h(spec.flight_levels)
            if self._descent_credit_is_bounded()
            else 0.0
        )
        return (
            w.time_cost_per_hour
            + w.fuel_cost_per_kg * ff_min
            + w.risk_cost_per_exposure * risk_min
        ) / gs_max

    def _descent_credit_is_bounded(self) -> bool:
        """True when a climb costs at least as much fuel as the matching descent
        refunds, so no climb/descent cycle can generate fuel."""
        return (
            self.aircraft.descent_fuel_credit_kg_per_1000ft
            <= self.aircraft.climb_fuel_penalty_kg_per_1000ft + 1e-12
        )

    def constant_cost_offset(self) -> float:
        """Constant that a distance-proportional lower bound must give back.

        Segment fuel is ``ff * t + vertical_delta``, and ``vertical_delta`` is
        negative on a descent.  That term is *not* proportional to distance, so
        ``speed_cost_lower_bound_per_nm() * distance`` can exceed the true fuel
        cost of a descending trajectory -- which is exactly the defect this
        method exists to cancel.

        Because a climb costs at least as much as the matching descent refunds,
        the most fuel any trajectory can be credited is a single monotone
        descent across the full altitude span:

            credit_max = (alt_span_ft / 1000) * descent_fuel_credit_kg_per_1000ft

        Zero whenever there is one flight level, fuel is unpriced, or the
        aircraft has no descent credit -- i.e. in most configurations.
        """
        if not self._descent_credit_is_bounded():
            return 0.0  # the fuel term was already dropped from the bound
        spec = self.airspace.spec
        if spec.num_levels < 2 or self.weights.fuel_cost_per_kg == 0.0:
            return 0.0
        span_ft = spec.altitude_ft(spec.num_levels - 1) - spec.altitude_ft(0)
        credit_kg = (span_ft / 1000.0) * self.aircraft.descent_fuel_credit_kg_per_1000ft
        return self.weights.fuel_cost_per_kg * max(0.0, credit_kg)

    def with_weights(self, weights: CostWeights) -> "CostModel":
        return CostModel(
            self.airspace,
            self.aircraft,
            weights,
            integration_samples=self.integration_samples,
            enforce_vertical_rate=self.enforce_vertical_rate,
            turn_model=self.turn_model,
            max_turn_deg=self.max_turn_deg,
        )


def scale_weights(weights: CostWeights, **overrides: float) -> CostWeights:
    return replace(weights, **overrides)
