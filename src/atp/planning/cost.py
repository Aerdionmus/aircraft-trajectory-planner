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
4. **Monotone refinement of the turn model.**  No turn modelling may lower the
   cost of a transition relative to its no-turn value *at the same speed*.
   Turn modelling may only add cost (time, fuel, risk) or remove edges; it
   never adds or subtracts distance, and in particular the geometric shortening
   a fly-by arc produces is measured (``corner_cut_nm``) but never credited.
   It is checked directly by ``tests/test_turn_cost.py``.

   **This invariant is scoped to the turn model and does NOT extend across the
   speed decision.**  Milestone 2 could state it as "no Milestone 2 edge is
   cheaper than its Milestone 1 value" and inherit admissibility from it.  That
   statement is *false* in Milestone 3 and must not be carried over: an edge
   flown at a faster selectable speed can legitimately cost less than the same
   edge at the old fixed cruise speed, because it takes less time.  Making
   speed a decision variable is precisely the act of allowing cheaper edges.

   Milestone 3 therefore does **not** inherit admissibility.  The lower bound in
   :meth:`CostModel.speed_cost_lower_bound_per_nm` is re-derived directly over
   the whole speed envelope, and the property that replaces monotone refinement
   as the regression guard is **fixed-speed parity**: a singleton envelope
   containing only the Milestone 2 cruise speed must reproduce Milestone 2
   exactly.  See ``docs/milestone3_results.md`` and
   ``tests/test_m2_fixed_speed_parity.py``.

Speed (Milestone 3)
-------------------
Every method that evaluates a transition or a corner takes an optional speed
index into the aircraft's :class:`~atp.aircraft.envelope.SpeedEnvelope`.
``None`` means "the aircraft's default commanded TAS", which is what Milestone 1
and 2 call sites pass implicitly and what reproduces their numbers exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ..aircraft.envelope import SpeedEnvelope
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
    #: True airspeed the segment was flown at.  On an accumulated total this
    #: becomes the distance-weighted mean, matching ``ground_speed_kt``.
    tas_kt: float = 0.0
    #: Extremes of the selected TAS along an accumulated trajectory.  ``inf`` /
    #: ``-inf`` are the identities for a trajectory with no priced segment, so
    #: that an empty accumulation does not claim a speed of zero.
    min_tas_kt: float = math.inf
    max_tas_kt: float = -math.inf
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
            tas = (
                self.tas_kt * self.ground_distance_nm
                + other.tas_kt * other.ground_distance_nm
            ) / total_d
        else:
            gs = 0.0
            tas = 0.0
        return SegmentMetrics(
            feasible=self.feasible and other.feasible,
            ground_distance_nm=total_d,
            vertical_ft=self.vertical_ft + other.vertical_ft,
            time_h=self.time_h + other.time_h,
            fuel_kg=self.fuel_kg + other.fuel_kg,
            risk_exposure=self.risk_exposure + other.risk_exposure,
            restriction_penalty=self.restriction_penalty + other.restriction_penalty,
            ground_speed_kt=gs,
            tas_kt=tas,
            min_tas_kt=min(self.min_tas_kt, other.min_tas_kt),
            max_tas_kt=max(self.max_tas_kt, other.max_tas_kt),
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
        self._available_by_level: dict[int, tuple[int, ...]] = {}

    @property
    def models_turns(self) -> bool:
        return self.turn_model != "none"

    # -- speed ---------------------------------------------------------------
    @property
    def envelope(self) -> SpeedEnvelope:
        return self.aircraft.envelope

    @property
    def models_speed(self) -> bool:
        """True when there is an actual speed decision to make.

        False for a singleton envelope, which is the default and which makes the
        Milestone 2 code path structurally identical rather than merely
        numerically close. This governs the *state representation* (whether
        speed becomes a branching dimension of the search) and is deliberately
        independent of :attr:`~atp.aircraft.envelope.SpeedEnvelope.has_limits`:
        a singleton envelope can still declare a CAS or Mach limit that makes
        its one speed infeasible somewhere, with no decision involved at all.
        See :meth:`_resolve_speed` for where that distinction is enforced.
        """
        return not self.envelope.is_fixed

    @property
    def default_speed_index(self) -> int:
        """Index of the cruise speed: the reference operating point used by the
        baselines and by any evaluation that names no speed."""
        return self.envelope.cruise_index

    def speed_tas_kt(self, speed_index: int) -> float:
        return self.envelope.tas_kt(speed_index)

    def available_speed_indices(self, level_index: int) -> tuple[int, ...]:
        """Speed options inside the operating limits at a flight level.

        Cached per level: the envelope's CAS/Mach limits are evaluated through
        the ISA atmosphere, which is far too expensive to redo per edge.
        """
        cached = self._available_by_level.get(level_index)
        if cached is None:
            cached = self.envelope.available_indices(
                self.airspace.spec.altitude_ft(level_index)
            )
            self._available_by_level[level_index] = cached
        return cached

    def _resolve_speed(
        self, speed_index: int | None, level_a: int, level_b: int
    ) -> tuple[float, str]:
        """``(tas_kt, infeasible_reason)`` for a transition between two levels.

        A selected speed must lie inside the operating limits at **both**
        endpoint levels.  Testing the endpoints rather than the mid-altitude is
        the conservative choice, consistent with the way a level-changing
        transition is already tested against the whole altitude band it spans.

        The check goes through :meth:`available_speed_indices`, which is cached
        per level, rather than re-evaluating the envelope's CAS and Mach limits
        directly.  Those limits are evaluated through the ISA atmosphere, and
        doing that once per transition rather than once per level dominated the
        search loop by an order of magnitude when it was first written.

        ``speed_index=None`` means "resolve the default speed" and is handled in
        two different ways depending on whether the envelope declares any
        operating limit at all -- **not** on whether it offers a decision.
        Those are different facts (:attr:`~atp.aircraft.envelope.SpeedEnvelope.has_limits`
        versus :attr:`~atp.aircraft.envelope.SpeedEnvelope.is_fixed`) and
        conflating them was a real gap: a singleton envelope that explicitly
        declares a restrictive CAS or Mach limit still has to enforce it, even
        though it offers nothing to choose between.

        - With no limit declared -- ``SpeedEnvelope.fixed(V)``, the default for
          every aircraft that does not declare an envelope, or any other
          envelope built with no CAS/Mach limits -- this is the **unchanged
          Milestone 1/2 behaviour**: a single commanded TAS with no envelope
          check at all, since an unlimited envelope has nothing a transition
          could fail to satisfy. Preserved bit-for-bit so fixed-speed parity
          stays exact.
        - With any limit declared -- whether the envelope offers one
          selectable speed or several -- ``None`` is resolved to
          :attr:`default_speed_index` (the envelope's cruise speed) and **then
          validated exactly like an explicit choice**. Leaving this
          unvalidated was a real hole: a caller that evaluates a transition
          without naming a speed -- as every pre-Milestone-3 call site still
          does, and as the whole planner still does whenever there is no
          decision to make -- would silently fly the cruise TAS even where the
          envelope's own CAS or Mach limit makes it illegal, rather than
          reporting the transition infeasible the way an explicit
          out-of-envelope choice already does.
        """
        if speed_index is None:
            if not self.envelope.has_limits:
                spec = self.airspace.spec
                mid_alt = 0.5 * (
                    spec.altitude_ft(level_a) + spec.altitude_ft(level_b)
                )
                return self.aircraft.tas_kt(mid_alt), ""
            speed_index = self.default_speed_index
        tas = self.envelope.tas_kt(speed_index)
        levels = (level_a,) if level_a == level_b else (level_a, level_b)
        for level in levels:
            if speed_index not in self.available_speed_indices(level):
                return tas, (
                    f"TAS {tas:.0f} kt is outside the speed envelope at "
                    f"{self.airspace.spec.altitude_ft(level):.0f} ft"
                )
        return tas, ""

    # -- physical evaluation -------------------------------------------------
    def evaluate(
        self, a: GridState, b: GridState, speed_index: int | None = None
    ) -> SegmentMetrics:
        """Physical evaluation of one transition, flown at a selected speed.

        ``speed_index=None`` means the aircraft's default commanded TAS and no
        envelope check, which is exactly the Milestone 1/2 behaviour.
        """
        spec = self.airspace.spec
        pa, pb = spec.centre_nm(a), spec.centre_nm(b)
        alt_a, alt_b = spec.altitude_ft(a.il), spec.altitude_ft(b.il)
        delta_alt = alt_b - alt_a

        if not self.aircraft.can_operate_at(max(alt_a, alt_b)):
            return SegmentMetrics.infeasible("above service ceiling")

        tas_kt, speed_reason = self._resolve_speed(speed_index, a.il, b.il)
        if speed_reason:
            return SegmentMetrics.infeasible(speed_reason)

        blocking = self.airspace.restrictions.blocking_region(pa, pb, alt_a, alt_b)
        if blocking is not None:
            return SegmentMetrics.infeasible(f"hard restriction {blocking.region_id}")

        delta = pb - pa
        horizontal_nm = delta.norm()

        if horizontal_nm < 1e-9:
            return self._pure_vertical(a, b, pa, alt_a, alt_b, delta_alt, tas_kt)

        track = delta.normalized()
        # Mid-segment wind sample: one evaluation per edge keeps the search loop
        # cheap; the error is O(h^2) in cell size for smooth fields.
        mid = pa + delta * 0.5
        mid_alt = 0.5 * (alt_a + alt_b)
        wind = self.airspace.wind.at(mid.x, mid.y, mid_alt)
        solution = solve_ground_speed(tas_kt, wind, track)
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
            pa,
            pb,
            alt_a,
            alt_b,
            horizontal_nm,
            delta_alt,
            time_h,
            solution.ground_speed_kt,
            tas_kt,
        )

    def _pure_vertical(
        self,
        a: GridState,
        b: GridState,
        pa: Vec2,
        alt_a: float,
        alt_b: float,
        delta_alt: float,
        tas_kt: float,
    ) -> SegmentMetrics:
        """Level change with no horizontal displacement.

        Charged at the maximum vertical rate.  Note the aircraft does in reality
        travel forward during the manoeuvre; that displacement is ignored, which
        makes pure level changes slightly optimistic in distance terms.

        The selected speed still sets the fuel flow -- a pure climb burns fuel at
        the rate the selected speed implies -- but it does not set the duration,
        which is fixed by the vertical rate limit.
        """
        if delta_alt == 0.0:
            return SegmentMetrics.infeasible("null transition")
        rate_fpm = self.aircraft.max_vertical_rate_fpm(delta_alt)
        time_h = abs(delta_alt) / (rate_fpm * MIN_PER_H)
        return self._finish(pa, pa, alt_a, alt_b, 0.0, delta_alt, time_h, 0.0, tas_kt)

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
        tas_kt: float | None = None,
    ) -> SegmentMetrics:
        mid_alt = 0.5 * (alt_a + alt_b)
        fuel = self.aircraft.fuel_flow_kg_per_h(mid_alt, tas_kt) * time_h
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

        flown_tas = self.aircraft.tas_kt(mid_alt) if tas_kt is None else tas_kt

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
            tas_kt=flown_tas,
            min_tas_kt=flown_tas,
            max_tas_kt=flown_tas,
        )

    # -- turns ---------------------------------------------------------------
    def turn_metrics(
        self,
        node: GridState,
        in_track_unit: Vec2 | None,
        out_track_unit: Vec2,
        leg_in_nm: float,
        leg_out_nm: float,
        speed_index_in: int | None = None,
        speed_index_out: int | None = None,
    ) -> TurnMetrics:
        """Evaluate the corner flown at ``node`` between two ground tracks.

        ``in_track_unit is None`` means there is no incoming leg (the start
        state, or a preceding pure level change): no corner, no charge, no gate.

        Speed (Milestone 3)
        -------------------
        The two legs may be flown at different selected speeds, and both matter:

        * the **air heading** of each leg is recovered from the wind triangle at
          that leg's own TAS, so the heading change the bank limit constrains is
          a function of both speeds, not just of the two ground tracks;
        * the **turn itself** is charged at ``max(V_in, V_out)``.  The corner is
          flown somewhere in the speed transition between the two legs and the
          model does not resolve where, so it is charged at the faster of the
          two, which is conservative in both directions at once: the turn radius
          grows with ``V^2`` (harder to fit, so the gate over-blocks rather than
          under-blocks) and the turn rate falls as ``1/V`` (longer, so the charge
          is an upper bound).  With a single selectable speed the two coincide
          and this reduces to the Milestone 2 expression exactly.

        ``speed_index_* = None`` means the default commanded TAS, i.e. Milestone
        2 behaviour.

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
        default_tas = self.aircraft.tas_kt(alt)
        tas_in = (
            default_tas
            if speed_index_in is None
            else self.envelope.tas_kt(speed_index_in)
        )
        tas_out = (
            default_tas
            if speed_index_out is None
            else self.envelope.tas_kt(speed_index_out)
        )
        # The corner is charged at the faster of the two legs: see the docstring.
        tas = max(tas_in, tas_out)
        wind = self.airspace.wind.at(p.x, p.y, alt)

        incoming = solve_ground_speed(tas_in, wind, in_track_unit)
        if not incoming.feasible:
            return TurnMetrics.infeasible(
                f"incoming track unflyable at node: {incoming.reason}"
            )
        outgoing = solve_ground_speed(tas_out, wind, out_track_unit)
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

        omega = self.aircraft.turn_rate_rad_per_h(alt, tas)
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
            self.aircraft.fuel_flow_kg_per_h(alt, tas)
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
        self,
        a: GridState,
        b: GridState,
        turn: TurnMetrics,
        speed_index: int | None = None,
    ) -> tuple[float, SegmentMetrics]:
        metrics = self.with_turn(self.evaluate(a, b, speed_index), turn)
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

    def transition_cost(
        self, a: GridState, b: GridState, speed_index: int | None = None
    ) -> tuple[float, SegmentMetrics]:
        metrics = self.evaluate(a, b, speed_index)
        if not metrics.feasible:
            return math.inf, metrics
        return self.price(metrics).total, metrics

    # -- heuristic support ---------------------------------------------------
    def speed_cost_lower_bound_per_nm(self) -> float:
        """Lower bound on the *speed-derived* cost per NM of **horizontal**
        progress, i.e. the time, fuel and risk terms only.

        Derivation (Milestone 3: re-derived, not inherited)
        ---------------------------------------------------
        Milestone 2 could bound this with a single ``GS_max`` because there was
        a single TAS.  With a speed envelope the planner may fly any selectable
        speed, so the bound must hold for *every* speed it could choose, which
        makes it a minimisation over the envelope rather than a single quotient.

        Consider any feasible transition, flown at some selectable speed ``V``
        at some level, with horizontal length ``H``.  The ground speed satisfies
        ``GS <= V + |w|_max``, so ``H / GS >= H / (V + |w|_max)``, and therefore

            time cost = c_time * H/GS            >= c_time * H / (V + |w|max)
            fuel cost = c_fuel * ff(V, alt) * H/GS
                                                 >= c_fuel * ff_min(V) * H / (V + |w|max)
            risk cost = c_risk * rho * H/GS      >= c_risk * rho_min * H / (V + |w|max)

        where ``ff_min(V)`` is the least fuel flow at speed ``V`` over the
        enumerated levels.  Summing and minimising over the speeds the envelope
        offers gives the returned constant

            A = min over selectable V of
                    (c_time + c_fuel * ff_min(V) + c_risk * rho_min) / (V + |w|max)

        and hence ``time/fuel/risk cost >= A * (horizontal path length)`` for any
        trajectory.  Climb fuel, turn time, turn fuel, turn risk and soft
        penalties are all dropped; every one of them is non-negative, so the
        bound only gets looser.

        **Why the minimisation is over speeds rather than over ``GS_max`` alone.**
        Taking ``ff_min`` over the whole envelope and dividing by the envelope's
        largest ``GS_max`` would also be sound, but needlessly loose: it pairs
        the fuel flow of the slowest option with the ground speed of the fastest,
        a combination no transition can realise.  Minimising the quotient keeps
        each speed's fuel flow with its own ground-speed bound.  Both forms are
        lower bounds; this one is tighter, and with a singleton envelope the two
        coincide, which is what makes fixed-speed parity exact.

        **Soundness of restricting to *selectable* speeds.**  The minimisation
        ranges over the speeds available at at least one level.  A transition can
        only be flown at a speed inside the operating limits at both of its
        endpoint levels, so every realisable speed is in that set and the minimum
        over it is a valid lower bound.  If the set is empty -- no speed is legal
        anywhere, so no transition exists at all -- the minimisation falls back
        to the whole envelope, which is a superset and therefore still sound.

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
        wind_max = self.airspace.wind.max_magnitude_kt()
        risk_min = max(0.0, self.airspace.risk.min_density())
        w = self.weights
        price_fuel = self._descent_credit_is_bounded()

        best = math.inf
        for index in self._bounding_speed_indices():
            tas = self.envelope.tas_kt(index)
            gs_max = max_possible_ground_speed_kt(tas, wind_max)
            if gs_max <= 0.0:
                return 0.0
            ff_min = (
                self.aircraft.min_fuel_flow_over_levels_kg_per_h(
                    spec.flight_levels, tas
                )
                if price_fuel
                else 0.0
            )
            candidate = (
                w.time_cost_per_hour
                + w.fuel_cost_per_kg * ff_min
                + w.risk_cost_per_exposure * risk_min
            ) / gs_max
            best = min(best, candidate)
        return 0.0 if not math.isfinite(best) else best

    def _bounding_speed_indices(self) -> tuple[int, ...]:
        """Speeds the lower bound must range over: those selectable at some
        level, falling back to the whole envelope if none is."""
        selectable: set[int] = set()
        for il in range(self.airspace.spec.num_levels):
            selectable.update(self.available_speed_indices(il))
        if not selectable:
            return self.envelope.indices()
        return tuple(sorted(selectable))

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
