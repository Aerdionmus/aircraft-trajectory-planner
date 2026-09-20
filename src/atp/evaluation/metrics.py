"""Trajectory evaluation.

Metrics are recomputed from the state sequence rather than accumulated during
search.  That is intentional: it means a path produced by A*, by Dijkstra, or by
the direct-route baseline is scored by exactly the same code, and a bug in the
search cannot flatter its own results.

Infeasible trajectories
-----------------------
A segment that violates a hard constraint has no defined cost under this model,
so it contributes nothing to the totals.  The consequence matters for
experiments: the distance/time/fuel/cost totals of an infeasible trajectory
cover only its **feasible** segments and are therefore *not* comparable with
those of a feasible one.  ``unpriced_segments`` records how many were dropped
and :attr:`TrajectoryEvaluation.comparable_cost` is ``inf`` unless the whole
trajectory is feasible.  Comparisons between planners must use
``comparable_cost``, never ``cost.total``.

Speed (Milestone 3)
-------------------
Heading is *recomputed* here from the cell sequence and the planner's heading
index is ignored, because heading is a geometric consequence of the route.
**Speed is not.**  It is a decision, and a decision is part of the trajectory in
exactly the way the cell sequence is -- two trajectories through the same cells
at different speeds are different trajectories with different fuel and different
turn geometry.  So the speed schedule is *read off* the trajectory
(``FlightState.isp`` on each state, i.e. the speed flown on the leg that arrived
at it) rather than reconstructed, and any state that carries none -- every
baseline, and every Milestone 1/2 path -- is scored at the envelope's cruise
speed, the reference operating point.

This does not weaken the property that A*, Dijkstra and both baselines are
scored by identical code: the same function, with the same cost model, prices
whatever (cells, speeds) pair it is handed.  What it does mean is that a
baseline is a *fixed-speed* reference even when the planner had a choice, which
is the honest comparison -- the baseline has no mechanism for choosing.

Turns
-----
A corner is a property of three consecutive states, not two, so turn effects are
recomputed here from consecutive *triples* of the projected cell sequence --
never read back from the planner.  The heading index in a planner state is
ignored entirely.  That keeps the Milestone 1 property that A*, Dijkstra and
every baseline are scored by identical code, and it means the rasterised
direct-route baseline is charged for its own staircase corners rather than
being quietly exempted.

Two distinct quantities are reported, and they must not be conflated:

``geometric_corners``
    How many direction changes the returned ground track actually contains.
    Computed from the cell sequence alone, so it is **independent of whether a
    turn model is active**.  A Milestone 1 trajectory flown with
    ``turn_model="none"`` still has corners, and this counts them.

``num_turns``
    How many corners the *active turn model* evaluated and charged.  It is zero
    by construction when ``turn_model="none"``, because no turn was modelled --
    which is a statement about the model, not about the trajectory's shape.

Before this distinction existed the turn-ablation experiment reported
``num_turns = 0`` for the ``none`` arm and invited the reader to conclude the
Milestone 1 route was straight, when in fact it contained the same corners as
the ``gate`` arm.  Reporting both makes the ablation interpretable: ``gate``
prunes corners the aircraft could not fly, and ``gate+cost`` additionally prices
the ones it can, so the geometric count is what actually moves between arms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..core.geometry import Vec2
from ..core.units import hours_to_minutes
from ..environment.airspace import GridState
from ..planning.astar import SearchStatistics
from ..planning.cost import NO_TURN, CostBreakdown, CostModel, SegmentMetrics
from ..planning.state import cell_of


@dataclass
class TrajectoryEvaluation:
    feasible: bool
    num_waypoints: int = 0
    distance_nm: float = 0.0
    time_h: float = 0.0
    fuel_kg: float = 0.0
    risk_exposure: float = 0.0
    max_segment_mean_risk_density: float = 0.0
    soft_penalty: float = 0.0
    level_changes: int = 0
    total_climb_ft: float = 0.0
    #: Direction changes present in the returned ground track, counted from the
    #: cell sequence and therefore independent of the active turn model.
    geometric_corners: int = 0
    #: Corners the active turn model evaluated and charged.  Zero whenever
    #: ``turn_model="none"``, since no turn was modelled.  See the module
    #: docstring: this is not the same quantity as ``geometric_corners``.
    num_turns: int = 0
    total_heading_change_deg: float = 0.0
    max_heading_change_deg: float = 0.0
    total_track_change_deg: float = 0.0
    wind_heading_excess_deg: float = 0.0
    turn_time_min: float = 0.0
    turn_fuel_kg: float = 0.0
    corner_cut_nm: float = 0.0
    #: Segments dropped from the totals because they violate a hard constraint.
    unpriced_segments: int = 0
    mean_ground_speed_kt: float = 0.0
    #: Distance-weighted mean of the TAS actually flown.
    mean_tas_kt: float = 0.0
    min_tas_kt: float = 0.0
    max_tas_kt: float = 0.0
    #: Number of segment boundaries at which the selected speed changed,
    #: counting **every** consecutive pair of segments -- horizontal-to-
    #: horizontal, horizontal-to-level, or level-to-horizontal alike -- not
    #: only transitions between two horizontal legs.  A speed change made
    #: entirely *during* a pure level segment, with the horizontal legs on
    #: either side of it left unchanged, still counts: the aircraft genuinely
    #: changed speed once, at that node, and this field is meant to reflect
    #: that regardless of which kind of segment the change happened on.  Like
    #: ``geometric_corners`` this describes the shape of the returned
    #: trajectory, so it counts the whole sequence rather than only the priced
    #: part.  It is the count of instantaneous speed changes the model grants
    #: for free; see the limitation recorded in :mod:`atp.planning.problem`.
    speed_changes: int = 0
    cost: CostBreakdown = field(default_factory=CostBreakdown)
    hard_violations: list[str] = field(default_factory=list)
    infeasible_reason: str = ""

    @property
    def time_min(self) -> float:
        return hours_to_minutes(self.time_h)

    @property
    def comparable_cost(self) -> float:
        """Cost usable for planner comparison: ``inf`` if any segment is
        infeasible, because the priced total then covers only part of the
        trajectory."""
        return self.cost.total if self.feasible else math.inf

    def as_dict(self) -> dict[str, object]:
        return {
            "feasible": self.feasible,
            "num_waypoints": self.num_waypoints,
            "distance_nm": self.distance_nm,
            "time_h": self.time_h,
            "time_min": self.time_min,
            "fuel_kg": self.fuel_kg,
            "risk_exposure": self.risk_exposure,
            "max_segment_mean_risk_density": self.max_segment_mean_risk_density,
            "unpriced_segments": self.unpriced_segments,
            "comparable_cost": self.comparable_cost,
            "soft_penalty": self.soft_penalty,
            "level_changes": self.level_changes,
            "total_climb_ft": self.total_climb_ft,
            "geometric_corners": self.geometric_corners,
            "num_turns": self.num_turns,
            "total_heading_change_deg": self.total_heading_change_deg,
            "max_heading_change_deg": self.max_heading_change_deg,
            "total_track_change_deg": self.total_track_change_deg,
            "wind_heading_excess_deg": self.wind_heading_excess_deg,
            "turn_time_min": self.turn_time_min,
            "turn_fuel_kg": self.turn_fuel_kg,
            "corner_cut_nm": self.corner_cut_nm,
            "mean_ground_speed_kt": self.mean_ground_speed_kt,
            "mean_tas_kt": self.mean_tas_kt,
            "min_tas_kt": self.min_tas_kt,
            "max_tas_kt": self.max_tas_kt,
            "speed_changes": self.speed_changes,
            "hard_violations": list(self.hard_violations),
            "infeasible_reason": self.infeasible_reason,
            **{f"cost_{k}": v for k, v in self.cost.as_dict().items()},
        }


def _move_of(a: GridState, b: GridState) -> tuple[int, int] | None:
    """Horizontal move between two consecutive states, or ``None`` for a pure
    level change (which has no ground track and therefore no corner)."""
    move = (b.ix - a.ix, b.iy - a.iy)
    return None if move == (0, 0) else move


def count_geometric_corners(
    cells: list[GridState], start_move: tuple[int, int] | None = None
) -> int:
    """Direction changes in a projected cell sequence.

    Purely a property of the ground track: two consecutive horizontal moves that
    differ form a corner.  Nothing here consults the cost model, the turn model
    or the aircraft, so the count is the same whether or not turns are being
    gated and priced -- which is the whole point of reporting it separately from
    :attr:`TrajectoryEvaluation.num_turns`.

    Pure level changes contribute no ground track and therefore no corner; the
    heading is carried across them, matching :meth:`CostModel.turn_metrics`.
    ``start_move`` is the declared departure track, if the scenario has one, so
    that a first leg which departs from it counts as a corner exactly as the
    turn model would charge it.

    Unlike the priced metrics this counts the whole returned sequence, including
    segments that violate a hard constraint: it describes the shape of the
    trajectory that came back, not the part of it that could be priced.
    """
    corners = 0
    previous = start_move
    for a, b in zip(cells, cells[1:]):
        move = _move_of(a, b)
        if move is None:
            continue
        if previous is not None and move != previous:
            corners += 1
        previous = move
    return corners


def speed_schedule_of(
    cost_model: CostModel, path: list
) -> list[int | None]:
    """The speed flown on each leg of ``path``, one entry per segment.

    ``None`` means "no speed was selected", which is what every Milestone 1/2
    call site and every baseline produces; the cost model then evaluates at the
    default commanded TAS and performs no envelope check, reproducing the
    earlier behaviour exactly.

    With a multi-speed envelope the schedule is read from ``FlightState.isp``,
    the speed flown on the leg arriving at that state.  A state that carries no
    speed index is scored at the envelope's cruise index.
    """
    if not cost_model.models_speed:
        return [None] * max(0, len(path) - 1)
    default = cost_model.default_speed_index
    return [
        getattr(state, "isp", default) for state in path[1:]
    ]


def evaluate_trajectory(
    cost_model: CostModel,
    path: list,
    *,
    start_move: tuple[int, int] | None = None,
    start_speed_index: int | None = None,
    speed_indices: list[int | None] | None = None,
) -> TrajectoryEvaluation:
    """Re-evaluate a state sequence against the cost model.

    ``path`` may hold :class:`~atp.environment.airspace.GridState` or
    :class:`~atp.planning.state.FlightState` values; either way only the
    projection ``pi`` is used.  ``start_move`` is the departure ground track, if
    the scenario declares one; without it the first leg has no preceding corner.

    Turns are evaluated only when the cost model declares a turn model, so a
    Milestone 1 call with two positional arguments is unchanged.

    ``speed_indices`` overrides the schedule that would otherwise be read from
    the path (see :func:`speed_schedule_of`); it exists so that a controlled
    experiment can re-price an existing route at a different fixed speed.
    ``start_speed_index`` is the speed of the notional leg arriving at the start,
    used only to evaluate the departure corner when ``start_move`` is given.
    """
    cells = [cell_of(s) for s in path]
    if len(cells) < 2:
        return TrajectoryEvaluation(
            feasible=len(cells) == 1,
            num_waypoints=len(cells),
            infeasible_reason="" if cells else "empty path",
        )

    cell_size = cost_model.airspace.spec.cell_size_nm
    total = SegmentMetrics(feasible=True)
    breakdown = CostBreakdown()
    violations: list[str] = []
    level_changes = 0
    climb_ft = 0.0
    reason = ""
    previous_move = start_move
    schedule = (
        speed_schedule_of(cost_model, path) if speed_indices is None else speed_indices
    )
    if len(schedule) != len(cells) - 1:
        raise ValueError(
            f"speed schedule has {len(schedule)} entries for "
            f"{len(cells) - 1} segment(s)"
        )
    if start_speed_index is None and cost_model.models_speed:
        start_speed_index = cost_model.default_speed_index
    previous_speed = start_speed_index
    speed_changes = 0

    for index, (a, b) in enumerate(zip(cells, cells[1:])):
        move = _move_of(a, b)
        speed = schedule[index]
        turn = NO_TURN
        if cost_model.models_turns and move is not None and previous_move is not None:
            turn = cost_model.turn_metrics(
                a,
                Vec2(float(previous_move[0]), float(previous_move[1])).normalized(),
                Vec2(float(move[0]), float(move[1])).normalized(),
                math.hypot(*previous_move) * cell_size,
                math.hypot(*move) * cell_size,
                speed_index_in=previous_speed,
                speed_index_out=speed,
            )
        metrics = cost_model.with_turn(cost_model.evaluate(a, b, speed), turn)
        # `speed_changes` counts every actual change in the selected speed
        # between one segment and the next -- horizontal or pure level, either
        # side -- because that is what the milestone's own abstraction grants
        # for free (see `planning/problem.py`'s "Speed transitions" section):
        # an instantaneous, unpriced speed change at *any* node, not only at
        # nodes joining two horizontal legs. Gating this on `move is not None`
        # (as the turn-eligibility check above still correctly does) would
        # miss exactly the case where the change happens *during* a pure level
        # segment and the horizontal legs on either side of it happen to match
        # -- e.g. 450 kt horizontal, a level change selecting 330 kt, then a
        # further horizontal leg still at 330 kt: one real speed change, at
        # the level transition, invisible to a horizontal-to-horizontal
        # comparison because the two horizontal legs never differ from each
        # other. `index > 0` excludes only the very first segment, since
        # `previous_speed` there is the notional speed of the start's
        # incoming leg (used solely to price the departure corner), not a
        # segment that was actually flown -- there is nothing for the first
        # segment to have "changed" from.
        if (
            index > 0
            and previous_speed is not None
            and speed is not None
            and speed != previous_speed
        ):
            speed_changes += 1
        if move is not None:
            previous_move = move
        # `previous_speed` tracks the speed carried by the most recent state,
        # so it updates on *every* segment -- including a pure level change,
        # which has no track of its own but does update `FlightState.isp` in
        # the planner (see `_successors_with_flight_state`'s pure-level-change
        # branch). Gating this update on `move is not None`, as `previous_move`
        # is gated, would leave a level-change segment's speed selection
        # invisible to the next horizontal leg's turn evaluation: the corner
        # would be priced at the speed flown *before* the level change instead
        # of the speed the aircraft actually carries into it, disagreeing with
        # the planner's own turn_metrics call for the identical transition.
        if speed is not None:
            previous_speed = speed
        if not metrics.feasible:
            violations.append(f"{a}->{b}: {metrics.infeasible_reason}")
            reason = reason or metrics.infeasible_reason
            continue
        total = total + metrics
        breakdown = breakdown + cost_model.price(metrics)
        if a.il != b.il:
            level_changes += 1
            alt_delta = cost_model.airspace.spec.altitude_ft(
                b.il
            ) - cost_model.airspace.spec.altitude_ft(a.il)
            if alt_delta > 0:
                climb_ft += alt_delta

    heading_deg = math.degrees(total.heading_change_rad)
    track_deg = math.degrees(total.track_change_rad)
    return TrajectoryEvaluation(
        feasible=not violations,
        num_waypoints=len(cells),
        distance_nm=total.ground_distance_nm,
        time_h=total.time_h,
        fuel_kg=total.fuel_kg,
        risk_exposure=total.risk_exposure,
        max_segment_mean_risk_density=total.mean_risk_density,
        unpriced_segments=len(violations),
        soft_penalty=total.restriction_penalty,
        level_changes=level_changes,
        total_climb_ft=climb_ft,
        geometric_corners=count_geometric_corners(cells, start_move),
        num_turns=total.num_turns,
        total_heading_change_deg=heading_deg,
        max_heading_change_deg=math.degrees(total.max_heading_change_rad),
        total_track_change_deg=track_deg,
        wind_heading_excess_deg=heading_deg - track_deg,
        turn_time_min=hours_to_minutes(total.turn_time_h),
        turn_fuel_kg=total.turn_fuel_kg,
        corner_cut_nm=total.corner_cut_nm,
        mean_ground_speed_kt=total.ground_speed_kt,
        mean_tas_kt=total.tas_kt,
        min_tas_kt=total.min_tas_kt if math.isfinite(total.min_tas_kt) else 0.0,
        max_tas_kt=total.max_tas_kt if math.isfinite(total.max_tas_kt) else 0.0,
        speed_changes=speed_changes,
        cost=breakdown,
        hard_violations=violations,
        infeasible_reason=reason,
    )


@dataclass
class PlanReport:
    """One planner run: what was asked, what came back, how hard it was."""

    scenario: str
    planner: str
    heuristic: str
    turn_model: str
    status: str
    search_cost: float
    evaluation: TrajectoryEvaluation
    statistics: SearchStatistics
    heuristic_info: dict[str, object] = field(default_factory=dict)
    #: The planned state sequence. Kept out of :meth:`as_row` because tabular
    #: results files should stay one row per run.
    path: list[GridState] = field(default_factory=list)

    def as_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "scenario": self.scenario,
            "planner": self.planner,
            "heuristic": self.heuristic,
            "turn_model": self.turn_model,
            "status": self.status,
            "search_cost": self.search_cost,
        }
        row.update(self.evaluation.as_dict())
        row.update(self.statistics.as_dict())
        row.update(
            {f"heuristic_{k}": v for k, v in self.heuristic_info.items() if k != "name"}
        )
        return row
