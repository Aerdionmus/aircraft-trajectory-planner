"""Search problem interfaces.

:class:`SearchProblem` is deliberately domain-free.  The A* implementation in
:mod:`atp.planning.astar` is written against this protocol alone, so the search
can be unit tested on a five-node hand-built graph with no airspace, aircraft or
geometry involved.  That separation is the main testability property of the
design.

:class:`TrajectoryPlanningProblem` is the aerospace instantiation.

Milestone 2 adds an optional heading dimension and Milestone 3 an optional speed
dimension.  With ``turn_model="none"`` *and* a single-speed envelope the problem
behaves exactly as in Milestone 1: states are
:class:`~atp.environment.airspace.GridState` triples and the successor set and
its order are untouched.  As soon as either dimension is active, states become
:class:`~atp.planning.state.FlightState` tuples; illegal turns are pruned, and
the corner flown at the current node is charged onto the outgoing transition --
which keeps the cost a pure function of ``(s, s')`` as :class:`SearchProblem`
requires.

Speed transitions (Milestone 3)
-------------------------------
**Speed is selected per segment and may only change at a node.**  Each outgoing
transition names one speed option from the envelope; that speed sets the segment
time, its fuel flow, and (with the faster of the two adjacent speeds) the turn
geometry at the node it departs from.  Charging the corner at the faster of the
two adjacent speeds is conservative *with respect to the modelled turn radius
and turn rate at that discrete speed pair* -- it is not, and is not claimed to
be, a physically validated bound on the unmodelled acceleration or deceleration
manoeuvre a real aircraft would fly between the two speeds; see the limitation
below for that manoeuvre's own status.

**[LIMITATION]** The change itself is *instantaneous and unpriced*: no
acceleration or deceleration dynamics are modelled, no time or fuel is charged
for the speed change, and no limit is placed on how far the speed may jump
between adjacent segments.  This is the simplest abstraction that still makes
speed a genuine decision variable, and it is chosen deliberately over a
half-modelled alternative -- a bounded index step would look more physical while
still not being a longitudinal-dynamics model, and would need its own
justification for the bound.  The abstraction is optimistic: a real aircraft
needs time and distance to change speed, so a trajectory containing speed
changes is flown slightly later and slightly less efficiently than reported.
The number of speed changes is counted and reported
(``TrajectoryEvaluation.speed_changes``) so the size of the abstraction is
measurable per trajectory rather than unknown; ``docs/milestone3_results.md``
quantifies it for the shipped experiments.  Nothing here is a flight-control law
or an autothrottle model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Hashable, Iterable, Protocol, TypeVar

from ..core.geometry import Vec2
from ..environment.airspace import Airspace, GridState
from .cost import NO_TURN, CostModel, SegmentMetrics
from .state import NO_HEADING, FlightState, TurnTable, cell_of

S = TypeVar("S", bound=Hashable)


@dataclass(frozen=True)
class Transition(Generic[S]):
    """One directed edge produced by :meth:`SearchProblem.successors`."""

    state: S
    cost: float
    metrics: SegmentMetrics | None = None


class SearchProblem(Protocol[S]):
    """Minimal interface consumed by the search algorithms."""

    def initial_state(self) -> S: ...

    def is_goal(self, state: S) -> bool: ...

    def successors(self, state: S) -> Iterable[Transition[S]]: ...


@dataclass(frozen=True)
class GoalSpec:
    """Goal definition.

    ``match_level`` controls whether the goal flight level must be reached.  With
    ``match_level=False`` the planner is free to arrive at any level, which is
    the right default for an en-route cruise study.

    ``heading_index`` optionally requires a particular arrival ground track (an
    index into ``GridSpec.moves``).  It is ignored unless the problem is
    heading-aware.  Constraining it can only raise the true cost-to-go, so the
    inherited-admissibility argument is unaffected.
    """

    state: GridState
    match_level: bool = False
    heading_index: int | None = None


class TrajectoryPlanningProblem:
    """Grid trajectory planning as a shortest-path problem.

    Successors are generated in a fixed order (horizontal moves in the order
    declared by the grid connectivity, each at the same level, then level-change
    variants, then pure level changes).  Fixed ordering plus deterministic
    tie-breaking in the search is what makes runs reproducible.
    """

    def __init__(
        self,
        airspace: Airspace,
        cost_model: CostModel,
        start: GridState,
        goal: GoalSpec,
        *,
        allow_level_change_with_move: bool = True,
        allow_pure_level_change: bool = False,
        max_level_step: int = 1,
        start_heading_index: int | None = None,
        start_speed_index: int | None = None,
    ) -> None:
        if not airspace.in_bounds(start):
            raise ValueError(f"start state {start} is outside the airspace")
        if not airspace.in_bounds(goal.state):
            raise ValueError(f"goal state {goal.state} is outside the airspace")
        if max_level_step < 1:
            raise ValueError("max_level_step must be >= 1")
        self.airspace = airspace
        self.cost_model = cost_model
        self.start = start
        self.goal = goal
        self.allow_level_change_with_move = allow_level_change_with_move
        self.allow_pure_level_change = allow_pure_level_change
        self.max_level_step = max_level_step
        self.turn_table = TurnTable(airspace.spec.moves, airspace.spec.cell_size_nm)
        num_moves = len(self.turn_table)
        if start_heading_index is not None and not 0 <= start_heading_index < num_moves:
            raise ValueError(
                f"start_heading_index {start_heading_index} is not a valid move index"
            )
        if goal.heading_index is not None and not 0 <= goal.heading_index < num_moves:
            raise ValueError(
                f"goal heading_index {goal.heading_index} is not a valid move index"
            )
        self.start_heading_index = start_heading_index
        num_speeds = cost_model.envelope.num_options
        if start_speed_index is not None and not 0 <= start_speed_index < num_speeds:
            raise ValueError(
                f"start_speed_index {start_speed_index} is not a valid speed "
                f"option (the envelope has {num_speeds})"
            )
        #: Speed flown on the notional leg arriving at the start.  It is only
        #: ever read to recover an incoming air heading, so it matters solely
        #: when a departure heading is declared; otherwise the start has no
        #: corner and the value is inert.  Defaults to the cruise speed, the
        #: reference operating point.
        self.start_speed_index = (
            cost_model.default_speed_index
            if start_speed_index is None
            else start_speed_index
        )
        if (
            cost_model.models_speed
            and cost_model.models_turns
            and self.start_speed_index
            not in cost_model.available_speed_indices(start.il)
        ):
            # The start speed is used to recover the air heading of the notional
            # incoming leg, so with a departure heading declared it really is
            # flown.  Letting it sit outside the operating limits would mean the
            # departure corner was computed from a speed the aircraft may not
            # use -- a silent substitution of exactly the kind the envelope
            # exists to prevent.  Refused explicitly rather than clamped.
            raise ValueError(
                f"start speed option {self.start_speed_index} "
                f"({cost_model.speed_tas_kt(self.start_speed_index):.0f} kt) is "
                f"outside the operating limits at the start level; available "
                f"options there are "
                f"{list(cost_model.available_speed_indices(start.il))}"
            )
        self._level_offsets = self._build_level_offsets()

    @property
    def heading_aware(self) -> bool:
        return self.cost_model.models_turns

    @property
    def speed_aware(self) -> bool:
        """True when the envelope offers a real choice of speed."""
        return self.cost_model.models_speed

    @property
    def uses_flight_state(self) -> bool:
        """True when the state carries a heading, a speed, or both.

        False means the Milestone 1 ``GridState`` successor generator runs
        untouched, which is what makes backward parity structural.
        """
        return self.heading_aware or self.speed_aware

    @property
    def start_move(self) -> tuple[int, int] | None:
        """The declared departure move, or ``None`` for a free start heading."""
        if not self.heading_aware or self.start_heading_index is None:
            return None
        return self.turn_table.moves[self.start_heading_index]

    def _build_level_offsets(self) -> tuple[int, ...]:
        if not self.allow_level_change_with_move:
            return (0,)
        offsets = [0]
        for step in range(1, self.max_level_step + 1):
            offsets.extend((step, -step))
        return tuple(offsets)

    # -- SearchProblem -------------------------------------------------------
    def initial_state(self):
        if not self.uses_flight_state:
            return self.start
        return FlightState(
            self.start.ix,
            self.start.iy,
            self.start.il,
            self.start_heading_index
            if (self.heading_aware and self.start_heading_index is not None)
            else NO_HEADING,
            self.start_speed_index if self.speed_aware else 0,
        )

    def is_goal(self, state) -> bool:
        if state.ix != self.goal.state.ix or state.iy != self.goal.state.iy:
            return False
        if self.goal.match_level and state.il != self.goal.state.il:
            return False
        if self.heading_aware and self.goal.heading_index is not None:
            return getattr(state, "ih", NO_HEADING) == self.goal.heading_index
        return True

    def successors(self, state) -> Iterable[Transition]:
        if self.uses_flight_state:
            yield from self._successors_with_flight_state(state)
        else:
            yield from self._successors_m1(state)

    # -- Milestone 1 successor generation, untouched -------------------------
    def _successors_m1(self, state: GridState) -> Iterable[Transition[GridState]]:
        spec = self.airspace.spec
        for dx, dy in spec.moves:
            nx, ny = state.ix + dx, state.iy + dy
            if not (0 <= nx < spec.cells_x and 0 <= ny < spec.cells_y):
                continue
            for dl in self._level_offsets:
                nl = state.il + dl
                if not (0 <= nl < spec.num_levels):
                    continue
                candidate = GridState(nx, ny, nl)
                if self.airspace.is_blocked(candidate):
                    continue
                cost, metrics = self.cost_model.transition_cost(state, candidate)
                if metrics.feasible:
                    yield Transition(candidate, cost, metrics)

        if self.allow_pure_level_change:
            for dl in (1, -1):
                nl = state.il + dl
                if not (0 <= nl < spec.num_levels):
                    continue
                candidate = GridState(state.ix, state.iy, nl)
                if self.airspace.is_blocked(candidate):
                    continue
                cost, metrics = self.cost_model.transition_cost(state, candidate)
                if metrics.feasible:
                    yield Transition(candidate, cost, metrics)

    # -- Milestone 2/3 successor generation ----------------------------------
    def _successors_with_flight_state(
        self, state: FlightState
    ) -> Iterable[Transition[FlightState]]:
        """Successors over the heading- and/or speed-augmented state.

        Order is moves outer, then **speed options**, then level offsets, with
        pure level changes last.  Illegal turns and speeds outside the operating
        limits are simply skipped, so the accepted successors are always a
        subsequence of the Milestone 1 ones once projected.

        With a single-speed envelope the speed loop has exactly one iteration
        and the ordering collapses to Milestone 2's moves-then-levels, which is
        why fixed-speed parity holds expansion for expansion and not merely in
        final cost.

        The corner is evaluated once per ``(move, outgoing speed)`` pair rather
        than once per successor: it depends on the node, the two ground tracks
        and the two speeds, but not on the level offset.
        """
        spec = self.airspace.spec
        table = self.turn_table
        node = cell_of(state)
        heading_aware = self.heading_aware
        speed_aware = self.speed_aware

        in_index = state.ih if heading_aware else NO_HEADING
        in_unit = table.track_unit(in_index) if in_index != NO_HEADING else None
        leg_in = table.length_nm[in_index] if in_index != NO_HEADING else 0.0
        speed_in = state.isp if speed_aware else None
        # Speeds legal at the node's own level; the transition's other endpoint
        # level is checked inside `CostModel.evaluate`, which is what makes a
        # level change to an altitude the speed is not legal at infeasible
        # rather than silently reinterpreted.
        speed_options: tuple[int | None, ...] = (
            self.cost_model.available_speed_indices(state.il)
            if speed_aware
            else (None,)
        )

        for mi, (dx, dy) in enumerate(spec.moves):
            nx, ny = state.ix + dx, state.iy + dy
            if not (0 <= nx < spec.cells_x and 0 <= ny < spec.cells_y):
                continue
            out_unit = table.track_unit(mi)
            leg_out = table.length_nm[mi]
            for speed_out in speed_options:
                turn = self.cost_model.turn_metrics(
                    node,
                    in_unit,
                    out_unit,
                    leg_in,
                    leg_out,
                    speed_index_in=speed_in,
                    speed_index_out=speed_out,
                )
                if not turn.feasible:
                    continue
                for dl in self._level_offsets:
                    nl = state.il + dl
                    if not (0 <= nl < spec.num_levels):
                        continue
                    candidate = GridState(nx, ny, nl)
                    if self.airspace.is_blocked(candidate):
                        continue
                    cost, metrics = self.cost_model.transition_cost_with_turn(
                        node, candidate, turn, speed_out
                    )
                    if metrics.feasible:
                        yield Transition(
                            FlightState(
                                nx,
                                ny,
                                nl,
                                mi if heading_aware else NO_HEADING,
                                speed_out if speed_out is not None else 0,
                            ),
                            cost,
                            metrics,
                        )

        if self.allow_pure_level_change:
            for dl in (1, -1):
                nl = state.il + dl
                if not (0 <= nl < spec.num_levels):
                    continue
                candidate = GridState(state.ix, state.iy, nl)
                if self.airspace.is_blocked(candidate):
                    continue
                for speed_out in speed_options:
                    # No horizontal displacement, so no corner: the heading is
                    # carried through unchanged and nothing is charged.  The
                    # speed may still change, and it still sets the fuel flow.
                    cost, metrics = self.cost_model.transition_cost_with_turn(
                        node, candidate, NO_TURN, speed_out
                    )
                    if metrics.feasible:
                        yield Transition(
                            FlightState(
                                state.ix,
                                state.iy,
                                nl,
                                in_index,
                                speed_out if speed_out is not None else 0,
                            ),
                            cost,
                            metrics,
                        )

    # -- convenience ---------------------------------------------------------
    def position_nm(self, state) -> Vec2:
        return self.airspace.centre_nm(cell_of(state))

    def goal_position_nm(self) -> Vec2:
        return self.airspace.centre_nm(self.goal.state)

    def start_is_valid(self) -> bool:
        return not self.airspace.is_blocked(self.start)

    def goal_is_valid(self) -> bool:
        return not self.airspace.is_blocked(self.goal.state)
