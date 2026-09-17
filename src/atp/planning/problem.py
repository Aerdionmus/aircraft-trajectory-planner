"""Search problem interfaces.

:class:`SearchProblem` is deliberately domain-free.  The A* implementation in
:mod:`atp.planning.astar` is written against this protocol alone, so the search
can be unit tested on a five-node hand-built graph with no airspace, aircraft or
geometry involved.  That separation is the main testability property of the
design.

:class:`TrajectoryPlanningProblem` is the aerospace instantiation.

Milestone 2 adds an optional heading dimension.  With ``turn_model="none"`` the
problem behaves exactly as in Milestone 1: states are
:class:`~atp.environment.airspace.GridState` triples and the successor set and
its order are untouched.  With any other turn model states become
:class:`~atp.planning.state.FlightState` quadruples, illegal turns are pruned,
and the corner flown at the current node is charged onto the outgoing
transition -- which keeps the cost a pure function of ``(s, s')`` as
:class:`SearchProblem` requires.
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
        self._level_offsets = self._build_level_offsets()

    @property
    def heading_aware(self) -> bool:
        return self.cost_model.models_turns

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
        if not self.heading_aware:
            return self.start
        return FlightState(
            self.start.ix,
            self.start.iy,
            self.start.il,
            self.start_heading_index
            if self.start_heading_index is not None
            else NO_HEADING,
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
        if self.heading_aware:
            yield from self._successors_with_heading(state)
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

    # -- Milestone 2 successor generation ------------------------------------
    def _successors_with_heading(
        self, state: FlightState
    ) -> Iterable[Transition[FlightState]]:
        """Same order as Milestone 1 -- moves outer, level offsets inner, pure
        level changes last -- with illegal turns simply skipped.  The accepted
        successors are therefore always a subsequence of the Milestone 1 ones.
        """
        spec = self.airspace.spec
        table = self.turn_table
        node = cell_of(state)
        in_index = state.ih
        in_unit = table.track_unit(in_index) if in_index != NO_HEADING else None
        leg_in = table.length_nm[in_index] if in_index != NO_HEADING else 0.0

        for mi, (dx, dy) in enumerate(spec.moves):
            nx, ny = state.ix + dx, state.iy + dy
            if not (0 <= nx < spec.cells_x and 0 <= ny < spec.cells_y):
                continue
            # The corner is a property of the node and the two tracks, so it is
            # evaluated once per move rather than once per level offset.
            turn = self.cost_model.turn_metrics(
                node, in_unit, table.track_unit(mi), leg_in, table.length_nm[mi]
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
                    node, candidate, turn
                )
                if metrics.feasible:
                    yield Transition(FlightState(nx, ny, nl, mi), cost, metrics)

        if self.allow_pure_level_change:
            for dl in (1, -1):
                nl = state.il + dl
                if not (0 <= nl < spec.num_levels):
                    continue
                candidate = GridState(state.ix, state.iy, nl)
                if self.airspace.is_blocked(candidate):
                    continue
                # No horizontal displacement, so no corner: the heading is
                # carried through unchanged and nothing is charged.
                cost, metrics = self.cost_model.transition_cost_with_turn(
                    node, candidate, NO_TURN
                )
                if metrics.feasible:
                    yield Transition(
                        FlightState(state.ix, state.iy, nl, in_index), cost, metrics
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
