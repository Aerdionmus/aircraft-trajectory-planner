"""Search problem interfaces.

:class:`SearchProblem` is deliberately domain-free.  The A* implementation in
:mod:`atp.planning.astar` is written against this protocol alone, so the search
can be unit tested on a five-node hand-built graph with no airspace, aircraft or
geometry involved.  That separation is the main testability property of the
design.

:class:`TrajectoryPlanningProblem` is the aerospace instantiation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Hashable, Iterable, Protocol, TypeVar

from ..core.geometry import Vec2
from ..environment.airspace import Airspace, GridState
from .cost import CostModel, SegmentMetrics

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
    """

    state: GridState
    match_level: bool = False


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
        self._level_offsets = self._build_level_offsets()

    def _build_level_offsets(self) -> tuple[int, ...]:
        if not self.allow_level_change_with_move:
            return (0,)
        offsets = [0]
        for step in range(1, self.max_level_step + 1):
            offsets.extend((step, -step))
        return tuple(offsets)

    # -- SearchProblem -------------------------------------------------------
    def initial_state(self) -> GridState:
        return self.start

    def is_goal(self, state: GridState) -> bool:
        if state.ix != self.goal.state.ix or state.iy != self.goal.state.iy:
            return False
        return (not self.goal.match_level) or state.il == self.goal.state.il

    def successors(self, state: GridState) -> Iterable[Transition[GridState]]:
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

    # -- convenience ---------------------------------------------------------
    def position_nm(self, state: GridState) -> Vec2:
        return self.airspace.centre_nm(state)

    def goal_position_nm(self) -> Vec2:
        return self.airspace.centre_nm(self.goal.state)

    def start_is_valid(self) -> bool:
        return not self.airspace.is_blocked(self.start)

    def goal_is_valid(self) -> bool:
        return not self.airspace.is_blocked(self.goal.state)
