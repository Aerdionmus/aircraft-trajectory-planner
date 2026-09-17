"""A* search over an arbitrary :class:`~atp.planning.problem.SearchProblem`.

Implementation choices that matter
----------------------------------
*Determinism.*  Heap entries are ``(f, h, insertion_index, state_key)`` where the
insertion index is a strictly increasing counter.  Comparison therefore never
reaches the payload, and ties between equal-``f`` nodes are broken by preferring
the smaller ``h`` (goal-directed) and then by insertion order (FIFO).  Two runs
of the same problem expand the same nodes in the same order, on any platform.

*Lazy deletion.*  ``heapq`` has no decrease-key, so a node is pushed again when a
cheaper path to it is found and stale entries are discarded on pop.  This costs
memory proportional to the number of improvements rather than adding a priority
queue implementation.

*Re-opening.*  With a consistent heuristic a closed node never needs revisiting.
Because the package deliberately ships inadmissible heuristics for comparison,
re-opening is supported and counted; the counter being non-zero is direct
evidence that a heuristic is inconsistent.

*Termination.*  Goal test on *expansion*, not on generation.  Testing at
generation returns a suboptimal path whenever the heuristic is not perfect.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Generic, Hashable, TypeVar

from .problem import SearchProblem, Transition

S = TypeVar("S", bound=Hashable)


class SearchStatus(str, Enum):
    SOLVED = "solved"
    UNSOLVABLE = "unsolvable"
    LIMIT_REACHED = "limit_reached"


@dataclass
class SearchStatistics:
    expansions: int = 0
    generated: int = 0
    reopened: int = 0
    stale_pops: int = 0
    max_frontier: int = 0
    runtime_s: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "expansions": self.expansions,
            "generated": self.generated,
            "reopened": self.reopened,
            "stale_pops": self.stale_pops,
            "max_frontier": self.max_frontier,
            "runtime_s": self.runtime_s,
        }


@dataclass
class SearchResult(Generic[S]):
    status: SearchStatus
    path: list[S] = field(default_factory=list)
    transitions: list[Transition[S]] = field(default_factory=list)
    cost: float = float("inf")
    statistics: SearchStatistics = field(default_factory=SearchStatistics)
    message: str = ""

    @property
    def solved(self) -> bool:
        return self.status is SearchStatus.SOLVED


def reconstruct_path(
    parents: dict[S, tuple[S, Transition[S]]], goal: S
) -> tuple[list[S], list[Transition[S]]]:
    """Walk parent pointers back to the root.

    Returns ``(states, transitions)`` with ``len(transitions) == len(states) - 1``
    and ``transitions[i]`` describing the move ``states[i] -> states[i + 1]``.
    """
    states: list[S] = [goal]
    transitions: list[Transition[S]] = []
    cursor = goal
    while cursor in parents:
        parent, transition = parents[cursor]
        transitions.append(transition)
        states.append(parent)
        cursor = parent
    states.reverse()
    transitions.reverse()
    return states, transitions


def astar(
    problem: SearchProblem[S],
    heuristic: Callable[[S], float],
    *,
    max_expansions: int | None = None,
    time_limit_s: float | None = None,
    reopen_closed: bool = True,
) -> SearchResult[S]:
    start = problem.initial_state()
    started = time.perf_counter()
    stats = SearchStatistics()

    g_score: dict[S, float] = {start: 0.0}
    parents: dict[S, tuple[S, Transition[S]]] = {}
    closed: set[S] = set()
    counter = 0
    h0 = heuristic(start)
    frontier: list[tuple[float, float, int, S]] = [(h0, h0, counter, start)]
    stats.generated = 1

    while frontier:
        stats.max_frontier = max(stats.max_frontier, len(frontier))
        _, _, _, state = heapq.heappop(frontier)

        if state in closed:
            stats.stale_pops += 1
            continue
        closed.add(state)
        stats.expansions += 1

        if problem.is_goal(state):
            path, transitions = reconstruct_path(parents, state)
            stats.runtime_s = time.perf_counter() - started
            return SearchResult(
                SearchStatus.SOLVED, path, transitions, g_score[state], stats
            )

        if max_expansions is not None and stats.expansions >= max_expansions:
            stats.runtime_s = time.perf_counter() - started
            return SearchResult(
                SearchStatus.LIMIT_REACHED,
                statistics=stats,
                message=f"expansion limit {max_expansions} reached",
            )
        if time_limit_s is not None and (time.perf_counter() - started) > time_limit_s:
            stats.runtime_s = time.perf_counter() - started
            return SearchResult(
                SearchStatus.LIMIT_REACHED,
                statistics=stats,
                message=f"time limit {time_limit_s}s reached",
            )

        g_current = g_score[state]
        for transition in problem.successors(state):
            successor = transition.state
            if transition.cost == float("inf"):
                continue
            tentative = g_current + transition.cost
            known = g_score.get(successor)
            if known is not None and tentative >= known - 1e-12:
                continue
            if successor in closed:
                if not reopen_closed:
                    continue
                closed.discard(successor)
                stats.reopened += 1
            g_score[successor] = tentative
            parents[successor] = (state, transition)
            counter += 1
            h = heuristic(successor)
            heapq.heappush(frontier, (tentative + h, h, counter, successor))
            stats.generated += 1

    stats.runtime_s = time.perf_counter() - started
    return SearchResult(
        SearchStatus.UNSOLVABLE,
        statistics=stats,
        message="frontier exhausted without reaching the goal",
    )


def dijkstra(
    problem: SearchProblem[S],
    *,
    max_expansions: int | None = None,
    time_limit_s: float | None = None,
) -> SearchResult[S]:
    """Uniform-cost search: the optimality baseline the tests compare against."""
    return astar(
        problem,
        lambda _state: 0.0,
        max_expansions=max_expansions,
        time_limit_s=time_limit_s,
        reopen_closed=False,
    )
