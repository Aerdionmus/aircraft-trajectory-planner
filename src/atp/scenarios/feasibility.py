"""Scenario feasibility.

Non-blocked origin and destination cells are **not** sufficient for a scenario
to be solvable: several restrictions can jointly form a barrier that no
trajectory crosses, and vertical-rate limits can make a level-constrained goal
unreachable even in open airspace.  The only sound check is reachability over
the actual transition model, which is what this module does.

The search is an unweighted breadth-first flood over
:meth:`~atp.planning.problem.TrajectoryPlanningProblem.successors`, so it uses
exactly the transitions the planner would use, including every feasibility
rejection in the cost model.  It answers "is there any feasible path", not "what
does it cost", and so is far cheaper than running Dijkstra.
"""

from __future__ import annotations

from collections import deque

from ..environment.airspace import GridState
from ..planning.problem import TrajectoryPlanningProblem


def is_goal_reachable(
    problem: TrajectoryPlanningProblem, *, max_states: int | None = None
) -> bool:
    """True if any feasible transition sequence reaches the goal.

    ``max_states`` bounds the flood; exceeding it returns ``False``, i.e. the
    answer is "not shown to be reachable" rather than "proved unreachable".
    """
    start = problem.initial_state()
    if problem.airspace.is_blocked(start) or problem.airspace.is_blocked(
        problem.goal.state
    ):
        return False

    seen: set[GridState] = {start}
    queue: deque[GridState] = deque([start])
    while queue:
        state = queue.popleft()
        if problem.is_goal(state):
            return True
        if max_states is not None and len(seen) > max_states:
            return False
        for transition in problem.successors(state):
            if transition.state not in seen:
                seen.add(transition.state)
                queue.append(transition.state)
    return False
