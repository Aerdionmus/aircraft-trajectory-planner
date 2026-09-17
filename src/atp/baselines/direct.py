"""Direct-route baseline.

The lower bound any search must beat on distance and the upper bound it must
beat on constraint compliance: fly the straight line from origin to destination
at the starting level, ignoring restrictions, wind and risk entirely.

Implemented as a supercover line rasterisation so the produced cell chain is
contiguous under 8-connectivity and can be scored by exactly the same cost model
as a searched trajectory.  When the line crosses a hard restriction the baseline
still returns the path but the evaluation reports it as infeasible -- that
contrast is the point of the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..environment.airspace import GridState
from ..evaluation.metrics import TrajectoryEvaluation, evaluate_trajectory
from ..planning.problem import TrajectoryPlanningProblem


@dataclass
class BaselineResult:
    path: list[GridState]
    evaluation: TrajectoryEvaluation
    name: str = "direct-route"


def _bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Integer line rasterisation (8-connected, no duplicate cells)."""
    points: list[tuple[int, int]] = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return points


def plan_direct_route(problem: TrajectoryPlanningProblem) -> BaselineResult:
    start, goal = problem.start, problem.goal.state

    cells = _bresenham(start.ix, start.iy, goal.ix, goal.iy)
    levels = _level_profile(start.il, goal.il, len(cells), problem.goal.match_level)
    path = [GridState(ix, iy, il) for (ix, iy), il in zip(cells, levels)]

    evaluation = evaluate_trajectory(problem.cost_model, path)
    return BaselineResult(path=path, evaluation=evaluation)


def _level_profile(
    start_level: int, goal_level: int, length: int, match_level: bool
) -> list[int]:
    """Level index for each cell of the direct route.

    If the goal level is not constrained the baseline stays at the departure
    level; otherwise it changes level one step at a time, as early as possible.
    """
    if not match_level or goal_level == start_level or length == 0:
        return [start_level] * length
    step = 1 if goal_level > start_level else -1
    levels: list[int] = []
    current = start_level
    for _ in range(length):
        levels.append(current)
        if current != goal_level:
            current += step
    return levels
