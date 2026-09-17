"""Analytic straight-line reference.

Why this exists
---------------
:mod:`atp.baselines.direct` rasterises the origin-destination line with
Bresenham, so a "straight" route at, say, a 30 degree bearing is a staircase
alternating axis-aligned and diagonal moves -- a 45 degree corner every cell or
two.  Under a turn model those corners are charged, and on a fine grid they may
be rejected outright, so the rasterised baseline can look bad for a reason that
has nothing to do with the airspace.  Reading that as "A* beats the baseline"
would be the same class of contaminated comparison the Milestone 1 audit caught
in ``comparable_cost``.

This reference has **no corners by construction**: it is the single straight
segment from origin cell centre to destination cell centre, evaluated by the
same :class:`~atp.planning.cost.CostModel` and scored by the same
:func:`~atp.evaluation.metrics.evaluate_trajectory` as every other trajectory.

What it is and is not
---------------------
It is a *reference*, not a plan: a two-waypoint path is not a grid trajectory
and cannot be compared on expansions.  Its wind and risk are sampled at the same
resolution the cost model uses for any segment, so over a long leg a single
mid-segment wind sample is a much coarser approximation than it is over one
cell; its time and fuel are correspondingly rough.  It is exact about hard
restrictions, because those are tested exactly on the segment.  Use it to bound
distance and to separate genuine detour cost from rasterisation artefacts, not
as a fuel figure.
"""

from __future__ import annotations

from .direct import BaselineResult
from ..environment.airspace import GridState
from ..evaluation.metrics import evaluate_trajectory
from ..planning.problem import TrajectoryPlanningProblem


def plan_analytic_direct(problem: TrajectoryPlanningProblem) -> BaselineResult:
    """The origin-to-destination straight line as a single segment."""
    start = problem.start
    goal = problem.goal.state
    level = goal.il if problem.goal.match_level else start.il
    path = [start, GridState(goal.ix, goal.iy, level)]
    evaluation = evaluate_trajectory(problem.cost_model, path)
    return BaselineResult(path=path, evaluation=evaluation, name="analytic-direct")
