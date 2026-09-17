"""Heuristic functions.

Each heuristic declares whether it is admissible and consistent *under the
model's stated assumptions*, so experiments can report solution quality against
a property the code asserts rather than a claim in a report.  The declarations
are checked empirically by ``tests/test_heuristics.py``, which compares A*
solutions against Dijkstra optima on small grids.

Terminology
-----------
admissible
    ``h(n) <= h*(n)`` for all ``n``.  Guarantees A* returns an optimal path.
consistent
    ``h(n) <= c(n, n') + h(n')``.  Guarantees no node is ever re-expanded, so
    the closed list can be trusted.

A consistent heuristic is admissible; the converse does not hold.  The search
supports re-opening closed nodes so that inadmissible/inconsistent heuristics
still terminate with a well-defined (if suboptimal) answer.

Heading and inherited admissibility (Milestone 2)
-------------------------------------------------
Every heuristic here is a function of the projection
``pi(ix, iy, il, ih) = (ix, iy, il)`` alone, which is what lets the Milestone 1
admissibility arguments carry over verbatim to the heading-augmented state
space.  Write ``G1`` for the Milestone 1 graph, ``G2`` for the Milestone 2 one.
Two facts hold by construction:

P1  every edge of ``G2`` projects onto an edge of ``G1`` -- the turn gate only
    *removes* successors, it never invents a move outside ``GridSpec.moves``;
P2  ``c2(s, s') >= c1(pi(s), pi(s'))`` -- turn terms are non-negative and no
    distance is ever subtracted (invariant 4 in :mod:`atp.planning.cost`).

*Admissibility is inherited.*  Let ``P2`` be an optimal ``G2`` path from ``s`` to
the goal.  By P1 its projection is a walk to the goal in ``G1``, and by P2 that
walk costs no more than ``h*_2(s)``.  All costs are non-negative, so no walk is
cheaper than the optimal path, giving
``h(pi(s)) <= h*_1(pi(s)) <= c1(pi(P2)) <= h*_2(s)``.

*Consistency is inherited.*  For any ``(s, s')`` in ``G2``,
``h(pi(s)) <= c1(pi(s), pi(s')) + h(pi(s')) <= c2(s, s') + h(pi(s'))``, and
``h`` still vanishes on goal states.

Both are asserted rather than claimed: ``tests/test_heuristics.py`` runs Dijkstra
from every state of small grids under both turn models, and
``tests/test_admissibility_m2.py`` checks the consistency inequality edge by
edge.  The argument is *fragile*: it depends entirely on invariant 4, so any
future change that credits a path for cutting a corner or smooths it after the
fact invalidates every admissibility flag in this module.

Lower-bound decomposition
-------------------------
For any feasible trajectory the total cost satisfies

    cost >= A * H_path + c_dist * D3_path

where ``H_path`` is the horizontal path length, ``D3_path`` the 3D path length,
``A = speed_cost_lower_bound_per_nm()`` (time + fuel + risk per horizontal NM)
and ``c_dist = distance_cost_per_nm``.  Since ``H_path >= H_straight`` and
``D3_path >= hypot(H_straight, V)``, the admissible heuristic is

    h = max(0, A * H_straight + c_dist * hypot(H_straight, V) - K)

where ``K = constant_cost_offset()`` gives back the one non-distance-proportional
term in the cost model: the fuel a descent is credited.  ``K`` is zero unless
fuel is priced *and* more than one flight level exists.

The two terms must be kept separate.  An earlier version multiplied the
*combined* per-NM bound (including ``c_dist``) by the 3D straight-line distance,
which charges the speed-derived cost against vertical distance as well.  That
overestimates whenever the goal flight level is constrained: vertical travel is
far slower than ``GS_max``, so ``A`` per vertical NM is not a lower bound on
anything.  A worked counterexample is in
``tests/test_audit_regressions.py::test_optimistic_heuristic_does_not_overestimate_on_level_constrained_diagonal``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from ..core.geometry import Vec2
from ..core.units import ft_to_nm
from ..environment.airspace import GridState
from .problem import TrajectoryPlanningProblem
from .state import cell_of


class Heuristic(ABC):
    """Maps a state to an estimate of the remaining cost."""

    name: str = "heuristic"
    admissible: bool = True
    consistent: bool = True

    @abstractmethod
    def __call__(self, state) -> float: ...

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "admissible": self.admissible,
            "consistent": self.consistent,
        }


class ZeroHeuristic(Heuristic):
    """h = 0. A* degenerates to Dijkstra -- the optimality reference."""

    name = "zero"
    admissible = True
    consistent = True

    def __call__(self, state) -> float:
        return 0.0


class _GoalRelative(Heuristic):
    def __init__(self, problem: TrajectoryPlanningProblem) -> None:
        self.problem = problem
        self.goal_xy: Vec2 = problem.goal_position_nm()
        self.goal_alt_ft = problem.airspace.altitude_ft(problem.goal.state)
        self.match_level = problem.goal.match_level

    def horizontal_nm(self, state) -> float:
        p = self.problem.airspace.centre_nm(cell_of(state))
        return (self.goal_xy - p).norm()

    def vertical_nm(self, state) -> float:
        if not self.match_level:
            return 0.0
        alt = self.problem.airspace.altitude_ft(cell_of(state))
        return ft_to_nm(abs(self.goal_alt_ft - alt))


class EuclideanDistanceHeuristic(_GoalRelative):
    """Straight-line distance priced only by ``distance_cost_per_nm``.

    Admissible because the straight-line 3D distance is a lower bound on the
    flown 3D distance -- which is exactly what ``distance_cost_per_nm`` prices --
    and every other cost term is non-negative.  Consistent because it is a
    metric scaled by a non-negative constant.  It is a very loose bound whenever
    distance is not the dominant priced term, and is identically zero when
    ``distance_cost_per_nm == 0``, in which case A* degenerates to Dijkstra.
    """

    name = "euclidean"
    admissible = True
    consistent = True

    def __call__(self, state) -> float:
        d = math.hypot(self.horizontal_nm(state), self.vertical_nm(state))
        return self.problem.cost_model.weights.distance_cost_per_nm * d


class OptimisticCostHeuristic(_GoalRelative):
    """Full-cost lower bound, decomposed as documented at the top of this module:

        h = A * H_straight + c_dist * hypot(H_straight, V)

    Admissible and consistent under the model's assumptions.  Admissible because
    each term separately lower-bounds the corresponding term of the true cost;
    consistent because it is a non-negative combination of two metrics and every
    edge cost dominates the same combination over that edge.  This is the
    default heuristic.
    """

    name = "optimistic-cost"
    admissible = True
    consistent = True

    def __init__(self, problem: TrajectoryPlanningProblem) -> None:
        super().__init__(problem)
        self.speed_cost_per_nm = problem.cost_model.speed_cost_lower_bound_per_nm()
        self.distance_cost_per_nm = problem.cost_model.weights.distance_cost_per_nm
        self.offset = problem.cost_model.constant_cost_offset()

    def __call__(self, state) -> float:
        horizontal = self.horizontal_nm(state)
        vertical = self.vertical_nm(state)
        return max(
            0.0,
            self.speed_cost_per_nm * horizontal
            + self.distance_cost_per_nm * math.hypot(horizontal, vertical)
            - self.offset,
        )


class OctileHeuristic(_GoalRelative):
    """Exact grid-distance lower bound for 4/8-connectivity.

    For 8-connectivity the octile distance is the shortest achievable path
    length on an empty grid, so it dominates the Euclidean bound (tighter, still
    admissible).  For 16-connectivity it **over**estimates the achievable path
    length and is therefore flagged inadmissible.
    """

    name = "octile"

    def __init__(self, problem: TrajectoryPlanningProblem) -> None:
        super().__init__(problem)
        self.speed_cost_per_nm = problem.cost_model.speed_cost_lower_bound_per_nm()
        self.distance_cost_per_nm = problem.cost_model.weights.distance_cost_per_nm
        self.offset = problem.cost_model.constant_cost_offset()
        self.cell = problem.airspace.spec.cell_size_nm
        connectivity = problem.airspace.spec.connectivity
        self.diagonal = connectivity >= 8
        self.admissible = connectivity <= 8
        self.consistent = connectivity <= 8

    def _octile_nm(self, state) -> float:
        dx = abs(state.ix - self.problem.goal.state.ix)
        dy = abs(state.iy - self.problem.goal.state.iy)
        if self.diagonal:
            straight = dx + dy - 2 * min(dx, dy)
            cells = straight + math.sqrt(2.0) * min(dx, dy)
        else:
            cells = dx + dy
        return cells * self.cell

    def __call__(self, state) -> float:
        horizontal = self._octile_nm(state)
        vertical = self.vertical_nm(state)
        return max(
            0.0,
            self.speed_cost_per_nm * horizontal
            + self.distance_cost_per_nm * math.hypot(horizontal, vertical)
            - self.offset,
        )


class ManhattanHeuristic(_GoalRelative):
    """Included precisely because it is **inadmissible** for diagonal grids.

    Keeps the "effect of heuristic formulation" experiment honest: it expands
    fewer nodes and can return worse paths.  Note that on 4-connectivity it is
    admissible *only* with respect to the speed-derived term; it deliberately
    ignores the distance price, which keeps it a lower bound rather than adding
    a second way to overestimate.
    """

    name = "manhattan"
    admissible = False
    consistent = False

    def __init__(self, problem: TrajectoryPlanningProblem) -> None:
        super().__init__(problem)
        self.cost_per_nm = problem.cost_model.speed_cost_lower_bound_per_nm()
        self.offset = problem.cost_model.constant_cost_offset()
        self.cell = problem.airspace.spec.cell_size_nm
        if problem.airspace.spec.connectivity == 4:
            self.admissible = True
            self.consistent = True

    def __call__(self, state) -> float:
        dx = abs(state.ix - self.problem.goal.state.ix)
        dy = abs(state.iy - self.problem.goal.state.iy)
        return max(0.0, self.cost_per_nm * (dx + dy) * self.cell - self.offset)


class WeightedHeuristic(Heuristic):
    """``w * h`` with ``w >= 1``: weighted A*.

    Bounded suboptimality: the returned solution costs at most ``w`` times the
    optimum when the base heuristic is admissible.  Trades optimality for
    expansions, which is the standard knob for large airspaces.
    """

    def __init__(self, base: Heuristic, weight: float) -> None:
        if weight < 1.0:
            raise ValueError("weighted A* requires weight >= 1")
        self.base = base
        self.weight = weight
        self.name = f"weighted({base.name},{weight:g})"
        self.admissible = base.admissible and weight == 1.0
        self.consistent = base.consistent and weight == 1.0

    def __call__(self, state) -> float:
        return self.weight * self.base(state)

    def describe(self) -> dict[str, object]:
        info = super().describe()
        info["suboptimality_bound"] = self.weight if self.base.admissible else None
        return info


#: Factories keyed by name, for the CLI and the experiment harness.
HEURISTIC_FACTORIES = {
    "zero": lambda problem: ZeroHeuristic(),
    "euclidean": EuclideanDistanceHeuristic,
    "optimistic": OptimisticCostHeuristic,
    "octile": OctileHeuristic,
    "manhattan": ManhattanHeuristic,
}


def build_heuristic(
    name: str, problem: TrajectoryPlanningProblem, *, weight: float = 1.0
) -> Heuristic:
    key = name.lower()
    if key not in HEURISTIC_FACTORIES:
        raise KeyError(
            f"unknown heuristic {name!r}; available: {sorted(HEURISTIC_FACTORIES)}"
        )
    heuristic = HEURISTIC_FACTORIES[key](problem)
    if weight != 1.0:
        heuristic = WeightedHeuristic(heuristic, weight)
    return heuristic
