"""Heuristic properties checked against a brute-force optimum.

The claims in :mod:`atp.planning.heuristics` are not taken on trust: every
heuristic advertised as admissible is compared with the exact cost-to-go
computed by Dijkstra from each state, and A* results are compared with the
Dijkstra optimum.
"""

from __future__ import annotations

import pytest

from atp.core.geometry import Vec2
from atp.environment.airspace import GridState
from atp.environment.restrictions import CircularRestriction, RestrictionSet
from atp.environment.risk import GaussianHazard
from atp.environment.wind import UniformWind, VortexWind
from atp.planning.astar import astar, dijkstra
from atp.planning.cost import CostWeights
from atp.planning.heuristics import (
    EuclideanDistanceHeuristic,
    ManhattanHeuristic,
    OctileHeuristic,
    OptimisticCostHeuristic,
    WeightedHeuristic,
    ZeroHeuristic,
    build_heuristic,
)
from conftest import make_problem

ADMISSIBLE = [EuclideanDistanceHeuristic, OptimisticCostHeuristic, OctileHeuristic]

ENVIRONMENTS = {
    "still-air": {},
    "wind": {"wind": UniformWind(Vec2(70.0, -30.0))},
    "vortex": {"wind": VortexWind(Vec2(60.0, 60.0), 90.0, 30.0)},
    "hazard": {"risk": GaussianHazard(Vec2(60.0, 60.0), peak=2.0, sigma_nm=20.0)},
    "restricted": {
        "restrictions": RestrictionSet(
            regions=(
                CircularRestriction(
                    region_id="P-1", centre_nm=Vec2(60.0, 60.0), radius_nm=22.0
                ),
            )
        )
    },
}

WEIGHTS = CostWeights(
    time_cost_per_hour=3000.0,
    fuel_cost_per_kg=0.8,
    distance_cost_per_nm=2.0,
    risk_cost_per_exposure=5000.0,
)


def exact_cost_to_go(problem) -> dict[GridState, float]:  # noqa: C901
    """Dijkstra from every state, by running it backwards from the goal via a
    full forward sweep. The grid is small, so an O(V) repeat is acceptable."""
    costs: dict[GridState, float] = {}
    spec = problem.airspace.spec
    for state in spec.states():
        if problem.airspace.is_blocked(state):
            continue
        sub = make_problem(
            cells=spec.cells_x,
            cell_size_nm=spec.cell_size_nm,
            levels=spec.flight_levels,
            wind=problem.airspace.wind,
            restrictions=problem.airspace.restrictions,
            risk=problem.airspace.risk,
            weights=problem.cost_model.weights,
            start=state,
            goal=problem.goal.state,
            match_level=problem.goal.match_level,
        )
        result = dijkstra(sub)
        if result.solved:
            costs[state] = result.cost
    return costs


@pytest.mark.parametrize("env_name", sorted(ENVIRONMENTS))
@pytest.mark.parametrize("factory", ADMISSIBLE, ids=[f.__name__ for f in ADMISSIBLE])
def test_declared_admissible_heuristics_never_overestimate(env_name, factory):
    problem = make_problem(cells=7, weights=WEIGHTS, **ENVIRONMENTS[env_name])
    heuristic = factory(problem)
    assert heuristic.admissible
    for state, true_cost in exact_cost_to_go(problem).items():
        assert heuristic(state) <= true_cost + 1e-6, (
            f"{heuristic.name} overestimates at {state} in {env_name}"
        )


@pytest.mark.parametrize("env_name", sorted(ENVIRONMENTS))
@pytest.mark.parametrize("factory", ADMISSIBLE, ids=[f.__name__ for f in ADMISSIBLE])
def test_astar_with_admissible_heuristic_matches_dijkstra_cost(env_name, factory):
    problem = make_problem(cells=9, weights=WEIGHTS, **ENVIRONMENTS[env_name])
    reference = dijkstra(problem)
    result = astar(problem, factory(problem))
    assert reference.solved and result.solved
    assert result.cost == pytest.approx(reference.cost, rel=1e-9)


@pytest.mark.parametrize("factory", ADMISSIBLE, ids=[f.__name__ for f in ADMISSIBLE])
def test_admissibility_holds_when_the_goal_level_is_constrained(factory):
    """The original suite only exercised ``match_level=False``, which is what
    allowed an inadmissible vertical term to go unnoticed."""
    problem = make_problem(
        cells=6,
        cell_size_nm=20.0,
        levels=(280, 300, 320),
        weights=WEIGHTS,
        start=GridState(0, 0, 0),
        goal=GridState(5, 5, 2),
        match_level=True,
    )
    heuristic = factory(problem)
    assert heuristic.admissible
    for state, true_cost in exact_cost_to_go(problem).items():
        assert heuristic(state) <= true_cost + 1e-6


def test_zero_heuristic_expands_at_least_as_much_as_a_guided_one():
    problem = make_problem(cells=11, weights=WEIGHTS)
    blind = astar(problem, ZeroHeuristic())
    guided = astar(problem, OptimisticCostHeuristic(problem))
    assert guided.statistics.expansions <= blind.statistics.expansions
    assert guided.cost == pytest.approx(blind.cost, rel=1e-9)


def test_octile_declares_itself_inadmissible_on_16_connectivity():
    problem = make_problem(cells=9, connectivity=16, weights=WEIGHTS)
    assert not OctileHeuristic(problem).admissible
    assert OctileHeuristic(make_problem(cells=9, connectivity=8)).admissible


def test_manhattan_is_flagged_inadmissible_on_diagonal_grids():
    assert not ManhattanHeuristic(make_problem(cells=9)).admissible
    assert ManhattanHeuristic(make_problem(cells=9, connectivity=4)).admissible


def test_weighted_astar_respects_its_suboptimality_bound():
    problem = make_problem(cells=13, weights=WEIGHTS)
    optimum = dijkstra(problem).cost
    weight = 1.6
    heuristic = WeightedHeuristic(OptimisticCostHeuristic(problem), weight)
    result = astar(problem, heuristic, reopen_closed=True)
    assert result.solved
    assert result.cost <= weight * optimum + 1e-6
    assert not heuristic.admissible
    assert heuristic.describe()["suboptimality_bound"] == pytest.approx(weight)


def test_weighted_heuristic_rejects_weights_below_one():
    with pytest.raises(ValueError):
        WeightedHeuristic(ZeroHeuristic(), 0.5)


def test_build_heuristic_registry():
    problem = make_problem(cells=5)
    assert build_heuristic("optimistic", problem).name == "optimistic-cost"
    assert build_heuristic("zero", problem).name == "zero"
    assert build_heuristic("octile", problem, weight=2.0).weight == 2.0
    with pytest.raises(KeyError):
        build_heuristic("does-not-exist", problem)
