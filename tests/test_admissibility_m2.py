"""T-24 to T-29: the heuristic guarantees, re-proved empirically over the
heading-augmented state space.

Milestone 1 asserted admissibility by running Dijkstra from every ``GridState``
of small grids.  Milestone 2 does the same over every ``FlightState``, which is
``K`` times as much work, so the grids are smaller and the cell size is 12 NM --
fine enough to search quickly, coarse enough that the aircraft can actually turn
(at 5 NM and 25 degrees of bank nothing but straight flight is legal; see
``tests/test_heading_state.py``).

The argument these tests check is in :mod:`atp.planning.heuristics`: every
heuristic is a function of the projection ``pi`` alone, the Milestone 2 edge set
projects into the Milestone 1 one, and no Milestone 2 edge is cheaper than its
projection -- so admissibility and consistency are *inherited* rather than
re-derived.  If invariant 4 in :mod:`atp.planning.cost` is ever broken, these
are the tests that should fail.
"""

from __future__ import annotations

import math

import pytest

from atp.core.geometry import Vec2
from atp.environment.restrictions import CircularRestriction, RestrictionSet
from atp.environment.risk import GaussianHazard
from atp.environment.wind import UniformWind, VortexWind
from atp.planning.astar import astar, dijkstra
from atp.planning.cost import CostWeights
from atp.planning.heuristics import (
    HEURISTIC_FACTORIES,
    EuclideanDistanceHeuristic,
    ManhattanHeuristic,
    OctileHeuristic,
    OptimisticCostHeuristic,
    WeightedHeuristic,
    ZeroHeuristic,
    build_heuristic,
)
from atp.planning.state import NO_HEADING, FlightState
from conftest import make_problem

ADMISSIBLE = [EuclideanDistanceHeuristic, OptimisticCostHeuristic, OctileHeuristic]

WEIGHTS = CostWeights(
    time_cost_per_hour=3000.0,
    fuel_cost_per_kg=0.8,
    distance_cost_per_nm=2.0,
    risk_cost_per_exposure=5000.0,
)

ENVIRONMENTS = {
    "still-air": {},
    "wind": {"wind": UniformWind(Vec2(70.0, -30.0))},
    "vortex": {"wind": VortexWind(Vec2(40.0, 40.0), 90.0, 30.0)},
    "hazard": {"risk": GaussianHazard(Vec2(40.0, 40.0), peak=2.0, sigma_nm=20.0)},
    "restricted": {
        "restrictions": RestrictionSet(
            regions=(
                CircularRestriction(
                    region_id="P-1", centre_nm=Vec2(40.0, 40.0), radius_nm=14.0
                ),
            )
        )
    },
}

TURN_MODELS = ["none", "gate", "gate+cost"]


def _problem(env_name, turn_model, *, cells=5, start=None, start_heading=None, **kw):
    return make_problem(
        cells=cells,
        cell_size_nm=12.0,
        weights=WEIGHTS,
        turn_model=turn_model,
        start=start,
        start_heading=start_heading,
        **ENVIRONMENTS[env_name],
        **kw,
    )


def _states(problem):
    """Every state the search could legitimately start from."""
    spec = problem.airspace.spec
    for cell in spec.states():
        if problem.airspace.is_blocked(cell):
            continue
        if not problem.heading_aware:
            yield cell, None
            continue
        for ih in list(range(len(problem.turn_table))) + [NO_HEADING]:
            yield cell, ih


def exact_cost_to_go(env_name, turn_model, cells=5):
    """Dijkstra from every state, over the graph the planner actually uses."""
    reference = _problem(env_name, turn_model, cells=cells)
    spec = reference.airspace.spec
    costs = {}
    for cell, ih in _states(reference):
        sub = _problem(
            env_name,
            turn_model,
            cells=cells,
            start=cell,
            start_heading=None if ih in (None, NO_HEADING) else ih,
            goal=reference.goal.state,
        )
        result = dijkstra(sub)
        if result.solved:
            key = (
                cell
                if ih is None
                else FlightState(cell.ix, cell.iy, cell.il, NO_HEADING if ih == NO_HEADING else ih)
            )
            costs[key] = result.cost
    assert spec.size > 0
    return reference, costs


@pytest.mark.parametrize("env_name", sorted(ENVIRONMENTS))
@pytest.mark.parametrize("turn_model", TURN_MODELS)
@pytest.mark.parametrize("factory", ADMISSIBLE, ids=[f.__name__ for f in ADMISSIBLE])
def test_declared_admissible_heuristics_never_overestimate(
    env_name, turn_model, factory
):
    """T-24."""
    problem, costs = exact_cost_to_go(env_name, turn_model)
    heuristic = factory(problem)
    assert heuristic.admissible
    assert costs, "the reference sweep found no reachable state"
    for state, true_cost in costs.items():
        assert heuristic(state) <= true_cost + 1e-6, (
            f"{heuristic.name} overestimates at {state} "
            f"in {env_name}/{turn_model}: {heuristic(state)} > {true_cost}"
        )


@pytest.mark.parametrize("turn_model", ["gate", "gate+cost"])
@pytest.mark.parametrize("factory", ADMISSIBLE, ids=[f.__name__ for f in ADMISSIBLE])
def test_admissibility_holds_when_the_goal_level_is_constrained(turn_model, factory):
    """T-24.  Constraining the goal level is what exposed the Milestone 1
    heuristic defect, so it is exercised under turns too."""
    reference = make_problem(
        cells=4,
        cell_size_nm=12.0,
        levels=(300, 320),
        weights=WEIGHTS,
        turn_model=turn_model,
        goal=None,
        match_level=True,
    )
    heuristic = factory(reference)
    spec = reference.airspace.spec
    for cell in spec.states():
        for ih in list(range(len(reference.turn_table))) + [NO_HEADING]:
            sub = make_problem(
                cells=4,
                cell_size_nm=12.0,
                levels=(300, 320),
                weights=WEIGHTS,
                turn_model=turn_model,
                start=cell,
                start_heading=None if ih == NO_HEADING else ih,
                goal=reference.goal.state,
                match_level=True,
            )
            result = dijkstra(sub)
            if not result.solved:
                continue
            state = FlightState(cell.ix, cell.iy, cell.il, ih)
            assert heuristic(state) <= result.cost + 1e-6, (state, turn_model)


@pytest.mark.parametrize("env_name", sorted(ENVIRONMENTS))
@pytest.mark.parametrize("turn_model", ["gate", "gate+cost"])
@pytest.mark.parametrize(
    "factory",
    ADMISSIBLE + [lambda p: ZeroHeuristic()],
    ids=[f.__name__ for f in ADMISSIBLE] + ["ZeroHeuristic"],
)
def test_consistency_inequality_holds_on_every_edge(env_name, turn_model, factory):
    """T-25.  ``h(s) <= c(s, s') + h(s')`` checked directly, edge by edge, rather
    than inferred from A* happening to agree with Dijkstra."""
    problem = _problem(env_name, turn_model, cells=5)
    heuristic = factory(problem)
    if not heuristic.consistent:
        pytest.skip("heuristic does not claim consistency")

    checked = 0
    for cell, ih in _states(problem):
        state = FlightState(cell.ix, cell.iy, cell.il, ih)
        for transition in problem.successors(state):
            assert heuristic(state) <= transition.cost + heuristic(transition.state) + 1e-6
            checked += 1
    assert checked > 100

    # And h must vanish on the goal, or the inequality proves nothing.
    goal = problem.goal.state
    for ih in list(range(len(problem.turn_table))) + [NO_HEADING]:
        goal_state = FlightState(goal.ix, goal.iy, goal.il, ih)
        assert problem.is_goal(goal_state)
        assert heuristic(goal_state) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("env_name", sorted(ENVIRONMENTS))
@pytest.mark.parametrize("turn_model", TURN_MODELS)
@pytest.mark.parametrize("factory", ADMISSIBLE, ids=[f.__name__ for f in ADMISSIBLE])
def test_astar_matches_the_dijkstra_optimum(env_name, turn_model, factory):
    """T-26: optimality on small state spaces, under every turn model."""
    problem = _problem(env_name, turn_model, cells=6)
    reference = dijkstra(problem)
    result = astar(problem, factory(problem))
    assert reference.solved == result.solved
    if reference.solved:
        assert result.cost == pytest.approx(reference.cost, rel=1e-9)


@pytest.mark.parametrize("turn_model", ["gate", "gate+cost"])
@pytest.mark.parametrize("factory", ADMISSIBLE, ids=[f.__name__ for f in ADMISSIBLE])
def test_consistent_heuristics_never_reopen_a_closed_node(turn_model, factory):
    """T-16.  A non-zero ``reopened`` count is direct evidence that a turn cost
    has broken consistency."""
    problem = _problem("wind", turn_model, cells=6)
    result = astar(problem, factory(problem), reopen_closed=True)
    assert result.solved
    assert result.statistics.reopened == 0


@pytest.mark.parametrize("weight", [1.0, 1.5, 2.5])
def test_weighted_astar_respects_its_suboptimality_bound_under_turns(weight):
    """T-29.  The wrapper is state-agnostic, so its semantics are unchanged; what
    is checked here is that the bound still holds over the larger graph."""
    problem = _problem("wind", "gate+cost", cells=6)
    optimum = dijkstra(problem)
    assert optimum.solved
    heuristic = WeightedHeuristic(OptimisticCostHeuristic(problem), weight)
    result = astar(problem, heuristic, reopen_closed=not heuristic.consistent)
    assert result.solved
    assert result.cost <= optimum.cost * weight + 1e-6
    assert heuristic.describe()["suboptimality_bound"] == weight
    if weight == 1.0:
        assert heuristic.admissible and heuristic.consistent
    else:
        assert not heuristic.admissible


def test_weighted_heuristic_still_rejects_weights_below_one():
    problem = _problem("still-air", "gate+cost")
    with pytest.raises(ValueError):
        WeightedHeuristic(OptimisticCostHeuristic(problem), 0.9)


def test_manhattan_remains_the_inadmissible_control():
    """Milestone 2 must not quietly promote the negative control."""
    problem = _problem("still-air", "gate+cost")
    heuristic = ManhattanHeuristic(problem)
    assert not heuristic.admissible and not heuristic.consistent
    four_connected = make_problem(
        cells=5, cell_size_nm=12.0, connectivity=4, turn_model="gate+cost"
    )
    assert ManhattanHeuristic(four_connected).admissible


def test_octile_still_declares_itself_inadmissible_on_sixteen_connectivity():
    problem = make_problem(
        cells=5, cell_size_nm=12.0, connectivity=16, turn_model="gate+cost"
    )
    assert not OctileHeuristic(problem).admissible


def test_heuristic_registry_is_unchanged_and_carries_no_dubins_entry():
    """T-27/T-28 are **not** implemented, deliberately.

    The design review proposed a ``DubinsLowerBoundHeuristic`` built on the
    free-terminal-heading Dubins length.  The closed form for a turn-then-tangent
    (CS) path is straightforward, but admissibility needs that length to be a
    *lower* bound on the flown path, which requires CS to be the **optimal**
    curvature-bounded path to a point with free final heading.  Minimising over a
    subset of path families yields an upper bound on the optimum, not a lower
    one, and the case where the target lies inside one of the two turning circles
    was not settled within this milestone.  Shipping ``admissible = True`` on an
    unproved claim is precisely the failure mode the Milestone 1 audit caught, so
    the heuristic is omitted rather than guessed at.  ``docs/turn_model.md``
    records the derivation that was reached and what remains to be proved.
    """
    assert sorted(HEURISTIC_FACTORIES) == [
        "euclidean",
        "manhattan",
        "octile",
        "optimistic",
        "zero",
    ]
    with pytest.raises(KeyError):
        build_heuristic("dubins", _problem("still-air", "gate+cost"))


def test_optimistic_remains_the_default_and_is_admissible_everywhere():
    """The safe default must stay safe: it depends on no wind precondition."""
    for env_name in ENVIRONMENTS:
        problem = _problem(env_name, "gate+cost")
        heuristic = OptimisticCostHeuristic(problem)
        assert heuristic.admissible and heuristic.consistent
        assert heuristic.name == "optimistic-cost"
        assert build_heuristic("optimistic", problem).name == heuristic.name


def test_heading_blind_heuristics_are_sound_but_lose_information():
    """The cost of inheriting: ``h`` cannot see that an aircraft pointing away
    from the goal must first spend time turning round, so the bound is looser
    than it was.  Soundness is asserted above; this pins that the two states
    differing only in heading get the same estimate, which is the mechanism."""
    problem = _problem("still-air", "gate+cost")
    heuristic = OptimisticCostHeuristic(problem)
    values = {heuristic(FlightState(1, 1, 0, ih)) for ih in range(8)}
    assert len(values) == 1

    from atp.environment.airspace import GridState

    interior, goal = GridState(3, 3, 0), GridState(6, 6, 0)
    towards = dijkstra(
        _problem(
            "still-air", "gate+cost", cells=7, start=interior, start_heading=1, goal=goal
        )
    )
    away = dijkstra(
        _problem(
            "still-air", "gate+cost", cells=7, start=interior, start_heading=5, goal=goal
        )
    )
    assert towards.solved and away.solved
    # Starting pointed away from the goal genuinely costs more -- here nearly
    # twice as much -- and the heuristic above is blind to exactly that
    # difference, which is the informedness the inheritance argument gives up.
    assert away.cost > towards.cost
    big = _problem("still-air", "gate+cost", cells=7, goal=goal)
    big_heuristic = OptimisticCostHeuristic(big)
    assert big_heuristic(FlightState(3, 3, 0, 1)) == big_heuristic(
        FlightState(3, 3, 0, 5)
    )
    assert big_heuristic(FlightState(3, 3, 0, 5)) <= away.cost + 1e-6


def test_a_corner_start_pointed_at_the_wall_is_reported_unsolvable():
    """A real consequence of turn dynamics, worth pinning: from the corner cell
    the only in-bounds moves are 135 and 180 degrees off a heading that points
    out of the airspace, so every one of them is gated away and the instance has
    no solution at all.  It must be reported, not crashed or silently relaxed."""
    problem = _problem("still-air", "gate+cost", start_heading=5)
    result = dijkstra(problem)
    assert not result.solved
    assert result.status.value == "unsolvable"
    assert list(problem.successors(problem.initial_state())) == []


def test_infinite_costs_are_never_pushed_onto_the_frontier():
    problem = _problem("restricted", "gate+cost", cells=6)
    for cell, ih in _states(problem):
        for transition in problem.successors(FlightState(cell.ix, cell.iy, cell.il, ih)):
            assert not math.isinf(transition.cost)
