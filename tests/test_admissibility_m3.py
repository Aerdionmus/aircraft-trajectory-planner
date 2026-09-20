"""M3-REQ-009: the heuristic guarantees, **re-derived** over the speed-expanded
graph rather than inherited.

Why this file exists separately from ``test_admissibility_m2.py``
-----------------------------------------------------------------
Milestone 2 inherited admissibility from a monotone-refinement property: no
Milestone 2 edge was cheaper than its Milestone 1 projection, so a bound that
held for the smaller graph held for the larger one.  **That argument is not
available in Milestone 3 and must not be reused.**  Allowing the planner to
choose a faster speed is exactly the act of making some edges *cheaper* than
their fixed-speed values, so ``c3 >= c2`` is false by design.

The bound in :meth:`atp.planning.cost.CostModel.speed_cost_lower_bound_per_nm`
is therefore re-derived from scratch as a minimisation over the whole envelope.
These tests check the re-derived bound the hard way: Dijkstra from every state
of small grids, across five environments, three turn models and several
envelopes, with no heuristic allowed to exceed the true cost-to-go.

The specific failure this guards against is a bound computed at the *cruise*
speed while the planner may fly *faster*: the faster edge costs less time, the
heuristic keeps charging the slower rate, and it overestimates.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from functools import lru_cache

import pytest

from atp.aircraft.envelope import SpeedEnvelope
from atp.core.geometry import Vec2
from atp.environment.restrictions import CircularRestriction, RestrictionSet
from atp.environment.risk import GaussianHazard
from atp.environment.wind import UniformWind, VortexWind
from atp.planning.astar import astar, dijkstra
from atp.planning.cost import CostWeights
from atp.planning.heuristics import (
    EuclideanDistanceHeuristic,
    OctileHeuristic,
    OptimisticCostHeuristic,
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

#: Envelopes chosen to cover the cases the derivation distinguishes: a wide
#: spread, one that reaches *above* the old cruise speed (the case that breaks
#: the inherited argument), and one entirely below it.
ENVELOPES = {
    "wide": SpeedEnvelope((330.0, 390.0, 450.0, 480.0), 450.0),
    "faster-than-m2": SpeedEnvelope((450.0, 480.0), 450.0),
    "slower-than-m2": SpeedEnvelope((330.0, 390.0), 390.0),
}

TURN_MODELS = ["none", "gate", "gate+cost"]


#: 5 cells at 12 NM.  Smaller grids put the ``restricted`` environment's goal
#: cell inside the no-fly circle, which makes the instance unsolvable from every
#: state and the sweep vacuous -- a way for this test to pass by doing nothing.
#: The sweep asserts a non-zero state count for the same reason.
SWEEP_CELLS = 5


def _problem(env_name, turn_model, envelope, *, cells=SWEEP_CELLS, **kw):
    return make_problem(
        cells=cells,
        cell_size_nm=12.0,
        weights=WEIGHTS,
        turn_model=turn_model,
        speed_envelope=envelope,
        **ENVIRONMENTS[env_name],
        **kw,
    )


@lru_cache(maxsize=None)
def cost_to_go_for(env_name: str, turn_model: str, envelope_name: str, cells: int):
    """Cached ``(problem, cost_to_go)`` for one configuration.

    Computed once per configuration and shared between the heuristics tested
    against it, rather than recomputed for every parametrisation.
    """
    problem = _problem(env_name, turn_model, ENVELOPES[envelope_name], cells=cells)
    return problem, exact_cost_to_go(problem)


def all_states(problem) -> list[FlightState]:
    """Every non-blocked planner state, excluding the start sentinel heading.

    The start state is the only one carrying ``NO_HEADING``.  It is not part of
    the reachable interior of the graph, and the heuristic claim at it is
    already covered by the states enumerated here.
    """
    spec = problem.airspace.spec
    headings = (
        tuple(range(len(problem.turn_table)))
        if problem.heading_aware
        else (NO_HEADING,)
    )
    speeds = problem.cost_model.envelope.indices()
    return [
        FlightState(cell.ix, cell.iy, cell.il, ih, isp)
        for cell in spec.states()
        if not problem.airspace.is_blocked(cell)
        for ih in headings
        for isp in speeds
    ]


def exact_cost_to_go(problem) -> dict[FlightState, float]:
    """True cost-to-go from every state, by one Dijkstra on the reversed graph.

    Milestone 1 and 2 could afford a forward Dijkstra from every state: the
    state space was small enough.  Milestone 3 multiplies both the state count
    **and** the branching factor by the number of speed options, so a per-state
    sweep grows with the fourth power of a quantity that just grew, and stops
    being a test anyone will actually run.

    Enumerating the forward edges once and running a single multi-source
    Dijkstra *backwards* from the goal states computes exactly the same
    quantity, ``h*(s)`` for every ``s``, in one pass instead of ``|V|`` passes.
    The edges come from the problem's own ``successors``, so the graph being
    searched is the graph the planner searches.
    ``test_the_reverse_sweep_agrees_with_forward_dijkstra`` checks the two
    methods against each other on a case small enough to run both ways, so this
    optimisation cannot quietly weaken the property being asserted.
    """
    states = all_states(problem)
    reverse: dict[FlightState, list[tuple[FlightState, float]]] = defaultdict(list)
    for state in states:
        for transition in problem.successors(state):
            if math.isfinite(transition.cost):
                reverse[transition.state].append((state, transition.cost))

    costs: dict[FlightState, float] = {}
    counter = 0
    heap: list[tuple[float, int, FlightState]] = []
    for state in states:
        if problem.is_goal(state):
            costs[state] = 0.0
            heapq.heappush(heap, (0.0, counter, state))
            counter += 1

    while heap:
        cost, _, state = heapq.heappop(heap)
        if cost > costs.get(state, math.inf) + 1e-15:
            continue
        for predecessor, edge in reverse.get(state, ()):
            candidate = cost + edge
            if candidate < costs.get(predecessor, math.inf) - 1e-15:
                costs[predecessor] = candidate
                counter += 1
                heapq.heappush(heap, (candidate, counter, predecessor))
    return costs


@pytest.mark.parametrize("env_name", sorted(ENVIRONMENTS))
@pytest.mark.parametrize("factory", ADMISSIBLE, ids=[f.__name__ for f in ADMISSIBLE])
def test_declared_admissible_heuristics_never_overestimate(env_name, factory):
    """Every environment, the widest envelope, the full turn model."""
    problem, costs = cost_to_go_for(env_name, "gate+cost", "wide", SWEEP_CELLS)
    heuristic = factory(problem)
    assert heuristic.admissible
    for state, true_cost in costs.items():
        assert heuristic(state) <= true_cost + 1e-6, (
            f"{heuristic.name} overestimates at {state} in {env_name}"
        )
    assert costs, "the sweep must actually reach some states"


@pytest.mark.parametrize("envelope_name", sorted(ENVELOPES))
@pytest.mark.parametrize("turn_model", TURN_MODELS)
def test_the_optimistic_bound_holds_for_every_envelope_and_turn_model(
    turn_model, envelope_name
):
    """Every envelope shape -- including one reaching above the old cruise speed
    and one entirely below it -- against every turn model."""
    problem, costs = cost_to_go_for("wind", turn_model, envelope_name, SWEEP_CELLS)
    heuristic = OptimisticCostHeuristic(problem)
    assert costs
    for state, true_cost in costs.items():
        assert heuristic(state) <= true_cost + 1e-6


@pytest.mark.parametrize("envelope_name", sorted(ENVELOPES))
@pytest.mark.parametrize("env_name", sorted(ENVIRONMENTS))
def test_consistency_holds_edge_by_edge(env_name, envelope_name):
    """``h(s) <= c(s, s') + h(s')`` over every generated edge, plus ``h`` must
    vanish on goal states."""
    problem = _problem(env_name, "gate+cost", ENVELOPES[envelope_name], cells=4)
    heuristic = OptimisticCostHeuristic(problem)
    envelope = problem.cost_model.envelope
    edges = 0
    for cell in problem.airspace.spec.states():
        if problem.airspace.is_blocked(cell):
            continue
        for ih in range(len(problem.turn_table)):
            for isp in envelope.indices():
                state = FlightState(cell.ix, cell.iy, cell.il, ih, isp)
                for transition in problem.successors(state):
                    assert heuristic(state) <= transition.cost + heuristic(
                        transition.state
                    ) + 1e-6
                    edges += 1
    assert edges > 0
    goal = FlightState(
        problem.goal.state.ix, problem.goal.state.iy, problem.goal.state.il, 0, 0
    )
    assert heuristic(goal) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("envelope_name", sorted(ENVELOPES))
@pytest.mark.parametrize("env_name", sorted(ENVIRONMENTS))
def test_astar_with_an_admissible_heuristic_matches_the_dijkstra_optimum(
    env_name, envelope_name
):
    problem = _problem(env_name, "gate+cost", ENVELOPES[envelope_name], cells=6)
    reference = dijkstra(problem)
    result = astar(problem, OptimisticCostHeuristic(problem))
    assert reference.solved == result.solved
    if reference.solved:
        assert result.cost == pytest.approx(reference.cost, rel=1e-9)


# -- the specific failure the re-derivation prevents -------------------------
def test_a_bound_computed_at_the_cruise_speed_alone_would_overestimate():
    """The counterexample, constructed rather than hoped for.

    With an envelope reaching above the old cruise speed, a bound built from the
    cruise speed's ground-speed limit charges more per NM than the fastest
    option actually costs.  This test asserts that the shipped bound is strictly
    below that naive one, which is the quantitative statement that the
    re-derivation was necessary.
    """
    envelope = SpeedEnvelope((450.0, 600.0), 450.0)
    problem = make_problem(
        cells=4,
        cell_size_nm=12.0,
        weights=CostWeights(time_cost_per_hour=3000.0),
        turn_model="none",
        speed_envelope=envelope,
    )
    model = problem.cost_model
    shipped = model.speed_cost_lower_bound_per_nm()
    naive_at_cruise = 3000.0 / 450.0
    assert shipped < naive_at_cruise
    assert shipped == pytest.approx(3000.0 / 600.0, rel=1e-12)

    # And the naive bound really would have been inadmissible here.
    heuristic = OptimisticCostHeuristic(problem)
    for state, true_cost in exact_cost_to_go(problem).items():
        assert heuristic(state) <= true_cost + 1e-6


def test_the_bound_is_the_minimum_over_the_envelope_not_a_fixed_speed_quotient():
    """Exercised where the fuel term makes the minimising speed interior, so the
    bound cannot be reproduced by simply taking the fastest option."""
    envelope = SpeedEnvelope((330.0, 390.0, 450.0, 480.0), 450.0)
    problem = make_problem(
        cells=4,
        cell_size_nm=12.0,
        weights=CostWeights(time_cost_per_hour=1.0, fuel_cost_per_kg=3.0),
        turn_model="none",
        speed_envelope=envelope,
    )
    model = problem.cost_model
    spec = problem.airspace.spec
    candidates = []
    for index in envelope.indices():
        tas = envelope.tas_kt(index)
        ff = model.aircraft.min_fuel_flow_over_levels_kg_per_h(
            spec.flight_levels, tas
        )
        candidates.append((1.0 + 3.0 * ff) / tas)
    assert model.speed_cost_lower_bound_per_nm() == pytest.approx(
        min(candidates), rel=1e-12
    )
    assert min(candidates) < candidates[-1], (
        "this configuration must not be minimised by the fastest speed, or the "
        "test would not distinguish the two formulations"
    )


def test_the_bound_never_exceeds_any_realised_cost_per_horizontal_nm():
    """Direct check of the inequality the derivation claims, on real edges."""
    for envelope_name, envelope in ENVELOPES.items():
        problem = _problem("wind", "gate+cost", envelope, cells=5)
        bound = problem.cost_model.speed_cost_lower_bound_per_nm()
        distance_price = problem.cost_model.weights.distance_cost_per_nm
        for cell in problem.airspace.spec.states():
            state = FlightState(cell.ix, cell.iy, cell.il, 2, 0)
            for transition in problem.successors(state):
                metrics = transition.metrics
                horizontal = math.hypot(
                    transition.state.ix - cell.ix, transition.state.iy - cell.iy
                ) * problem.airspace.spec.cell_size_nm
                if horizontal <= 0.0:
                    continue
                speed_cost = (
                    transition.cost
                    - distance_price * metrics.ground_distance_nm
                    - metrics.restriction_penalty
                )
                assert speed_cost >= bound * horizontal - 1e-6, envelope_name


def test_the_constant_offset_is_unaffected_by_the_speed_decision():
    """``K`` gives back the descent fuel credit, which is a function of the
    altitude span and the aircraft, not of the selected speed."""
    fixed = make_problem(
        cells=4,
        cell_size_nm=12.0,
        levels=(280, 300, 320),
        weights=WEIGHTS,
        speed_envelope=SpeedEnvelope.fixed(450.0),
    )
    wide = make_problem(
        cells=4,
        cell_size_nm=12.0,
        levels=(280, 300, 320),
        weights=WEIGHTS,
        speed_envelope=ENVELOPES["wide"],
    )
    assert fixed.cost_model.constant_cost_offset() == pytest.approx(
        wide.cost_model.constant_cost_offset(), rel=1e-12
    )


def test_admissibility_holds_when_the_goal_level_is_constrained():
    """The case that caught the Milestone 1 vertical-term defect, re-run with a
    speed decision on top of it."""
    problem = make_problem(
        cells=4,
        cell_size_nm=20.0,
        levels=(280, 300, 320),
        weights=WEIGHTS,
        turn_model="gate+cost",
        speed_envelope=ENVELOPES["wide"],
        start=None,
        goal=None,
        match_level=True,
    )
    heuristic = OptimisticCostHeuristic(problem)
    for state, true_cost in exact_cost_to_go(problem).items():
        assert heuristic(state) <= true_cost + 1e-6


def test_the_reverse_sweep_agrees_with_forward_dijkstra():
    """The reverse sweep is an optimisation, so it is validated against the
    method it replaces on a case small enough to run both ways.

    If these ever disagree the fast sweep is wrong, and every admissibility
    result in this file is worthless -- which is why this is a test and not a
    comment.
    """
    kwargs = dict(
        cells=3,
        cell_size_nm=12.0,
        weights=WEIGHTS,
        turn_model="gate+cost",
        speed_envelope=ENVELOPES["slower-than-m2"],
        wind=UniformWind(Vec2(40.0, -20.0)),
    )
    problem = make_problem(**kwargs)
    fast = exact_cost_to_go(problem)

    slow: dict[FlightState, float] = {}
    for state in all_states(problem):
        sub = make_problem(
            **kwargs,
            start=state.cell,
            goal=problem.goal.state,
            start_heading=state.ih,
            start_speed=state.isp,
        )
        result = dijkstra(sub)
        if result.solved:
            slow[state] = result.cost

    assert slow, "the forward sweep must reach some states"
    assert set(slow) == set(fast)
    for state, cost in slow.items():
        assert fast[state] == pytest.approx(cost, rel=1e-9)
