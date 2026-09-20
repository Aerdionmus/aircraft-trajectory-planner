"""M3-REQ-004 / M3-REQ-007: speed as part of the planning state.

The load-bearing claim of this file is the one in
:mod:`atp.planning.state`: once speed can vary, the heading index alone is no
longer a sufficient statistic for the corner at the next node, because the air
heading of a leg depends on its TAS through the wind triangle.  If that were
false, ``isp`` would be dead weight in the state and Milestone 3 could have
chosen the speed per edge without a state dimension at all.  It is checked here
directly, on a concrete wind.
"""

from __future__ import annotations

import math

import pytest

from atp.aircraft.envelope import SYNTHETIC_JET_ENVELOPE, SpeedEnvelope
from atp.core.geometry import Vec2
from atp.environment.airspace import GridState
from atp.environment.wind import UniformWind, ZeroWind
from atp.evaluation.metrics import evaluate_trajectory
from atp.planning.astar import astar, dijkstra
from atp.planning.cost import CostWeights
from atp.planning.state import DEFAULT_SPEED_INDEX, NO_HEADING, FlightState
from conftest import make_problem

WEIGHTS = CostWeights(
    time_cost_per_hour=3000.0,
    fuel_cost_per_kg=0.8,
    distance_cost_per_nm=2.0,
)

ENVELOPE = SpeedEnvelope((330.0, 390.0, 450.0), 450.0)
FIXED = SpeedEnvelope.fixed(450.0)


def _problem(**kw):
    kw.setdefault("cells", 6)
    kw.setdefault("cell_size_nm", 12.0)
    kw.setdefault("weights", WEIGHTS)
    return make_problem(**kw)


# -- the state itself --------------------------------------------------------
def test_a_four_argument_flight_state_still_means_what_it_did():
    """Milestone 2 constructed ``FlightState(ix, iy, il, ih)``.  That must
    remain the same tuple, or every Milestone 2 test comparing states would be
    comparing different objects."""
    assert FlightState(1, 2, 0, 3) == FlightState(1, 2, 0, 3, 0)
    assert FlightState(1, 2, 0, 3).isp == DEFAULT_SPEED_INDEX
    assert hash(FlightState(1, 2, 0, 3)) == hash(FlightState(1, 2, 0, 3, 0))


def test_the_projection_ignores_both_heading_and_speed():
    states = [FlightState(1, 2, 0, h, s) for h in (0, 3) for s in (0, 1, 2)]
    assert {s.cell for s in states} == {GridState(1, 2, 0)}


# -- why speed has to be in the state ----------------------------------------
def test_in_wind_the_air_heading_of_a_leg_depends_on_its_speed():
    """The wind triangle solves ``psi_hat = (GS t_hat - w) / TAS``.  Two legs on
    the *same* ground track at different TAS therefore point differently."""
    from atp.aircraft.kinematics import solve_ground_speed

    wind = Vec2(0.0, 80.0)  # 80 kt from the south, pure crosswind on an east track
    track = Vec2(1.0, 0.0)
    slow = solve_ground_speed(330.0, wind, track)
    fast = solve_ground_speed(450.0, wind, track)
    assert slow.feasible and fast.feasible
    assert abs(slow.drift_angle_deg) > abs(fast.drift_angle_deg) + 1.0, (
        "a slower aircraft must crab further into the same crosswind"
    )
    assert slow.air_heading_unit != fast.air_heading_unit


def test_the_corner_charged_at_a_node_depends_on_the_incoming_speed():
    """The consequence of the above: with the two ground tracks held fixed, the
    heading change the bank limit constrains still moves with the incoming
    speed.  This is what makes ``isp`` part of the state rather than a
    per-edge choice."""
    problem = _problem(
        wind=UniformWind(Vec2(0.0, 80.0)),
        turn_model="gate+cost",
        speed_envelope=ENVELOPE,
    )
    model = problem.cost_model
    node = GridState(3, 3, 0)
    east = Vec2(1.0, 0.0)
    north_east = Vec2(1.0, 1.0).normalized()
    leg = problem.turn_table.cell_size_nm

    slow_in = model.turn_metrics(
        node, east, north_east, leg, leg, speed_index_in=0, speed_index_out=2
    )
    fast_in = model.turn_metrics(
        node, east, north_east, leg, leg, speed_index_in=2, speed_index_out=2
    )
    assert slow_in.feasible and fast_in.feasible
    assert slow_in.delta_heading_rad != pytest.approx(
        fast_in.delta_heading_rad, rel=1e-9
    ), "the incoming speed must change the heading change at the corner"
    # The ground-track change is identical in both cases, which is precisely the
    # distinction Milestone 2 introduced and Milestone 3 now depends on.
    assert slow_in.delta_track_rad == pytest.approx(fast_in.delta_track_rad)


def test_in_still_air_the_speed_coupling_vanishes():
    """The coupling is a wind effect.  In still air air heading equals ground
    track, so the incoming speed cannot change the heading change -- and the
    claim above must not be an artefact of something else."""
    problem = _problem(
        wind=ZeroWind(), turn_model="gate+cost", speed_envelope=ENVELOPE
    )
    model = problem.cost_model
    node = GridState(3, 3, 0)
    east = Vec2(1.0, 0.0)
    north_east = Vec2(1.0, 1.0).normalized()
    leg = problem.turn_table.cell_size_nm
    slow_in = model.turn_metrics(
        node, east, north_east, leg, leg, speed_index_in=0, speed_index_out=2
    )
    fast_in = model.turn_metrics(
        node, east, north_east, leg, leg, speed_index_in=2, speed_index_out=2
    )
    assert slow_in.delta_heading_rad == pytest.approx(
        fast_in.delta_heading_rad, rel=1e-12
    )


def test_the_corner_is_charged_at_the_faster_of_the_two_legs():
    """Documented as the conservative choice: larger radius (over-blocks) and
    lower turn rate (longer, so an upper bound on the charge)."""
    problem = _problem(turn_model="gate+cost", speed_envelope=ENVELOPE)
    model = problem.cost_model
    node = GridState(3, 3, 0)
    east = Vec2(1.0, 0.0)
    north_east = Vec2(1.0, 1.0).normalized()
    leg = problem.turn_table.cell_size_nm

    both_fast = model.turn_metrics(
        node, east, north_east, leg, leg, speed_index_in=2, speed_index_out=2
    )
    decelerating = model.turn_metrics(
        node, east, north_east, leg, leg, speed_index_in=2, speed_index_out=0
    )
    accelerating = model.turn_metrics(
        node, east, north_east, leg, leg, speed_index_in=0, speed_index_out=2
    )
    both_slow = model.turn_metrics(
        node, east, north_east, leg, leg, speed_index_in=0, speed_index_out=0
    )
    assert decelerating.time_h == pytest.approx(both_fast.time_h, rel=1e-12)
    assert accelerating.time_h == pytest.approx(both_fast.time_h, rel=1e-12)
    assert both_slow.time_h < both_fast.time_h
    assert both_slow.radius_nm < both_fast.radius_nm


# -- successor generation ----------------------------------------------------
def test_a_single_speed_envelope_produces_exactly_the_milestone_2_successors():
    """Structural parity, not numerical: same states, same order, same costs."""
    m2 = _problem(turn_model="gate+cost", start_heading=2)
    m3 = _problem(turn_model="gate+cost", start_heading=2, speed_envelope=FIXED)
    assert not m2.speed_aware and not m3.speed_aware
    state = m3.initial_state()
    assert state == m2.initial_state()
    left = list(m2.successors(state))
    right = list(m3.successors(state))
    assert [t.state for t in left] == [t.state for t in right]
    assert [t.cost for t in left] == [t.cost for t in right]


def test_successors_branch_over_every_available_speed():
    problem = _problem(turn_model="gate+cost", speed_envelope=ENVELOPE, cells=5)
    assert problem.speed_aware and problem.uses_flight_state
    state = FlightState(2, 2, 0, 2, 2)
    successors = list(problem.successors(state))
    speeds = {t.state.isp for t in successors}
    assert speeds == {0, 1, 2}
    # Every successor is reachable at each speed, so the branching factor is
    # exactly the Milestone 2 one times the number of speed options.
    m2 = _problem(turn_model="gate+cost", cells=5)
    m2_successors = list(m2.successors(FlightState(2, 2, 0, 2)))
    assert len(successors) == len(m2_successors) * 3


def test_successor_order_is_moves_then_speeds_then_levels():
    """Determinism rests on a fixed generation order; it is asserted rather than
    assumed."""
    problem = _problem(turn_model="gate+cost", speed_envelope=ENVELOPE, cells=5)
    state = FlightState(2, 2, 0, 2, 2)
    seen = [(t.state.ih, t.state.isp) for t in problem.successors(state)]
    assert seen, "the state must have successors for this to test anything"
    # Move index is the outer loop: it never revisits a value it has left.
    move_order = [ih for ih, _ in seen]
    first_seen = []
    for ih in move_order:
        if ih not in first_seen:
            first_seen.append(ih)
    assert move_order == [
        ih for ih in first_seen for _ in range(move_order.count(ih))
    ], "move index must be the outer loop"
    # Speed index is the next loop in, so within one move it ascends.
    for move in first_seen:
        speeds = [isp for ih, isp in seen if ih == move]
        assert speeds == sorted(speeds)
    # And it follows the declared connectivity order, not an arbitrary one.
    assert first_seen == sorted(first_seen)


def test_speed_is_carried_in_the_state_without_a_turn_model():
    """With ``turn_model='none'`` the heading is not tracked, but the speed
    still is, so the state stays a FlightState and the heading stays the
    sentinel."""
    problem = _problem(turn_model="none", speed_envelope=ENVELOPE)
    assert problem.speed_aware and not problem.heading_aware
    assert problem.uses_flight_state
    state = problem.initial_state()
    assert isinstance(state, FlightState)
    assert state.ih == NO_HEADING
    assert all(t.state.ih == NO_HEADING for t in problem.successors(state))


def test_a_milestone_1_problem_still_plans_over_grid_states():
    problem = _problem(turn_model="none")
    assert not problem.uses_flight_state
    assert isinstance(problem.initial_state(), GridState)


def test_the_start_speed_index_is_validated():
    with pytest.raises(ValueError, match="start_speed_index"):
        _problem(turn_model="gate+cost", speed_envelope=ENVELOPE, start_speed=9)


def test_the_start_speed_defaults_to_the_envelope_cruise():
    problem = _problem(turn_model="gate+cost", speed_envelope=ENVELOPE)
    assert problem.start_speed_index == ENVELOPE.cruise_index
    assert problem.initial_state().isp == ENVELOPE.cruise_index


def test_a_speed_outside_the_limits_at_a_level_is_never_generated():
    """The Mach limit makes 450 kt illegal high up; the successor generator must
    not offer it there."""
    envelope = SpeedEnvelope((330.0, 450.0), 450.0, max_mach=0.70)
    problem = _problem(
        turn_model="gate+cost",
        speed_envelope=envelope,
        levels=(300,),
        start_speed=0,
    )
    assert problem.cost_model.available_speed_indices(0) == (0,)
    state = FlightState(2, 2, 0, 2, 0)
    assert {t.state.isp for t in problem.successors(state)} == {0}


def test_an_unavailable_speed_makes_a_transition_infeasible_rather_than_clamped():
    envelope = SpeedEnvelope((330.0, 450.0), 450.0, max_mach=0.70)
    problem = _problem(
        turn_model="gate+cost", speed_envelope=envelope, start_speed=0
    )
    metrics = problem.cost_model.evaluate(
        GridState(1, 1, 0), GridState(2, 1, 0), 1
    )
    assert not metrics.feasible
    assert "speed envelope" in metrics.infeasible_reason


# -- end to end --------------------------------------------------------------
def test_the_planner_records_the_speed_it_chose():
    problem = _problem(turn_model="gate+cost", speed_envelope=ENVELOPE, cells=8)
    result = astar(problem, lambda _s: 0.0)
    assert result.solved
    assert all(isinstance(s, FlightState) for s in result.path)
    assert {s.isp for s in result.path[1:]} <= {0, 1, 2}


def test_a_repeated_run_returns_an_identical_speed_sequence():
    def run():
        problem = _problem(
            turn_model="gate+cost", speed_envelope=ENVELOPE, cells=8
        )
        result = dijkstra(problem)
        return [tuple(s) for s in result.path], result.cost

    assert run() == run()


def test_a_start_speed_outside_the_limits_at_the_start_level_is_refused():
    """The start speed defines the air heading of the notional incoming leg, so
    it is genuinely flown once a departure heading exists.  An option outside
    the operating limits there must be refused, not quietly clamped."""
    envelope = SpeedEnvelope((330.0, 450.0), 450.0, max_mach=0.70)
    with pytest.raises(ValueError, match="outside the operating limits"):
        _problem(turn_model="gate+cost", speed_envelope=envelope, start_speed=1)
    # The default start speed is the envelope cruise, so the same misconfiguration
    # must be caught when nothing is declared at all.
    with pytest.raises(ValueError, match="outside the operating limits"):
        _problem(turn_model="gate+cost", speed_envelope=envelope)


# -- regression: a pure level speed change must reach the next turn ---------
def test_a_pure_level_speed_change_is_visible_to_the_next_turn_evaluation():
    """horizontal @ V1 -> pure level change @ V2 -> horizontal @ V2 -> turn.

    ``evaluate_trajectory`` must price the corner using ``V2`` -- the speed
    actually carried by the state that arrives at the corner -- and not the
    speed of the horizontal leg that preceded the level change. A stale
    incoming speed at the corner is not merely a wrong number: at the sizes
    used here it flips a turn from feasible to infeasible, because turn
    radius grows with the square of speed.

    The grid is built so the 45 degree turn fits at 330 kt but not at 450 kt:
    at 25 degrees of bank the fly-by needs 2.62 NM of half-leg at 450 kt
    against 1.41 NM at 330 kt, and the shorter leg here provides exactly
    2.00 NM. So getting the incoming speed wrong does not just mis-price this
    trajectory -- it makes the evaluator disagree with the planner about
    whether the trajectory is even flyable.
    """
    envelope = SpeedEnvelope((330.0, 450.0), 450.0)  # index 0 = 330, index 1 = 450
    problem = make_problem(
        cells=8,
        cell_size_nm=4.0,
        levels=(300, 320),
        turn_model="gate+cost",
        allow_pure_level_change=True,
        speed_envelope=envelope,
        weights=CostWeights(
            time_cost_per_hour=3000.0, fuel_cost_per_kg=0.8, distance_cost_per_nm=2.0
        ),
    )
    model = problem.cost_model

    start = FlightState(2, 2, 0, NO_HEADING, 1)
    after_move = FlightState(3, 2, 0, 2, 1)  # east (move index 2), 450 kt
    after_level = FlightState(3, 2, 1, 2, 0)  # pure level change to 330 kt
    after_turn = FlightState(4, 3, 1, 1, 0)  # 45 deg turn (east -> NE, index 1), 330 kt
    path = [start, after_move, after_level, after_turn]

    # The planner itself must offer this exact transition as feasible: the
    # corner is legal because the state arriving at it carries 330 kt, not
    # because of anything evaluate_trajectory does.
    planner_successors = {
        t.state: t for t in problem.successors(after_level)
    }
    assert after_turn in planner_successors, (
        "the planner must accept this turn at the post-level-change speed"
    )
    planner_cost = planner_successors[after_turn].cost

    result = evaluate_trajectory(model, path, start_move=None)
    assert result.feasible, result.infeasible_reason
    assert not any(result.hard_violations)
    assert result.num_turns == 1
    assert result.geometric_corners == 1

    # Rebuild the same three segments independently and compare bit-exactly,
    # so this is a proof of agreement rather than a plausibility check.
    east = Vec2(1.0, 0.0)
    ne = Vec2(1.0, 1.0).normalized()
    leg = problem.turn_table.cell_size_nm
    expected_turn = model.turn_metrics(
        after_level.cell, east, ne, leg, leg * math.sqrt(2.0),
        speed_index_in=0, speed_index_out=0,
    )
    assert expected_turn.feasible, (
        "the test's own geometry must make the turn feasible at 330 kt"
    )
    expected_cost, expected_metrics = model.transition_cost_with_turn(
        after_level.cell, after_turn.cell, expected_turn, 0
    )
    assert expected_cost == pytest.approx(planner_cost, rel=1e-12)
    assert expected_metrics.time_h == pytest.approx(
        model.evaluate(after_level.cell, after_turn.cell, 0).time_h
        + expected_turn.time_h,
        rel=1e-12,
    )


def test_the_pure_level_speed_change_bug_would_have_made_the_turn_infeasible():
    """The negative control for the regression above: confirms the test's own
    geometry is discriminating, by evaluating the same corner at the *stale*
    incoming speed (450 kt) the bug would have used and showing it is rejected.

    If this did not fail, the positive test above would not actually be
    exercising the bug it claims to guard against.
    """
    envelope = SpeedEnvelope((330.0, 450.0), 450.0)
    problem = make_problem(
        cells=8,
        cell_size_nm=4.0,
        levels=(300, 320),
        turn_model="gate+cost",
        allow_pure_level_change=True,
        speed_envelope=envelope,
    )
    model = problem.cost_model
    node = GridState(3, 2, 1)
    east = Vec2(1.0, 0.0)
    ne = Vec2(1.0, 1.0).normalized()
    leg = problem.turn_table.cell_size_nm
    stale = model.turn_metrics(
        node, east, ne, leg, leg * math.sqrt(2.0),
        speed_index_in=1, speed_index_out=0,  # the bug: incoming still 450 kt
    )
    assert not stale.feasible, (
        "450 kt must not fit this turn, or the positive-side test above is "
        "not actually distinguishing the fix from the bug"
    )


# -- regression: an illegal default speed must not be silently flown -------
def test_a_default_speed_illegal_at_the_altitude_is_rejected_not_silently_flown():
    """``evaluate(a, b)`` with no speed named must mean "the default speed,
    validated" once there is a real decision to make -- not "the default
    speed, unchecked".

    The envelope's cruise speed is 450 kt, but a low Mach limit makes it
    illegal at this level. Before this was fixed, naming no speed at all
    skipped the envelope check entirely and evaluated at 450 kt regardless;
    an explicit ``speed_index=1`` (the same speed) was already correctly
    rejected by ``test_an_unavailable_speed_makes_a_transition_infeasible_rather_than_clamped``
    above. The two must now agree.
    """
    envelope = SpeedEnvelope((330.0, 450.0), 450.0, max_mach=0.60)
    problem = _problem(turn_model="gate+cost", speed_envelope=envelope, start_speed=0)
    model = problem.cost_model
    assert model.default_speed_index == 1  # the cruise speed, 450 kt
    assert 1 not in model.available_speed_indices(0)  # illegal at this level

    explicit = model.evaluate(GridState(1, 1, 0), GridState(2, 1, 0), 1)
    unnamed = model.evaluate(GridState(1, 1, 0), GridState(2, 1, 0))
    assert not explicit.feasible
    assert not unnamed.feasible
    assert "speed envelope" in unnamed.infeasible_reason


def test_an_unnamed_speed_is_unaffected_when_there_is_no_real_decision():
    """The other half of the fix: with no envelope (or a singleton one) there
    is nothing an unnamed speed could fail to satisfy, and the Milestone 1/2
    call path -- no envelope lookup at all -- must be exactly preserved."""
    problem = _problem(turn_model="gate+cost")
    model = problem.cost_model
    assert not model.models_speed
    metrics = model.evaluate(GridState(1, 1, 0), GridState(2, 1, 0))
    assert metrics.feasible
    assert metrics.tas_kt == model.aircraft.cruise_tas_kt


# -- regression: the envelope's altitude sensitivity reaches the planner ----
def test_climbing_a_level_removes_speed_options_from_the_successor_generator():
    """The other half of the altitude-sensitivity demonstration: it is not
    enough for ``SpeedEnvelope.available_indices`` to change with altitude in
    isolation (see ``test_speed_envelope.py``) -- the planner's own successor
    generator has to act on it, or ISA would be present without being
    load-bearing for search.

    A compact two-level problem, FL300 and FL390, ``turn_model="none"`` so the
    only thing under test is speed availability rather than turn geometry.
    """
    envelope = SYNTHETIC_JET_ENVELOPE
    problem = _problem(
        turn_model="none", speed_envelope=envelope, levels=(300, 390), cells=4
    )
    speed_of_480 = envelope.planning_tas_kt.index(480.0)

    at_fl300 = FlightState(1, 1, 0, NO_HEADING, envelope.cruise_index)
    at_fl390 = FlightState(1, 1, 1, NO_HEADING, envelope.cruise_index)

    speeds_from_fl300 = {t.state.isp for t in problem.successors(at_fl300)}
    speeds_from_fl390 = {t.state.isp for t in problem.successors(at_fl390)}

    assert speeds_from_fl300 == set(envelope.indices())
    assert speed_of_480 in speeds_from_fl300
    assert speed_of_480 not in speeds_from_fl390, (
        "the successor generator must not offer 480 kt once the level makes "
        "it illegal under the envelope's own Mmo limit"
    )
    assert speeds_from_fl390 < speeds_from_fl300


# -- a cheap, deterministic wind+speed interaction through real search ------
def test_wind_shifts_the_optimal_speed_through_a_full_search():
    """A small, cheap end-to-end demonstration that wind and speed interact
    through actual A* search, not only inside the corner math the coupling
    tests above exercise directly.

    Under a fuel-weighted objective, a headwind erodes ground speed enough
    that flying faster becomes worth its extra fuel burn -- the optimal speed
    is not a property of the aircraft or the objective alone, it moves with
    the wind the route is actually flown against. This is the same
    non-degeneracy property the milestone's cost-sensitivity scenarios
    demonstrate in still air, shown here to survive when wind is added, on a
    grid small enough to add no meaningful cost to the suite.

    **[LIMITATION, recorded rather than expanded on]** This is a unit-level
    integration check, not a full library scenario: the shipped
    `speed-choice`/`speed-turn-frontier` scenarios remain zero-wind, so the
    wind/speed/heading coupling `test_the_corner_charged_at_a_node_depends_on_the_incoming_speed`
    proves at the corner level is exercised end-to-end here but not by any
    registered, documented experiment.
    """
    envelope = SpeedEnvelope((330.0, 390.0, 450.0), 450.0)
    weights = CostWeights(
        time_cost_per_hour=200.0, fuel_cost_per_kg=3.0, distance_cost_per_nm=0.5
    )

    def plan(wind):
        problem = make_problem(
            cells=6,
            cell_size_nm=8.0,
            weights=weights,
            turn_model="gate+cost",
            speed_envelope=envelope,
            wind=wind,
            start_heading=2,
        )
        from atp.planning.astar import astar
        from atp.planning.heuristics import OptimisticCostHeuristic

        result = astar(problem, OptimisticCostHeuristic(problem))
        assert result.solved
        return result

    still_air = plan(ZeroWind())
    headwind = plan(UniformWind(Vec2(-120.0, 0.0)))

    still_air_speeds = {s.isp for s in still_air.path[1:]}
    headwind_speeds = {s.isp for s in headwind.path[1:]}
    assert still_air_speeds == {envelope.planning_tas_kt.index(330.0)}
    assert headwind_speeds == {envelope.planning_tas_kt.index(390.0)}
    assert still_air_speeds != headwind_speeds, (
        "the whole point: the optimal speed must move with the wind"
    )

    # Determinism holds for the windy case too, matching the still-air
    # determinism already covered elsewhere in this file.
    repeat = plan(UniformWind(Vec2(-120.0, 0.0)))
    assert [tuple(s) for s in headwind.path] == [tuple(s) for s in repeat.path]
    assert headwind.cost == repeat.cost


# -- regression: singleton envelope + restrictive limit must still bind -----
def test_a_singleton_envelope_with_a_restrictive_mach_limit_makes_450kt_infeasible():
    """The specific gap the reviewer identified: ``models_speed`` (whether
    there is a *decision*) and ``has_limits`` (whether the envelope declares
    a *constraint*) had been conflated. A singleton envelope -- no decision to
    make, so it correctly does not become a `FlightState` branching dimension
    -- can still declare a restrictive Mach limit, and that limit must still
    reject a transition flown at the one available speed once it is
    violated. 450 kt is about Mach 0.68 even at sea level and stays above a
    0.60 limit at every altitude this test touches, so this is not sensitive
    to which level is picked.
    """
    envelope = SpeedEnvelope((450.0,), 450.0, max_mach=0.60)
    problem = _problem(turn_model="none", speed_envelope=envelope)
    model = problem.cost_model

    assert not model.models_speed, "a singleton offers no decision"
    assert model.envelope.has_limits, "but it does declare a constraint"
    assert not problem.speed_aware
    assert not problem.uses_flight_state, (
        "no decision means the state must stay a plain GridState, exactly as "
        "for SpeedEnvelope.fixed() -- this is unaffected by has_limits"
    )

    unnamed = model.evaluate(GridState(1, 1, 0), GridState(2, 1, 0))
    explicit = model.evaluate(GridState(1, 1, 0), GridState(2, 1, 0), 0)
    assert not unnamed.feasible
    assert not explicit.feasible
    assert "speed envelope" in unnamed.infeasible_reason

    # And the planner's own (Milestone 1) successor generator -- which is what
    # actually runs for a singleton, decision-free envelope -- must agree:
    # the transition must not be offered at all.
    successors = {t.state: t for t in problem.successors(GridState(1, 1, 0))}
    assert GridState(2, 1, 0) not in successors


def test_a_singleton_envelope_with_no_limits_is_untouched():
    """The other half of the fix: ``SpeedEnvelope.fixed(V)`` -- no limits --
    must still take the exact Milestone 1/2 path, with no envelope lookup at
    all. This is the parity the reviewer required to stay exact."""
    problem = _problem(turn_model="none", speed_envelope=SpeedEnvelope.fixed(450.0))
    model = problem.cost_model
    assert not model.models_speed
    assert not model.envelope.has_limits
    metrics = model.evaluate(GridState(1, 1, 0), GridState(2, 1, 0))
    assert metrics.feasible
    assert metrics.tas_kt == 450.0


# -- regression: a speed change made during a pure level segment must count -
def test_speed_changes_counts_a_change_made_on_a_pure_level_transition():
    """The reviewer's exact case: 450 kt horizontal -> a pure level transition
    that selects 330 kt -> a further horizontal leg still at 330 kt. The two
    horizontal legs never differ from each other (330 == 330), so a counter
    that only compared horizontal-to-horizontal speeds would see zero changes
    -- even though the aircraft genuinely changed speed exactly once, at the
    level transition. ``speed_changes`` must count that change regardless of
    which kind of segment it happened on.
    """
    envelope = SpeedEnvelope((330.0, 450.0), 450.0)  # index 0 = 330, index 1 = 450
    problem = _problem(
        turn_model="none",
        allow_pure_level_change=True,
        speed_envelope=envelope,
        levels=(300, 320),
    )
    model = problem.cost_model

    start = FlightState(2, 2, 0, NO_HEADING, 1)
    leg1 = FlightState(3, 2, 0, NO_HEADING, 1)  # horizontal @ 450 kt
    level = FlightState(3, 2, 1, NO_HEADING, 0)  # pure level change, selects 330 kt
    leg2 = FlightState(4, 2, 1, NO_HEADING, 0)  # horizontal @ 330 kt, unchanged from `level`
    path = [start, leg1, level, leg2]

    result = evaluate_trajectory(model, path, start_move=None)
    assert result.feasible, result.infeasible_reason
    assert result.speed_changes == 1, (
        "exactly one real speed change occurred, at the level transition"
    )


def test_speed_changes_does_not_double_count_a_level_change_that_keeps_speed():
    """The companion case: if the level change selects the *same* speed the
    horizontal legs already use, nothing changed and the count must stay at
    zero -- this is what proves the fix counts real changes rather than
    simply crediting every pure level segment."""
    envelope = SpeedEnvelope((330.0, 450.0), 450.0)
    problem = _problem(
        turn_model="none",
        allow_pure_level_change=True,
        speed_envelope=envelope,
        levels=(300, 320),
    )
    model = problem.cost_model

    start = FlightState(2, 2, 0, NO_HEADING, 1)
    leg1 = FlightState(3, 2, 0, NO_HEADING, 1)  # @ 450 kt
    level = FlightState(3, 2, 1, NO_HEADING, 1)  # pure level change, stays 450 kt
    leg2 = FlightState(4, 2, 1, NO_HEADING, 1)  # @ 450 kt
    path = [start, leg1, level, leg2]

    result = evaluate_trajectory(model, path, start_move=None)
    assert result.feasible, result.infeasible_reason
    assert result.speed_changes == 0


def test_speed_changes_still_agrees_with_the_planner_on_a_multi_change_path():
    """A longer case combining a horizontal-to-horizontal change with a
    level-transition change, cross-checked against the planner's own
    successors so the count is proven against real transitions rather than
    only against hand-built ones.

    The path opens with a leg flown at the same speed as the start's notional
    incoming speed, so the first *real* change happens on the second segment
    -- consistent with the departure-corner test above, a change against the
    notional start speed is not itself counted, since nothing was actually
    flown before the first real leg.
    """
    envelope = SpeedEnvelope((330.0, 390.0, 450.0), 450.0)
    problem = _problem(
        turn_model="none",
        allow_pure_level_change=True,
        speed_envelope=envelope,
        levels=(300, 320),
        cells=8,
    )
    model = problem.cost_model

    start = FlightState(2, 2, 0, NO_HEADING, 2)  # notional, 450 kt
    leg0 = FlightState(3, 2, 0, NO_HEADING, 2)  # @ 450 kt: matches start, no change
    leg1 = FlightState(4, 2, 0, NO_HEADING, 0)  # -> 330 kt: change 1 (horizontal)
    leg2 = FlightState(5, 2, 0, NO_HEADING, 0)  # stays 330 kt: no change
    level = FlightState(5, 2, 1, NO_HEADING, 1)  # -> 390 kt: change 2 (pure level)
    leg3 = FlightState(6, 2, 1, NO_HEADING, 1)  # stays 390 kt: no change
    path = [start, leg0, leg1, leg2, level, leg3]

    # Every one of these transitions must actually be offered by the planner,
    # so the count is checked against a real, reachable trajectory.
    for a, b in zip(path, path[1:]):
        successors = {t.state: t for t in problem.successors(a)}
        assert b in successors, f"{a} -> {b} must be a real planner transition"

    result = evaluate_trajectory(model, path, start_move=None)
    assert result.feasible, result.infeasible_reason
    assert result.speed_changes == 2
