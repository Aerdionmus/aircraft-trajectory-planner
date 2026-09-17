"""T-10 to T-16, T-22, T-23, T-31: the heading state and the turn gate.

The heading index is an index into ``GridSpec.moves``, so the representable
headings are exactly the representable ground tracks and there is no rounding
anywhere.  These tests pin that, the non-uniform spacing of ``MOVES_16``, the
gate's agreement with its own closed form, and the successor-ordering guarantee
the determinism argument rests on.
"""

from __future__ import annotations

import math

import pytest

from atp.aircraft.turn import tangent_length_nm
from atp.core.geometry import Vec2, angle_between
from atp.environment.airspace import CONNECTIVITY, MOVES_8, MOVES_16, GridState
from atp.planning.state import NO_HEADING, FlightState, TurnTable, cell_of
from conftest import make_problem


def _table(connectivity: int = 8, cell_size_nm: float = 12.0) -> TurnTable:
    return TurnTable(CONNECTIVITY[connectivity], cell_size_nm)


# -- state and projection ----------------------------------------------------
def test_flight_state_projects_onto_the_milestone_1_state():
    state = FlightState(3, 4, 1, 5)
    assert state.cell == GridState(3, 4, 1)
    assert cell_of(state) == GridState(3, 4, 1)
    assert cell_of(GridState(3, 4, 1)) == GridState(3, 4, 1)
    assert state.has_heading
    assert not FlightState(3, 4, 1).has_heading
    assert FlightState(3, 4, 1).ih == NO_HEADING


def test_flight_state_is_hashable_and_distinguishes_headings():
    assert FlightState(1, 1, 0, 0) != FlightState(1, 1, 0, 1)
    assert len({FlightState(1, 1, 0, i) for i in range(8)}) == 8


# -- turn table --------------------------------------------------------------
@pytest.mark.parametrize("connectivity", sorted(CONNECTIVITY))
def test_turn_table_is_symmetric_zero_on_the_diagonal_and_bounded(connectivity):
    """T-10."""
    table = _table(connectivity)
    n = len(table)
    for i in range(n):
        assert table.angle_rad[i][i] == pytest.approx(0.0, abs=1e-12)
        for j in range(n):
            assert table.angle_rad[i][j] == pytest.approx(table.angle_rad[j][i])
            assert -1e-12 <= table.angle_rad[i][j] <= math.pi + 1e-12


def test_eight_connectivity_angles_are_the_expected_multiples_of_45():
    """T-10."""
    table = _table(8)
    observed = {
        round(math.degrees(table.angle_rad[0][j]), 6) for j in range(len(table))
    }
    assert observed == {0.0, 45.0, 90.0, 135.0, 180.0}


def test_sixteen_connectivity_is_not_uniformly_spaced():
    """T-11.  The knight moves sit at atan2(1, 2) = 26.565 degrees, so the
    spacing alternates 26.565 / 18.435 rather than sitting at 360 / 16 = 22.5.
    Anything that assumes ``2 pi |i - j| / K`` is wrong for this move set."""
    table = _table(16)
    bearings = sorted(
        math.degrees(math.atan2(dx, dy)) % 360.0 for dx, dy in MOVES_16
    )
    gaps = [
        round((b - a) % 360.0, 3)
        for a, b in zip(bearings, bearings[1:] + [bearings[0] + 360.0])
    ]
    assert set(gaps) == {26.565, 18.435}
    assert 22.5 not in set(gaps)
    # And the table agrees with the actual vectors rather than with any formula.
    for i, (dx_i, dy_i) in enumerate(MOVES_16):
        for j, (dx_j, dy_j) in enumerate(MOVES_16):
            expected = angle_between(
                Vec2(float(dx_i), float(dy_i)), Vec2(float(dx_j), float(dy_j))
            )
            assert table.angle_rad[i][j] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("connectivity", sorted(CONNECTIVITY))
def test_heading_wraparound_takes_the_short_way_round(connectivity):
    """T-12.  Index 0 and index K-1 are neighbours in the move ordering; their
    angle must be the small one, never 360 minus it."""
    table = _table(connectivity)
    n = len(table)
    first_last = math.degrees(table.angle_rad[0][n - 1])
    assert first_last == pytest.approx(math.degrees(table.angle_rad[n - 1][0]))
    assert first_last <= 180.0
    if connectivity == 8:
        assert first_last == pytest.approx(45.0)
    for i in range(n):
        for j in range(n):
            assert math.degrees(table.angle_rad[i][j]) <= 180.0 + 1e-9


def test_turn_table_lengths_and_lookups():
    table = _table(8, cell_size_nm=5.0)
    assert table.length_nm[table.index_of((0, 1))] == pytest.approx(5.0)
    assert table.length_nm[table.index_of((1, 1))] == pytest.approx(5.0 * math.sqrt(2))
    assert table.moves[table.nearest_index_for_bearing(90.0)] == (1, 0)
    assert table.moves[table.nearest_index_for_bearing(0.0)] == (0, 1)
    assert table.moves[table.nearest_index_for_bearing(225.0)] == (-1, -1)


# -- the gate ----------------------------------------------------------------
def _feasible_by_angle(cell_size_nm, bank_deg, connectivity=8, wind=None):
    problem = make_problem(
        cells=9,
        cell_size_nm=cell_size_nm,
        connectivity=connectivity,
        turn_model="gate+cost",
        max_bank_deg=bank_deg,
        wind=wind,
    )
    table = problem.turn_table
    node = GridState(4, 4, 0)
    out: dict[int, set[bool]] = {}
    for i in range(len(table)):
        for j in range(len(table)):
            turn = problem.cost_model.turn_metrics(
                node,
                table.track_unit(i),
                table.track_unit(j),
                table.length_nm[i],
                table.length_nm[j],
            )
            key = round(math.degrees(table.angle_rad[i][j]))
            out.setdefault(key, set()).add(turn.feasible)
    return out


def test_feasibility_frontier_at_five_three_and_twelve_nautical_miles():
    """T-22.  A 450 kt jet at 25 degrees of bank has a 6.33 NM turn radius, so a
    45 degree fly-by needs 2.62 NM of leg on each side.

    On an 8-connected grid a 45 degree turn *always* joins an axis leg to a
    diagonal leg, so the binding half-leg is half the **cell size**, never half
    the diagonal.  At 5 NM that is 2.50 NM < 2.62 NM, so at 25 degrees of bank
    no turn at all is flyable on the 5 NM grids the Milestone 1 scenarios use --
    which is exactly why they stay on ``turn_model: none``.  At 3 NM the margin
    is worse still.  At 12 NM a 45 degree turn fits comfortably.
    """
    at_5 = _feasible_by_angle(5.0, 25.0)
    assert at_5[0] == {True}
    assert at_5[45] == {False}
    assert at_5[90] == {False}

    at_3 = _feasible_by_angle(3.0, 25.0)
    assert at_3[45] == {False} and at_3[90] == {False}

    at_12 = _feasible_by_angle(12.0, 25.0)
    assert at_12[45] == {True}
    # 90 degrees is mixed at 12 NM: diagonal-to-diagonal has 8.49 NM of half-leg
    # and fits, axis-to-axis has 6.00 NM and does not.
    assert at_12[90] == {False, True}

    # More bank buys the 5 NM grid its 45 degree turns back (R falls to 5.11 NM,
    # needing 2.12 NM against the 2.50 NM available).
    assert _feasible_by_angle(5.0, 30.0)[45] == {True}


def test_a_slower_aircraft_turns_inside_a_fine_grid():
    """T-22: the frontier is about radius, not about cell size alone."""
    from atp.aircraft.performance import REGIONAL_TURBOPROP

    problem = make_problem(
        cells=9,
        cell_size_nm=3.0,
        turn_model="gate+cost",
        aircraft=REGIONAL_TURBOPROP,
        levels=(200,),
    )
    table = problem.turn_table
    turn = problem.cost_model.turn_metrics(
        GridState(4, 4, 0),
        table.track_unit(0),
        table.track_unit(1),
        table.length_nm[0],
        table.length_nm[1],
    )
    assert turn.feasible
    assert math.degrees(turn.delta_heading_rad) == pytest.approx(45.0, abs=1e-9)


def test_gate_decisions_match_the_closed_form_exactly():
    """T-23: the implementation is checked against ``R tan(dpsi/2) <= L/2``."""
    for cell_size in (4.0, 8.0, 12.0, 20.0):
        for bank in (15.0, 25.0, 35.0):
            problem = make_problem(
                cells=9,
                cell_size_nm=cell_size,
                turn_model="gate+cost",
                max_bank_deg=bank,
            )
            table = problem.turn_table
            radius = problem.cost_model.aircraft.turn_radius_nm(30000.0)
            for i in range(len(table)):
                for j in range(len(table)):
                    turn = problem.cost_model.turn_metrics(
                        GridState(4, 4, 0),
                        table.track_unit(i),
                        table.track_unit(j),
                        table.length_nm[i],
                        table.length_nm[j],
                    )
                    dpsi = table.angle_rad[i][j]
                    available = 0.5 * min(table.length_nm[i], table.length_nm[j])
                    expected = (
                        True
                        if math.degrees(dpsi) < 0.5
                        else tangent_length_nm(radius, dpsi) <= available + 1e-9
                    )
                    assert turn.feasible is expected, (
                        cell_size, bank, i, j, math.degrees(dpsi)
                    )


def test_hard_turn_cap_rejects_independently_of_geometry():
    """T-23: gate G2."""
    problem = make_problem(
        cells=9, cell_size_nm=60.0, turn_model="gate+cost", max_turn_deg=50.0
    )
    table = problem.turn_table
    node = GridState(4, 4, 0)

    def turn(i, j):
        return problem.cost_model.turn_metrics(
            node,
            table.track_unit(i),
            table.track_unit(j),
            table.length_nm[i],
            table.length_nm[j],
        )

    assert turn(0, 1).feasible  # 45 degrees, under the cap
    ninety = turn(0, 2)
    assert not ninety.feasible
    assert "exceeds cap" in ninety.infeasible_reason


def test_a_course_reversal_is_always_rejected():
    problem = make_problem(cells=9, cell_size_nm=200.0, turn_model="gate+cost")
    table = problem.turn_table
    reversal = problem.cost_model.turn_metrics(
        GridState(4, 4, 0),
        table.track_unit(0),
        table.track_unit(4),
        table.length_nm[0],
        table.length_nm[4],
    )
    assert math.degrees(table.angle_rad[0][4]) == pytest.approx(180.0)
    assert not reversal.feasible


# -- successors --------------------------------------------------------------
def test_every_generated_successor_satisfies_the_gate():
    """T-13: exhaustive over a small grid."""
    problem = make_problem(cells=6, cell_size_nm=12.0, turn_model="gate+cost")
    table = problem.turn_table
    for ix in range(6):
        for iy in range(6):
            for ih in list(range(len(table))) + [NO_HEADING]:
                state = FlightState(ix, iy, 0, ih)
                for transition in problem.successors(state):
                    successor = transition.state
                    assert isinstance(successor, FlightState)
                    assert transition.cost >= 0.0
                    if ih == NO_HEADING:
                        continue
                    turn = problem.cost_model.turn_metrics(
                        cell_of(state),
                        table.track_unit(ih),
                        table.track_unit(successor.ih),
                        table.length_nm[ih],
                        table.length_nm[successor.ih],
                    )
                    assert turn.feasible


def test_accepted_successors_are_a_subsequence_of_the_milestone_1_ones():
    """T-31: the gate skips, it never reorders.  This is half the determinism
    guarantee -- the other half is the heap tie-breaking in astar.py."""
    m1 = make_problem(cells=6, cell_size_nm=12.0, turn_model="none")
    m2 = make_problem(cells=6, cell_size_nm=12.0, turn_model="gate+cost")
    for ih in range(len(m2.turn_table)):
        state = FlightState(2, 2, 0, ih)
        m1_cells = [cell_of(t.state) for t in m1.successors(GridState(2, 2, 0))]
        m2_cells = [cell_of(t.state) for t in m2.successors(state)]
        iterator = iter(m1_cells)
        assert all(cell in iterator for cell in m2_cells), (ih, m2_cells, m1_cells)


def test_a_free_start_heading_is_unconstrained_and_free():
    """T-15.  The sentinel state has no incoming leg, so no gate and no charge --
    even at a cell size where nothing else can turn."""
    problem = make_problem(cells=6, cell_size_nm=5.0, turn_model="gate+cost")
    start = problem.initial_state()
    assert start == FlightState(0, 0, 0, NO_HEADING)
    transitions = list(problem.successors(start))
    assert len(transitions) == 3  # the three in-bounds moves from a corner cell
    for transition in transitions:
        assert transition.metrics.turn_time_h == 0.0
        assert transition.metrics.num_turns == 0


def test_a_declared_start_heading_is_honoured():
    problem = make_problem(
        cells=6, cell_size_nm=12.0, turn_model="gate+cost", start_heading=1
    )
    assert problem.initial_state() == FlightState(0, 0, 0, 1)
    assert problem.start_move == (1, 1)
    with pytest.raises(ValueError):
        make_problem(cells=6, turn_model="gate+cost", start_heading=99)


def test_pure_level_change_preserves_heading_and_charges_no_turn():
    """T-15.  A manoeuvre with no horizontal displacement has no ground track,
    hence no corner: the heading is carried through and the gate is skipped."""
    problem = make_problem(
        cells=6,
        cell_size_nm=5.0,
        levels=(300, 320),
        turn_model="gate+cost",
        start_heading=0,
    )
    problem.allow_pure_level_change = True
    state = FlightState(2, 2, 0, 3)
    pure = [
        t
        for t in problem.successors(state)
        if t.state.ix == state.ix and t.state.iy == state.iy
    ]
    assert pure, "expected a pure level change"
    for transition in pure:
        assert transition.state.ih == 3
        assert transition.metrics.turn_time_h == 0.0
        assert transition.metrics.heading_change_rad == 0.0


def test_an_unflyable_incoming_track_is_reported_not_skipped():
    """T-21 at the node level: if the crosswind at the node exceeds TAS for the
    incoming track, the corner has no defined heading."""
    from atp.environment.wind import UniformWind

    problem = make_problem(
        cells=6,
        cell_size_nm=12.0,
        turn_model="gate+cost",
        wind=UniformWind(Vec2(0.0, 900.0)),
    )
    table = problem.turn_table
    turn = problem.cost_model.turn_metrics(
        GridState(2, 2, 0),
        table.track_unit(1),
        table.track_unit(2),
        table.length_nm[1],
        table.length_nm[2],
    )
    assert not turn.feasible
    assert "unflyable at node" in turn.infeasible_reason


def test_goal_heading_constraint_is_enforced():
    problem = make_problem(
        cells=6, cell_size_nm=12.0, turn_model="gate+cost", goal_heading=1
    )
    goal_cell = GridState(5, 5, 0)
    assert problem.is_goal(FlightState(5, 5, 0, 1))
    assert not problem.is_goal(FlightState(5, 5, 0, 2))
    assert not problem.is_goal(FlightState(5, 5, 0, NO_HEADING))
    assert goal_cell == problem.goal.state


def test_turn_model_none_keeps_grid_states_and_milestone_1_successors():
    problem = make_problem(cells=6, cell_size_nm=5.0, turn_model="none")
    assert problem.initial_state() == GridState(0, 0, 0)
    assert not problem.heading_aware
    assert problem.start_move is None
    for transition in problem.successors(GridState(2, 2, 0)):
        assert isinstance(transition.state, GridState)
        assert not isinstance(transition.state, FlightState)


def test_unknown_turn_model_is_rejected():
    with pytest.raises(ValueError, match="turn_model"):
        make_problem(turn_model="sometimes")


def test_moves_8_ordering_is_unchanged():
    """Determinism depends on this tuple; pin it."""
    assert MOVES_8 == (
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
    )
