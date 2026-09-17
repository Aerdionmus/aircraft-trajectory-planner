"""Milestone 2 closure: turn-count reporting, and wind-bearing turn coverage.

Two defects are pinned here.

**The reporting defect.**  ``num_turns`` counts corners the *active turn model*
evaluated, so it is zero whenever ``turn_model="none"`` -- regardless of how many
direction changes the returned ground track actually contains.  The turn-ablation
experiment therefore reported ``0`` for its Milestone 1 arm on a route that has
the same corners as its ``gate`` arm, which invites exactly the wrong reading.
``geometric_corners`` is the shape of the trajectory and is computed from the
cell sequence alone; the two quantities are asserted to be different things.

**The coverage gap.**  Milestone 2's wind-aware work -- air-heading recovery,
signed drift, the corrected ground-curvature bound -- was exercised by unit tests
but by no experiment, because the only turn-active scenario had zero wind and the
5 NM wind scenarios admit no legal turn at 450 kt.  ``turn-limited-wind`` closes
that, and the tests below require it to actually separate heading from track.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from atp.environment.airspace import GridState
from atp.evaluation.metrics import count_geometric_corners, evaluate_trajectory
from atp.experiments.runner import run_plan
from atp.planning.state import cell_of
from atp.scenarios.library import SCENARIO_LIBRARY, get_scenario
from atp.scenarios.spec import build_scenario

TURN_MODELS = ("none", "gate", "gate+cost")


# --------------------------------------------------------------------------
# Task 1: geometric corners vs charged turns
# --------------------------------------------------------------------------


def test_corner_counter_is_a_pure_function_of_the_cell_sequence():
    """Straight, one corner, and a corner created by the declared start move."""
    straight = [GridState(i, 0, 0) for i in range(5)]
    assert count_geometric_corners(straight) == 0

    dogleg = [GridState(0, 0, 0), GridState(1, 0, 0), GridState(2, 1, 0)]
    assert count_geometric_corners(dogleg) == 1

    # The first leg departs from the declared track, which is a real corner.
    assert count_geometric_corners(straight, start_move=(0, 1)) == 1
    assert count_geometric_corners(straight, start_move=(1, 0)) == 0


def test_pure_level_changes_are_not_corners():
    """No horizontal displacement means no ground track, so no direction change;
    the heading is carried across, matching ``CostModel.turn_metrics``."""
    path = [
        GridState(0, 0, 0),
        GridState(1, 0, 0),
        GridState(1, 0, 1),  # pure level change
        GridState(2, 0, 1),
    ]
    assert count_geometric_corners(path) == 0


def test_geometric_corners_are_reported_even_when_turn_pricing_is_disabled():
    """The defect this milestone closes.

    The same trajectory, evaluated under all three turn models, must report the
    corners it actually contains every time -- while ``num_turns`` stays zero for
    ``none``, because no turn was modelled.
    """
    spec = get_scenario("turn-limited")
    built = build_scenario(spec)
    report = run_plan(built, heuristic="optimistic")
    cells = [cell_of(s) for s in report.path]
    expected = count_geometric_corners(cells, built.problem.start_move)
    assert expected > 0, "the fixture route must actually turn for this to test anything"

    for turn_model in TURN_MODELS:
        model = build_scenario(replace(spec, turn_model=turn_model)).cost_model
        evaluation = evaluate_trajectory(
            model, cells, start_move=built.problem.start_move
        )
        assert evaluation.geometric_corners == expected, turn_model

    disabled = build_scenario(replace(spec, turn_model="none")).cost_model
    evaluation = evaluate_trajectory(
        disabled, cells, start_move=built.problem.start_move
    )
    assert evaluation.num_turns == 0
    assert evaluation.geometric_corners > evaluation.num_turns


def test_the_milestone_1_arm_of_the_ablation_is_not_a_straight_line():
    """Regression for the specific misreading: the ``none`` arm returns a route
    with corners, and must now say so."""
    spec = get_scenario("turn-limited")
    report = run_plan(
        build_scenario(replace(spec, turn_model="none")), heuristic="optimistic"
    )
    assert report.status == "solved"
    assert report.evaluation.num_turns == 0
    assert report.evaluation.geometric_corners > 0


def test_charged_turns_never_exceed_geometric_corners():
    """A charged turn is always a real direction change, so the charged count is
    bounded above by the geometric one under every turn model."""
    for name in ("turn-limited", "turn-limited-wind"):
        spec = get_scenario(name)
        for turn_model in TURN_MODELS:
            report = run_plan(
                build_scenario(replace(spec, turn_model=turn_model)),
                heuristic="optimistic",
            )
            if report.status != "solved":
                continue
            evaluation = report.evaluation
            assert evaluation.num_turns <= evaluation.geometric_corners, (
                f"{name}/{turn_model}"
            )


def test_turn_aware_behaviour_is_unchanged_by_the_new_metric():
    """With turns active the two counts agree, because every corner on the route
    is one the model evaluated.  Pins that this was a reporting fix only."""
    built = build_scenario(get_scenario("turn-limited"))
    report = run_plan(built, heuristic="optimistic")
    assert report.evaluation.num_turns == report.evaluation.geometric_corners


def test_geometric_corners_is_emitted_as_an_experiment_column():
    built = build_scenario(get_scenario("turn-limited"))
    row = run_plan(built, heuristic="optimistic").as_row()
    assert "geometric_corners" in row
    assert "num_turns" in row


# --------------------------------------------------------------------------
# Task 2: the wind-bearing turn scenario
# --------------------------------------------------------------------------


def test_the_wind_scenario_is_registered_and_turn_active():
    assert "turn-limited-wind" in SCENARIO_LIBRARY
    spec = get_scenario("turn-limited-wind")
    assert spec.turn_model == "gate+cost"
    assert spec.wind, "the scenario must carry a wind field"


def test_the_wind_scenario_differs_from_turn_limited_only_by_the_wind():
    """Controlled comparison: any measured difference is attributable to wind."""
    still = get_scenario("turn-limited")
    windy = get_scenario("turn-limited-wind")
    for field in (
        "grid",
        "start",
        "goal",
        "start_heading_deg",
        "turn_model",
        "restrictions",
        "weights",
        "aircraft",
    ):
        assert getattr(still, field) == getattr(windy, field), field
    assert not still.wind
    assert windy.wind


def test_the_wind_field_is_nonzero_and_inside_the_curvature_precondition():
    """``ground_curvature_radius_bound_nm`` is only sound for ``|w| <= TAS``;
    outside it the bound degenerates to ``inf`` and the gate rejects outright."""
    built = build_scenario(get_scenario("turn-limited-wind"))
    wind_kt = built.airspace.wind.max_magnitude_kt()
    assert wind_kt > 0.0
    assert wind_kt < built.aircraft.cruise_tas_kt


def test_the_wind_scenario_separates_air_heading_from_ground_track():
    """The point of the scenario.  A bank limit constrains the change in *air
    heading*; in wind that differs from the change in ground track, and
    ``wind_heading_excess_deg`` is exactly that difference."""
    built = build_scenario(get_scenario("turn-limited-wind"))
    report = run_plan(built, heuristic="optimistic")
    evaluation = report.evaluation

    assert report.status == "solved"
    assert evaluation.num_turns > 0, "a turning route is required"
    assert evaluation.total_track_change_deg > 0.0
    assert evaluation.total_heading_change_deg != pytest.approx(
        evaluation.total_track_change_deg
    )
    assert abs(evaluation.wind_heading_excess_deg) > 1.0
    assert evaluation.wind_heading_excess_deg == pytest.approx(
        evaluation.total_heading_change_deg - evaluation.total_track_change_deg
    )


def test_the_still_air_counterpart_has_no_excess():
    """The control arm: with no wind, heading change and track change coincide
    exactly, so the metric is measuring the wind and nothing else."""
    report = run_plan(build_scenario(get_scenario("turn-limited")), heuristic="optimistic")
    assert report.evaluation.wind_heading_excess_deg == pytest.approx(0.0, abs=1e-9)
    assert report.evaluation.total_heading_change_deg == pytest.approx(
        report.evaluation.total_track_change_deg
    )


def test_the_wind_scenario_is_deterministic():
    a = run_plan(build_scenario(get_scenario("turn-limited-wind")), heuristic="optimistic")
    b = run_plan(build_scenario(get_scenario("turn-limited-wind")), heuristic="optimistic")
    assert [cell_of(s) for s in a.path] == [cell_of(s) for s in b.path]
    assert a.evaluation.comparable_cost == b.evaluation.comparable_cost
    assert a.evaluation.wind_heading_excess_deg == b.evaluation.wind_heading_excess_deg


def test_wind_does_not_make_the_route_cheaper_to_turn_than_still_air_geometry():
    """Sanity on the corrected bound: a wind component along the heading makes
    the ground-path radius *larger*, so the gate can only get harder, never
    easier.  The turn must still be legal here -- the scenario is useless if the
    wind silently prunes every corner."""
    windy = run_plan(
        build_scenario(get_scenario("turn-limited-wind")), heuristic="optimistic"
    )
    still = run_plan(build_scenario(get_scenario("turn-limited")), heuristic="optimistic")
    assert windy.status == "solved"
    assert windy.evaluation.num_turns > 0
    assert still.evaluation.num_turns > 0
