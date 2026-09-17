"""T-3, T-30, T-32, T-34 to T-36: scenarios, experiments and baseline fairness.

The Milestone 1 audit defects stay pinned in ``tests/test_audit_regressions.py``;
what is added here is a re-check of the two that turn dynamics can reach -- the
heuristic lower bounds and the partially-priced-total rule -- under an active
turn model.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from atp.cli import main
from atp.environment.airspace import GridState
from atp.evaluation.metrics import evaluate_trajectory
from atp.experiments.runner import (
    NON_DETERMINISTIC_COLUMNS,
    ExperimentSpec,
    run_baseline,
    run_matrix,
    run_plan,
    summarise,
    write_results,
)
from atp.scenarios.feasibility import is_goal_reachable
from atp.scenarios.library import SCENARIO_LIBRARY, get_scenario, random_scenario
from atp.scenarios.spec import (
    DEFAULT_V2_TURN_MODEL,
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    ScenarioError,
    ScenarioSpec,
    build_scenario,
)

CONFIGS = Path(__file__).parent.parent / "configs"


# -- schema ------------------------------------------------------------------
def test_schema_version_two_is_current_and_one_is_still_supported():
    assert SCHEMA_VERSION == 2
    assert SUPPORTED_SCHEMA_VERSIONS == (1, 2)


@pytest.mark.parametrize(
    "name", ["scenario_two_no_fly.json", "scenario_convective_risk.json"]
)
def test_version_one_documents_migrate_to_turn_model_none(name):
    """T-34.  A document written before turn dynamics existed cannot be assumed
    to have been designed for them, so it is migrated conservatively."""
    path = CONFIGS / name
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 1
    assert "turn_model" not in raw

    spec = ScenarioSpec.load(path)
    assert spec.turn_model == "none"
    assert spec.max_bank_deg is None
    assert spec.start_heading_deg is None
    built = build_scenario(spec)
    assert not built.problem.heading_aware
    assert isinstance(built.problem.initial_state(), GridState)


def test_a_document_with_no_version_at_all_is_treated_as_version_one():
    """T-34: silently switching turns on underneath an unversioned document
    would be the wrong default."""
    doc = json.loads(get_scenario("empty-cruise").to_json())
    doc.pop("schema_version")
    assert ScenarioSpec.from_dict(doc).turn_model == "none"


def test_a_version_two_document_defaults_to_the_full_turn_model():
    doc = json.loads(get_scenario("empty-cruise").to_json())
    doc["schema_version"] = 2
    doc.pop("turn_model")
    assert ScenarioSpec.from_dict(doc).turn_model == DEFAULT_V2_TURN_MODEL


def test_turn_limited_config_round_trips_as_version_two():
    spec = ScenarioSpec.load(CONFIGS / "scenario_turn_limited.json")
    assert spec.schema_version == 2
    assert spec.turn_model == "gate+cost"
    assert spec.start_heading_deg == 90.0
    assert ScenarioSpec.from_json(spec.to_json()) == spec


def test_unknown_keys_are_still_rejected_rather_than_ignored():
    """T-35: the Milestone 1 guarantee must survive the new keys."""
    doc = json.loads(get_scenario("turn-limited").to_json())
    doc["turn_modle"] = "gate"
    with pytest.raises(ScenarioError, match="unknown key"):
        ScenarioSpec.from_dict(doc)


def test_unsupported_schema_versions_and_turn_models_are_rejected():
    """T-35."""
    doc = json.loads(get_scenario("empty-cruise").to_json())
    doc["schema_version"] = 99
    with pytest.raises(ScenarioError, match="schema version"):
        ScenarioSpec.from_dict(doc)

    doc = json.loads(get_scenario("turn-limited").to_json())
    doc["turn_model"] = "sometimes"
    with pytest.raises(ScenarioError, match="turn_model"):
        ScenarioSpec.from_dict(doc)


def test_out_of_range_bank_and_turn_caps_are_rejected_at_build_time():
    spec = get_scenario("turn-limited")
    with pytest.raises(ScenarioError, match="max_bank_deg"):
        build_scenario(replace(spec, max_bank_deg=95.0))
    with pytest.raises(ScenarioError, match="max_turn_deg"):
        build_scenario(replace(spec, max_turn_deg=0.0))


def test_headings_are_snapped_to_actual_grid_moves():
    spec = replace(get_scenario("turn-limited"), start_heading_deg=44.0)
    built = build_scenario(spec)
    assert built.problem.start_move == (1, 1)  # 045, the nearest 8-connected move
    assert built.problem.start_heading_index is not None


# -- reachability ------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(SCENARIO_LIBRARY))
def test_every_library_scenario_is_reachable_under_its_declared_turn_model(name):
    """T-36.  Turn dynamics can make an instance unsolvable; if a shipped
    scenario is affected that must be a declared fact, not a surprise."""
    built = build_scenario(get_scenario(name))
    assert is_goal_reachable(built.problem, max_states=400000)


def test_reachability_floods_the_heading_augmented_graph():
    """T-36: the flood must use the states the planner uses, or its answer is
    about a different problem."""
    built = build_scenario(get_scenario("turn-limited"))
    assert built.problem.heading_aware
    assert is_goal_reachable(built.problem, max_states=400000)
    # A bank limit tight enough to forbid every turn walls the goal off.
    frozen = build_scenario(
        replace(get_scenario("turn-limited"), max_bank_deg=5.0)
    )
    assert not is_goal_reachable(frozen.problem, max_states=200000)


def test_random_scenarios_are_unaffected_and_still_reachable():
    for seed in (101, 202):
        spec = random_scenario(seed)
        assert spec.turn_model == "none"
        assert is_goal_reachable(build_scenario(spec).problem, max_states=400000)


# -- audit regressions re-checked under turns --------------------------------
def test_infeasible_trajectories_still_report_infinite_comparable_cost():
    """T-3 (audit defect 4) with a turn model active: a trajectory that violates
    a hard constraint -- including a turn it cannot fly -- must not be scored on
    its feasible segments alone."""
    built = build_scenario(get_scenario("turn-limited"))
    report = run_baseline(built)
    assert not report.evaluation.feasible
    assert report.evaluation.unpriced_segments > 0
    assert math.isinf(report.evaluation.comparable_cost)
    assert report.evaluation.cost.total < math.inf  # the partial total still exists


def test_a_trajectory_with_an_unflyable_turn_is_flagged_by_the_evaluator():
    """T-3/T-32: the evaluator, not the planner, is what catches this."""
    tight = build_scenario(
        replace(get_scenario("turn-limited"), max_bank_deg=5.0)
    )
    zigzag = [
        GridState(2, 20, 0),
        GridState(3, 20, 0),
        GridState(4, 21, 0),
        GridState(5, 20, 0),
    ]
    evaluation = evaluate_trajectory(tight.cost_model, zigzag)
    assert not evaluation.feasible
    assert evaluation.unpriced_segments >= 1
    assert math.isinf(evaluation.comparable_cost)


# -- baseline fairness -------------------------------------------------------
def test_the_rasterised_baseline_is_charged_for_its_own_staircase():
    """T-32.  A Bresenham "straight" line is a staircase, so under a turn model
    it acquires corners that a straight flight would not have.  Those charges
    are real given the path, but they are a rasterisation artefact, which is why
    the analytic reference exists alongside it."""
    spec = replace(
        get_scenario("turn-limited"),
        start=[2, 4, 0],
        goal=[37, 30, 0],
        restrictions=[],
    )
    built = build_scenario(spec)
    rasterised = run_baseline(built)
    analytic = run_baseline(build_scenario(spec), analytic=True)

    assert rasterised.planner == "direct-route"
    assert analytic.planner == "analytic-direct"
    assert rasterised.evaluation.num_turns > 0
    assert analytic.evaluation.num_turns == 0
    assert analytic.evaluation.num_waypoints == 2
    # Both are scored by the same code path and both are feasible here.
    assert rasterised.evaluation.feasible and analytic.evaluation.feasible
    assert rasterised.evaluation.turn_time_min > analytic.evaluation.turn_time_min


def test_both_baselines_use_the_identical_evaluation_path():
    """T-32: a baseline must never be scored by code a planner does not share."""
    built = build_scenario(get_scenario("empty-cruise"))
    report = run_baseline(built)
    recomputed = evaluate_trajectory(
        built.cost_model, report.path, start_move=built.problem.start_move
    )
    assert recomputed.cost.total == pytest.approx(report.evaluation.cost.total)
    assert recomputed.distance_nm == pytest.approx(report.evaluation.distance_nm)


def test_the_analytic_reference_is_no_longer_than_the_rasterised_one():
    spec = replace(get_scenario("empty-cruise"))
    rasterised = run_baseline(build_scenario(spec))
    analytic = run_baseline(build_scenario(spec), analytic=True)
    assert analytic.evaluation.distance_nm <= rasterised.evaluation.distance_nm + 1e-9


# -- experiments -------------------------------------------------------------
def _deterministic_rows(reports):
    return [
        {k: v for k, v in r.as_row().items() if k not in NON_DETERMINISTIC_COLUMNS}
        for r in reports
    ]


def test_the_turn_ablation_matrix_is_deterministic():
    """T-30."""
    spec = ExperimentSpec.load(CONFIGS / "experiment_turn_ablation.json")
    assert spec.turn_models == ["none", "gate", "gate+cost"]
    first = _deterministic_rows(run_matrix(spec))
    second = _deterministic_rows(run_matrix(spec))
    assert first == second
    assert len(first) > 10


def test_the_bank_frontier_matrix_is_deterministic_and_spans_its_axis():
    """T-30 and the bank axis."""
    spec = ExperimentSpec.load(CONFIGS / "experiment_bank_frontier.json")
    rows = _deterministic_rows(run_matrix(spec))
    assert rows == _deterministic_rows(run_matrix(spec))
    assert len({row["scenario"] for row in rows}) == len(spec.bank_angles_deg)


def test_ablation_axes_are_inert_when_unset():
    """An unmodified Milestone 1 experiment document must produce an unmodified
    Milestone 1 matrix: the axes default to empty, not to a list of one."""
    spec = ExperimentSpec(
        name="inert", scenarios=["empty-cruise"], heuristics=["optimistic"]
    )
    assert spec.turn_models == [] and spec.bank_angles_deg == []
    reports = run_matrix(spec)
    assert [r.scenario for r in reports] == ["empty-cruise", "empty-cruise"]
    assert all(r.turn_model == "none" for r in reports)


def test_the_turn_model_is_recorded_on_every_row():
    spec = ExperimentSpec.load(CONFIGS / "experiment_turn_ablation.json")
    rows = [r.as_row() for r in run_matrix(spec)]
    assert {row["turn_model"] for row in rows} == {"none", "gate", "gate+cost"}
    assert all("num_turns" in row for row in rows)
    assert all("wind_heading_excess_deg" in row for row in rows)


def test_turns_only_appear_when_the_turn_model_is_active():
    spec = ExperimentSpec.load(CONFIGS / "experiment_turn_ablation.json")
    by_model: dict[str, list] = {}
    for report in run_matrix(spec):
        if report.planner == "astar" and report.status == "solved":
            by_model.setdefault(report.turn_model, []).append(report.evaluation)
    assert all(e.num_turns == 0 for e in by_model["none"])
    assert all(e.turn_time_min == 0.0 for e in by_model["gate"])
    assert any(e.turn_time_min > 0.0 for e in by_model["gate+cost"])


def test_results_files_carry_the_new_columns(tmp_path):
    spec = ExperimentSpec.load(CONFIGS / "experiment_turn_ablation.json")
    reports = run_matrix(spec)
    csv_path, json_path = write_results(reports, tmp_path, name="m2")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for column in (
        "turn_model",
        "num_turns",
        "total_heading_change_deg",
        "max_heading_change_deg",
        "wind_heading_excess_deg",
        "turn_time_min",
        "turn_fuel_kg",
        "corner_cut_nm",
    ):
        assert column in rows[0]
    document = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(document["results"]) == len(reports)


def test_experiment_spec_still_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown key"):
        ExperimentSpec.from_dict({"turn_modles": ["gate"]})


def test_summary_table_reports_turns():
    built = build_scenario(get_scenario("turn-limited"))
    text = summarise([run_plan(built, heuristic="optimistic")])
    assert "turn" in text
    assert "hdg_deg" in text
    assert "gate+cost" in text


# -- CLI ---------------------------------------------------------------------
def test_cli_plan_accepts_the_turn_overrides(capsys):
    code = main(
        [
            "plan",
            "--scenario",
            "two-no-fly",
            "--turn-model",
            "gate+cost",
            "--bank-deg",
            "30",
            "--with-baseline",
            "--with-analytic-baseline",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "analytic-direct" in out
    assert "gate+cost" in out


def test_cli_plan_without_overrides_is_milestone_1(capsys):
    code = main(["plan", "--scenario", "two-no-fly"])
    out = capsys.readouterr().out
    assert code == 0
    assert "none" in out


def test_cli_rejects_an_unknown_turn_model():
    with pytest.raises(SystemExit):
        main(["plan", "--scenario", "empty-cruise", "--turn-model", "sometimes"])


def test_cli_experiment_runs_the_ablation_config(tmp_path, capsys):
    code = main(
        [
            "experiment",
            "--config",
            str(CONFIGS / "experiment_turn_ablation.json"),
            "--output-dir",
            str(tmp_path),
        ]
    )
    capsys.readouterr()
    assert code == 0
    assert (tmp_path / "turn-ablation.csv").exists()
    assert (tmp_path / "turn-ablation.json").exists()
