from __future__ import annotations

import csv

import pytest

from atp.baselines.direct import _bresenham, plan_direct_route
from atp.cli import main
from atp.experiments.runner import (
    NON_DETERMINISTIC_COLUMNS,
    ExperimentSpec,
    run_baseline,
    run_matrix,
    run_plan,
    summarise,
    write_results,
)
from atp.scenarios.library import get_scenario
from atp.scenarios.spec import build_scenario
from atp.visualization.ascii_map import render_ascii


# -- baseline ---------------------------------------------------------------
def test_bresenham_is_contiguous_and_hits_both_endpoints():
    cells = _bresenham(0, 0, 9, 4)
    assert cells[0] == (0, 0) and cells[-1] == (9, 4)
    for a, b in zip(cells, cells[1:]):
        assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1


def test_direct_baseline_is_optimal_in_an_empty_airspace():
    built = build_scenario(get_scenario("empty-cruise"))
    baseline = plan_direct_route(built.problem)
    planned = run_plan(built, heuristic="optimistic")
    assert baseline.evaluation.feasible
    assert planned.evaluation.cost.total == pytest.approx(
        baseline.evaluation.cost.total, rel=1e-9
    )


def test_direct_baseline_violates_constraints_where_the_planner_does_not():
    built = build_scenario(get_scenario("two-no-fly"))
    baseline = run_baseline(built)
    planned = run_plan(build_scenario(get_scenario("two-no-fly")))
    assert not baseline.evaluation.feasible
    assert baseline.evaluation.hard_violations
    assert planned.evaluation.feasible
    # Compliance costs distance: the legal route is longer.
    assert planned.evaluation.distance_nm > baseline.evaluation.distance_nm


# -- determinism ------------------------------------------------------------
@pytest.mark.parametrize("scenario", ["two-no-fly", "convective-risk", "jetstream"])
def test_repeated_runs_are_bit_for_bit_identical(scenario):
    first = run_plan(build_scenario(get_scenario(scenario)))
    second = run_plan(build_scenario(get_scenario(scenario)))
    assert first.path == second.path
    assert first.search_cost == second.search_cost
    assert first.statistics.expansions == second.statistics.expansions
    assert first.statistics.generated == second.statistics.generated


def test_experiment_matrix_is_deterministic_apart_from_timings():
    spec = ExperimentSpec(
        name="determinism",
        scenarios=["empty-cruise"],
        random_seeds=[11],
        heuristics=["zero", "optimistic"],
    )
    rows_a = [r.as_row() for r in run_matrix(spec)]
    rows_b = [r.as_row() for r in run_matrix(spec)]
    strip = lambda rows: [
        {k: v for k, v in row.items() if k not in NON_DETERMINISTIC_COLUMNS}
        for row in rows
    ]
    assert strip(rows_a) == strip(rows_b)


# -- result serialisation ---------------------------------------------------
def test_results_are_written_as_csv_and_json(tmp_path):
    reports = run_matrix(
        ExperimentSpec(name="unit", scenarios=["empty-cruise"], heuristics=["optimistic"])
    )
    csv_path, json_path = write_results(reports, tmp_path, name="unit")
    assert csv_path.exists() and json_path.exists()
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(reports)
    assert "cost_total" in rows[0]
    assert summarise(reports).splitlines()[0].startswith("scenario")


def test_experiment_spec_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown key"):
        ExperimentSpec.from_dict({"scenarios": ["empty-cruise"], "planner": "astar"})


# -- rendering --------------------------------------------------------------
def test_ascii_render_marks_origin_destination_and_obstacles():
    built = build_scenario(get_scenario("two-no-fly"))
    report = run_plan(built)
    text = render_ascii(built.airspace, report.path, width=40, height=20)
    assert "S" in text and "G" in text and "#" in text
    body = text.splitlines()[:-1]
    assert all(len(line) == len(body[0]) for line in body)


# -- CLI --------------------------------------------------------------------
def test_cli_plan_returns_zero_and_prints_a_table(capsys):
    assert main(["plan", "--scenario", "empty-cruise", "--with-baseline"]) == 0
    assert "empty-cruise" in capsys.readouterr().out


def test_cli_scenarios_lists_the_library(capsys):
    assert main(["scenarios"]) == 0
    assert "two-no-fly" in capsys.readouterr().out


def test_cli_experiment_writes_results(tmp_path, capsys):
    config = tmp_path / "exp.json"
    config.write_text(
        '{"name": "cli-test", "scenarios": ["empty-cruise"], '
        '"heuristics": ["optimistic"], "include_baseline": false}',
        encoding="utf-8",
    )
    assert main(["experiment", "--config", str(config), "--output-dir", str(tmp_path)]) == 0
    assert (tmp_path / "cli-test.csv").exists()
    capsys.readouterr()


def test_cli_export_scenario_round_trips(tmp_path, capsys):
    out = tmp_path / "s.json"
    assert main(["export-scenario", "--scenario", "jetstream", "--output", str(out)]) == 0
    from atp.scenarios.spec import ScenarioSpec

    assert ScenarioSpec.load(out).name == "jetstream"
    capsys.readouterr()
