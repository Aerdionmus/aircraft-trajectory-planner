from __future__ import annotations

import json

import pytest

import atp.experiments.dynamic as dynamic_experiments
from atp.experiments.dynamic import (
    DynamicExperimentSpec,
    run_dynamic_experiment,
    run_dynamic_matrix,
    run_scaling_experiments,
    write_dynamic_results,
)


def small_spec() -> DynamicExperimentSpec:
    return DynamicExperimentSpec(
        name="small",
        cells=(3,),
        time_steps_h=(0.25, 0.5),
        wind_amplitudes_kt=(0.0, 10.0),
        wind_periods_h=(2.0,),
        speed_sets_kt=((330.0,),),
        turn_models=("none",),
        modes=("dynamic",),
        heuristics=("zero", "dynamic-optimistic"),
        seed=17,
    )


def test_dynamic_matrix_is_deterministic_and_has_stable_schema():
    first = run_dynamic_matrix(small_spec())
    second = run_dynamic_matrix(small_spec())
    assert [{k: v for k, v in row.items() if k != "runtime_s"} for row in first] == [
        {k: v for k, v in row.items() if k != "runtime_s"} for row in second
    ]
    required = {
        "scenario_name",
        "mode",
        "heuristic",
        "algorithm",
        "temporal_resolution_h",
        "wind_amplitude_kt",
        "wind_period_h",
        "speed_set_kt",
        "turn_model",
        "solved",
        "objective_cost",
        "distance_nm",
        "elapsed_time_h",
        "planned_time_h",
        "fuel_kg",
        "risk_exposure",
        "expansions",
        "generated_states",
        "runtime_s",
        "final_state",
        "final_time_bucket",
        "path_length",
        "speed_changes",
        "heading_transitions",
    }
    assert required <= set(first[0])


def test_dynamic_heuristic_matches_zero_reference():
    kwargs = dict(
        name="reference",
        mode="dynamic",
        cells=3,
        time_step_h=0.5,
        amplitude_kt=10.0,
        period_h=2.0,
        speeds_kt=(330.0,),
        turn_model="none",
        seed=1,
    )
    zero = run_dynamic_experiment(**kwargs, algorithm="dijkstra", heuristic="zero")
    guided = run_dynamic_experiment(
        **kwargs, algorithm="astar", heuristic="dynamic-optimistic"
    )
    assert zero["solved"] and guided["solved"]
    assert guided["objective_cost"] == pytest.approx(zero["objective_cost"])


def test_astar_zero_and_all_matrix_sensitivity_axes_execute():
    spec = DynamicExperimentSpec(
        name="axes",
        cells=(2, 3),
        time_steps_h=(0.25, 0.5),
        wind_amplitudes_kt=(0.0, 10.0),
        wind_periods_h=(2.0,),
        speed_sets_kt=((330.0,), (300.0, 330.0)),
        turn_models=("none", "gate"),
        modes=("dynamic",),
        heuristics=("zero", "dynamic-optimistic"),
        seed=5,
    )
    results = run_dynamic_matrix(spec)
    assert {row["algorithm"] for row in results} == {"dijkstra", "astar"}
    assert {"zero", "dynamic-optimistic"} <= {
        row["heuristic"] for row in results
    }
    assert {row["grid"]["cells_x"] for row in results} == {2, 3}
    assert {row["temporal_resolution_h"] for row in results} == {0.25, 0.5}
    assert {row["wind_amplitude_kt"] for row in results} == {0.0, 10.0}
    assert {tuple(row["speed_set_kt"]) for row in results} == {
        (330.0,),
        (300.0, 330.0),
    }
    assert {row["turn_model"] for row in results} == {"none", "gate"}


def test_temporal_resolution_and_seed_are_explicit_and_reproducible():
    kwargs = dict(
        name="resolution",
        mode="dynamic",
        cells=3,
        time_step_h=0.25,
        amplitude_kt=0.0,
        period_h=2.0,
        speeds_kt=(330.0,),
        turn_model="none",
        algorithm="astar",
        heuristic="zero",
        seed=99,
    )
    first = run_dynamic_experiment(**kwargs)
    second = run_dynamic_experiment(**kwargs)
    assert first["seed"] == 99
    assert {k: v for k, v in first.items() if k != "runtime_s"} == {
        k: v for k, v in second.items() if k != "runtime_s"
    }
    assert first["temporal_resolution_h"] == 0.25


def test_infeasible_or_limited_runs_are_not_reported_as_success():
    result = run_dynamic_experiment(
        name="limited",
        mode="dynamic",
        cells=8,
        time_step_h=0.5,
        amplitude_kt=0.0,
        period_h=2.0,
        speeds_kt=(330.0,),
        turn_model="none",
        algorithm="astar",
        heuristic="zero",
        max_expansions=1,
    )
    assert result["solved"] is False
    assert result["status"] == "limit_reached"
    assert result["objective_cost"] is None
    assert result["distance_nm"] is None


def test_static_regression_run_is_supported():
    result = run_dynamic_experiment(
        name="static",
        mode="static",
        cells=3,
        time_step_h=0.5,
        amplitude_kt=0.0,
        period_h=2.0,
        speeds_kt=(330.0,),
        turn_model="none",
        algorithm="astar",
        heuristic="zero",
    )
    assert result["solved"] is True
    assert result["temporal_resolution_h"] is None
    assert result["final_time_bucket"] == 0


def test_static_regression_reuses_existing_static_runner(monkeypatch):
    calls = []
    original = dynamic_experiments.run_static_plan

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(dynamic_experiments, "run_static_plan", spy)
    result = run_dynamic_experiment(
        name="static-regression",
        mode="static",
        cells=3,
        time_step_h=0.5,
        amplitude_kt=0.0,
        period_h=2.0,
        speeds_kt=(330.0,),
        turn_model="none",
        algorithm="dijkstra",
        heuristic="zero",
    )
    assert result["family"] == "STATIC_REGRESSION"
    assert len(calls) == 1
    assert "weight" not in calls[0][1]


def test_static_algorithm_label_matches_existing_runner_behavior(monkeypatch):
    calls = []
    original = dynamic_experiments.run_static_plan

    def spy(*args, **kwargs):
        report = original(*args, **kwargs)
        calls.append(report.planner)
        return report

    monkeypatch.setattr(dynamic_experiments, "run_static_plan", spy)
    for algorithm in ("dijkstra", "astar"):
        result = run_dynamic_experiment(
            name=f"static-{algorithm}",
            mode="static",
            cells=3,
            time_step_h=0.5,
            amplitude_kt=0.0,
            period_h=2.0,
            speeds_kt=(330.0,),
            turn_model="none",
            algorithm=algorithm,
            heuristic="zero",
        )
        assert result["algorithm"] == "m3-runner"
    assert calls == ["dijkstra", "dijkstra"]


def test_static_speed_set_is_applied_to_built_scenario(monkeypatch):
    envelopes = []
    original = dynamic_experiments.run_static_plan

    def spy(built, **kwargs):
        envelopes.append(built.aircraft.speed_envelope.planning_tas_kt)
        return original(built, **kwargs)

    monkeypatch.setattr(dynamic_experiments, "run_static_plan", spy)
    for speeds in ((330.0,), (300.0, 360.0)):
        run_dynamic_experiment(
            name="static-speed",
            mode="static",
            cells=3,
            time_step_h=0.5,
            amplitude_kt=0.0,
            period_h=2.0,
            speeds_kt=speeds,
            turn_model="none",
            algorithm="dijkstra",
            heuristic="zero",
        )
    assert envelopes == [(330.0,), (300.0, 360.0)]


def test_scaling_family_executes_independent_sizes_deterministically():
    spec = DynamicExperimentSpec(
        name="scaling",
        cells=(2, 3),
        modes=("dynamic",),
        heuristics=("zero",),
        time_steps_h=(0.5,),
        wind_amplitudes_kt=(0.0,),
        wind_periods_h=(2.0,),
        speed_sets_kt=((330.0,),),
        turn_models=("none",),
    )
    first = run_scaling_experiments(spec)
    second = run_scaling_experiments(spec)
    assert len(first) == len(second) == 4
    assert {row["family"] for row in first} == {"SCALING"}
    assert {row["grid"]["cells_x"] for row in first} == {2, 3}
    strip_runtime = lambda rows: [
        {key: value for key, value in row.items() if key != "runtime_s"}
        for row in rows
    ]
    assert strip_runtime(first) == strip_runtime(second)


def test_json_output_is_machine_readable(tmp_path):
    path = write_dynamic_results(run_dynamic_matrix(small_spec()), tmp_path / "results.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    assert len(document["results"]) == len(run_dynamic_matrix(small_spec()))
    assert "grid" in document["results"][0]


def test_invalid_combinations_are_rejected_explicitly():
    kwargs = dict(
        name="invalid",
        mode="dynamic",
        cells=3,
        time_step_h=0.5,
        amplitude_kt=0.0,
        period_h=2.0,
        speeds_kt=(330.0,),
        turn_model="none",
    )
    with pytest.raises(ValueError, match="zero heuristic"):
        run_dynamic_experiment(
            **kwargs, algorithm="dijkstra", heuristic="dynamic-optimistic"
        )
    with pytest.raises(ValueError, match="dynamic mode"):
        run_dynamic_experiment(
            **kwargs | {"mode": "static"},
            algorithm="astar",
            heuristic="dynamic-optimistic",
        )
