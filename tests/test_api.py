from __future__ import annotations

import json
import math

import pytest
from fastapi.testclient import TestClient

from atp.api.app import app

pytestmark = pytest.mark.filterwarnings("ignore:Using `httpx` with `starlette.testclient`")


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_health_and_scenario_endpoints(client: TestClient):
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    scenarios = client.get("/api/scenarios")
    assert scenarios.status_code == 200
    names = {item["name"] for item in scenarios.json()}
    assert "empty-cruise" in names

    detail = client.get("/api/scenarios/empty-cruise")
    assert detail.status_code == 200
    assert {"grid", "start", "goal", "aircraft"} <= detail.json().keys()


@pytest.mark.parametrize("algorithm", ["dijkstra", "astar"])
def test_static_plan_has_frontend_safe_trajectory(
    client: TestClient, algorithm: str
):
    response = client.post(
        "/api/plan",
        json={
            "scenario": "empty-cruise",
            "mode": "static",
            "algorithm": algorithm,
            "heuristic": "zero",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "solved"
    assert body["trajectory"]
    assert body["algorithm"] == algorithm
    assert body["trajectory"][0]["speed_tas_kt"] is None


def test_static_astar_zero_is_reported_as_actual_astar(client: TestClient):
    response = client.post(
        "/api/plan",
        json={
            "scenario": "empty-cruise",
            "mode": "static",
            "algorithm": "astar",
            "heuristic": "zero",
        },
    )
    assert response.status_code == 200
    assert response.json()["algorithm"] == "astar"


@pytest.mark.parametrize(
    ("algorithm", "heuristic"),
    [("dijkstra", "zero"), ("astar", "zero"), ("astar", "dynamic-optimistic")],
)
def test_dynamic_algorithm_and_heuristic_combinations(
    client: TestClient, algorithm: str, heuristic: str
):
    response = client.post(
        "/api/plan",
        json={
            "mode": "dynamic",
            "algorithm": algorithm,
            "heuristic": heuristic,
            "cells": 2,
            "speed_set_kt": [330.0],
        },
    )
    assert response.status_code == 200
    assert response.json()["algorithm"] == algorithm
    assert response.json()["heuristic"] == heuristic


def test_invalid_combinations_and_temporal_values_are_rejected(client: TestClient):
    invalid = client.post(
        "/api/plan",
        json={"mode": "dynamic", "algorithm": "dijkstra", "heuristic": "dynamic-optimistic"},
    )
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "VALIDATION_ERROR"

    invalid_step = client.post(
        "/api/plan",
        json={"mode": "dynamic", "temporal_resolution_h": 0},
    )
    assert invalid_step.status_code == 422


def test_limit_and_experiment_responses_are_json_safe(client: TestClient):
    limited = client.post(
        "/api/plan",
        json={"mode": "dynamic", "cells": 4, "max_expansions": 1},
    )
    assert limited.status_code == 200
    assert limited.json()["status"] == "limit_reached"
    serialized = json.dumps(limited.json(), allow_nan=False)
    assert "Infinity" not in serialized

    experiments = client.post(
        "/api/experiments/run",
        json={
            "name": "api",
            "cells": [2],
            "modes": ["dynamic"],
            "heuristics": ["zero"],
            "time_steps_h": [0.5],
            "wind_amplitudes_kt": [0.0],
            "wind_periods_h": [2.0],
            "speed_sets_kt": [[330.0]],
        },
    )
    assert experiments.status_code == 200
    assert experiments.json()["results"]
    assert math.isfinite(experiments.json()["results"][0]["objective_cost"])


def test_unknown_scenario_and_experiment_config(client: TestClient):
    missing = client.get("/api/scenarios/not-a-scenario")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "SCENARIO_NOT_FOUND"

    config = client.get("/api/experiments/config")
    assert config.status_code == 200
    assert "SCALING" in config.json()["families"]
