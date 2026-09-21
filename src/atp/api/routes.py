"""HTTP route definitions for the presentation API."""

from __future__ import annotations

from fastapi import APIRouter

from .models import ExperimentRequest, PlanRequest
from .services import (
    list_scenarios,
    plan,
    run_experiment,
    scenario_detail,
)

router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "aircraft-trajectory-planner-api"}


@router.get("/scenarios")
def scenarios() -> list[dict[str, object]]:
    return list_scenarios()


@router.get("/scenarios/{scenario_name}")
def scenario(scenario_name: str) -> dict[str, object]:
    return scenario_detail(scenario_name)


@router.post("/plan")
def create_plan(request: PlanRequest) -> dict[str, object]:
    return plan(request)


@router.post("/experiments/run")
def create_experiment(request: ExperimentRequest) -> dict[str, object]:
    return run_experiment(request)


@router.get("/experiments/config")
def experiment_config() -> dict[str, object]:
    return {
        "families": [
            "STATIC_REGRESSION",
            "DYNAMIC_REFERENCE",
            "DYNAMIC_HEURISTIC",
            "TIME_STEP_SENSITIVITY",
            "WIND_SENSITIVITY",
            "SPEED_INTERACTION",
            "TURN_INTERACTION",
            "SCALING",
        ],
        "defaults": ExperimentRequest().model_dump(),
    }

