"""Validated request and response models for the presentation API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..planning.cost import TURN_MODELS

class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario: str | None = None
    start: tuple[int, int, int] | None = None
    goal: tuple[int, int, int] | None = None
    start_airport: str | None = None
    goal_airport: str | None = None
    reference_airport: str | None = None
    mode: Literal["static", "dynamic"] = "static"
    algorithm: Literal["astar", "dijkstra"] = "astar"
    heuristic: Literal["zero", "dynamic-optimistic"] = "zero"
    speed_set_kt: tuple[float, ...] | None = None
    turn_model: str = "none"
    temporal_resolution_h: float = Field(default=0.5, gt=0)
    wind_amplitude_kt: float = Field(default=0.0, ge=0)
    wind_period_h: float = Field(default=4.0, gt=0)
    cells: int = Field(default=3, ge=2)
    max_expansions: int | None = Field(default=None, gt=0)
    aircraft_source: Literal["synthetic", "openap"] = "synthetic"
    aircraft_model: str | None = None
    weather_source: Literal["synthetic", "snapshot"] = "synthetic"

    @model_validator(mode="after")
    def validate_combination(self) -> "PlanRequest":
        if self.algorithm == "dijkstra" and self.heuristic != "zero":
            raise ValueError("dijkstra requires the zero heuristic")
        if self.mode == "static" and self.heuristic == "dynamic-optimistic":
            raise ValueError("dynamic-optimistic requires dynamic mode")
        if self.start_airport is not None or self.goal_airport is not None:
            if self.start_airport is None or self.goal_airport is None:
                raise ValueError("start_airport and goal_airport must be provided together")
            if self.start is not None or self.goal is not None:
                raise ValueError("use either integer grid coordinates or airport codes, not both")
        if self.speed_set_kt is not None:
            if not self.speed_set_kt or any(speed <= 0 for speed in self.speed_set_kt):
                raise ValueError("speed_set_kt must contain positive speeds")
            if len(set(self.speed_set_kt)) != len(self.speed_set_kt):
                raise ValueError("speed_set_kt must not contain duplicates")
        if self.turn_model not in TURN_MODELS:
            raise ValueError(f"unknown turn model {self.turn_model!r}")
        if self.aircraft_source == "openap" and not self.aircraft_model:
            raise ValueError("aircraft_model is required when aircraft_source is openap")
        if self.aircraft_source == "synthetic" and self.aircraft_model is not None:
            raise ValueError("aircraft_model is only valid with aircraft_source openap")
        return self


class ExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "api-experiment"
    cells: tuple[int, ...] = (3,)
    time_steps_h: tuple[float, ...] = (0.5,)
    wind_amplitudes_kt: tuple[float, ...] = (0.0,)
    wind_periods_h: tuple[float, ...] = (4.0,)
    speed_sets_kt: tuple[tuple[float, ...], ...] = ((330.0,),)
    turn_models: tuple[str, ...] = ("none",)
    modes: tuple[str, ...] = ("dynamic",)
    heuristics: tuple[str, ...] = ("zero", "dynamic-optimistic")
    seed: int = 0
    max_expansions: int | None = Field(default=None, gt=0)
    family: str = "MATRIX"

    @model_validator(mode="after")
    def validate_axes(self) -> "ExperimentRequest":
        if any(size < 2 for size in self.cells):
            raise ValueError("cells values must be at least 2")
        if any(step <= 0 for step in self.time_steps_h):
            raise ValueError("time_steps_h values must be positive")
        if any(amplitude < 0 for amplitude in self.wind_amplitudes_kt):
            raise ValueError("wind amplitudes must be non-negative")
        if any(period <= 0 for period in self.wind_periods_h):
            raise ValueError("wind periods must be positive")
        return self
