"""Deterministic dynamic traversal-time solver for one wind-affected segment.

This module intentionally stays independent from the planner search and A* stack:
its only responsibility is to evaluate a candidate segment under a dynamic wind
field using a fixed-point midpoint approximation and return the bucketed planner
representation.

The semantic split is explicit:

- ``tau_model_h`` is the continuous-time model estimate for the segment.
- ``tau_plan_h`` is the discrete planner duration used for bucketed state time.
- bucket quantization is a planner discretization, not aircraft waiting.
- no waiting action or continuous arrival-time state is represented here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..aircraft.kinematics import solve_ground_speed
from ..core.geometry import Vec2
from ..core.temporal import TemporalConfig
from ..environment.wind import DynamicWindField

DEFAULT_TRAVERSAL_TOLERANCE_H: float = 1e-9
DEFAULT_TRAVERSAL_MAX_ITERATIONS: int = 32
DEFAULT_TRAVERSAL_INITIAL_GUESS_H: float = 1.0


@dataclass(frozen=True, slots=True)
class ModeledTraversalResult:
    tau_model_h: float
    bucket_count: int
    tau_plan_h: float
    converged: bool
    used_fallback: bool
    rejected: bool
    iterations: int
    diagnostic: str = ""


def _validate_track_unit(track_unit: Vec2) -> Vec2:
    if not isinstance(track_unit, Vec2):
        raise ValueError("track_unit must be a Vec2")
    if not math.isfinite(track_unit.x) or not math.isfinite(track_unit.y):
        raise ValueError("track_unit components must be finite")
    norm = track_unit.norm()
    if norm <= 0.0:
        raise ValueError("track_unit must be non-zero")
    if not math.isclose(norm, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("track_unit must be a unit vector")
    return track_unit


def _check_accepted_result_invariants(
    result: "ModeledTraversalResult", *, time_step_h: float
) -> None:
    if result.rejected:
        return
    if not math.isfinite(result.tau_model_h) or result.tau_model_h < 0.0:
        raise ValueError("result tau_model_h must be finite and non-negative")
    if result.bucket_count < 0:
        raise ValueError("bucket_count must be non-negative")
    if not math.isfinite(result.tau_plan_h) or result.tau_plan_h < 0.0:
        raise ValueError("result tau_plan_h must be finite and non-negative")
    if result.bucket_count == 0:
        if result.tau_model_h != 0.0 or result.tau_plan_h != 0.0:
            raise ValueError("zero-distance result must have zero duration and zero plan")
        return
    if result.tau_model_h <= 0.0:
        raise ValueError("positive-distance result must have tau_model_h > 0")
    if result.bucket_count < 1:
        raise ValueError("positive-distance result must have bucket_count >= 1")
    if result.tau_plan_h <= 0.0:
        raise ValueError("positive-distance result must have tau_plan_h > 0")
    if result.tau_plan_h < result.tau_model_h:
        raise ValueError("tau_plan_h must be at least tau_model_h")
    expected_plan = result.bucket_count * time_step_h
    if not math.isclose(result.tau_plan_h, expected_plan, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("tau_plan_h must equal bucket_count * time_step_h")


def _reject_result(
    *,
    iterations: int,
    diagnostic: str,
) -> ModeledTraversalResult:
    return ModeledTraversalResult(
        tau_model_h=math.inf,
        bucket_count=0,
        tau_plan_h=math.inf,
        converged=False,
        used_fallback=False,
        rejected=True,
        iterations=iterations,
        diagnostic=diagnostic,
    )


def _fallback_result(
    *,
    distance_nm: float,
    tas_kt: float,
    wind_bound_kt: float,
    temporal_cfg: TemporalConfig,
    iterations: int,
    diagnostic: str,
) -> ModeledTraversalResult:
    tau_fallback_h = distance_nm / (tas_kt - wind_bound_kt)
    bucket_count = max(1, int(math.ceil(tau_fallback_h / temporal_cfg.time_step_h)))
    return ModeledTraversalResult(
        tau_model_h=tau_fallback_h,
        bucket_count=bucket_count,
        tau_plan_h=bucket_count * temporal_cfg.time_step_h,
        converged=False,
        used_fallback=True,
        rejected=False,
        iterations=iterations,
        diagnostic=diagnostic,
    )


def solve_modeled_traversal_duration(
    wind: DynamicWindField,
    *,
    distance_nm: float,
    track_unit: Vec2,
    tas_kt: float,
    departure_time_h: float,
    x_mid_nm: float,
    y_mid_nm: float,
    altitude_ft: float,
    temporal_cfg: TemporalConfig,
    tolerance_h: float = DEFAULT_TRAVERSAL_TOLERANCE_H,
    max_iterations: int = DEFAULT_TRAVERSAL_MAX_ITERATIONS,
    initial_guess_h: float | None = None,
) -> ModeledTraversalResult:
    """Solve a single midpoint-approximate dynamic-wind transit time.

    The computed duration is a deterministic model quantity used by the discrete
    planner. The continuous modeled time is transient: the returned result is the
    planner representation itself, not a stored mission-time state.
    """
    if wind is None:
        raise ValueError("wind must not be None")
    if not isinstance(temporal_cfg, TemporalConfig):
        raise ValueError("temporal_cfg must be a TemporalConfig")
    if not math.isfinite(distance_nm) or distance_nm < 0.0:
        raise ValueError("distance_nm must be finite and non-negative")
    if not math.isfinite(tas_kt) or tas_kt <= 0.0:
        raise ValueError("tas_kt must be finite and > 0")
    if not math.isfinite(departure_time_h):
        raise ValueError("departure_time_h must be finite")
    if not math.isfinite(x_mid_nm) or not math.isfinite(y_mid_nm):
        raise ValueError("midpoint coordinates must be finite")
    if not math.isfinite(altitude_ft):
        raise ValueError("altitude_ft must be finite")
    if not math.isfinite(tolerance_h) or tolerance_h <= 0.0:
        raise ValueError("tolerance_h must be finite and > 0")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")

    wind_bound_kt = wind.max_magnitude_kt()
    if not math.isfinite(wind_bound_kt) or wind_bound_kt < 0.0:
        raise ValueError("wind maximum magnitude must be finite and non-negative")

    track_unit = _validate_track_unit(track_unit)

    if distance_nm == 0.0:
        result = ModeledTraversalResult(
            tau_model_h=0.0,
            bucket_count=0,
            tau_plan_h=0.0,
            converged=True,
            used_fallback=False,
            rejected=False,
            iterations=0,
            diagnostic="zero-distance transition",
        )
        _check_accepted_result_invariants(result, time_step_h=temporal_cfg.time_step_h)
        return result

    estimate = D = distance_nm
    if initial_guess_h is None:
        estimate = D / tas_kt
    else:
        if not math.isfinite(initial_guess_h) or initial_guess_h <= 0.0:
            raise ValueError("initial_guess_h must be finite and > 0")
        estimate = float(initial_guess_h)

    if not math.isfinite(estimate) or estimate <= 0.0:
        raise ValueError("initial estimate must be finite and > 0")

    for iteration in range(1, max_iterations + 1):
        t_mid = departure_time_h + estimate / 2.0
        wind_mid = wind.at_time(x_mid_nm, y_mid_nm, altitude_ft, t_mid)
        if not math.isfinite(wind_mid.x) or not math.isfinite(wind_mid.y):
            raise ValueError("wind sample must be finite")

        solution = solve_ground_speed(tas_kt, wind_mid, track_unit)
        if not solution.feasible:
            if tas_kt > wind_bound_kt:
                result = _fallback_result(
                    distance_nm=distance_nm,
                    tas_kt=tas_kt,
                    wind_bound_kt=wind_bound_kt,
                    temporal_cfg=temporal_cfg,
                    iterations=iteration,
                    diagnostic="dynamic travel-time solve used conservative fallback after infeasible midpoint sample",
                )
                _check_accepted_result_invariants(result, time_step_h=temporal_cfg.time_step_h)
                return result
            return _reject_result(
                iterations=iteration,
                diagnostic="dynamic travel-time solve failed: no finite conservative fallback",
            )

        if not math.isfinite(solution.ground_speed_kt) or solution.ground_speed_kt <= 0.0:
            if tas_kt > wind_bound_kt:
                result = _fallback_result(
                    distance_nm=distance_nm,
                    tas_kt=tas_kt,
                    wind_bound_kt=wind_bound_kt,
                    temporal_cfg=temporal_cfg,
                    iterations=iteration,
                    diagnostic="dynamic travel-time solve used conservative fallback after invalid ground speed",
                )
                _check_accepted_result_invariants(result, time_step_h=temporal_cfg.time_step_h)
                return result
            return _reject_result(
                iterations=iteration,
                diagnostic="dynamic travel-time solve failed: no finite conservative fallback",
            )

        tau_next = distance_nm / solution.ground_speed_kt
        if not math.isfinite(tau_next) or tau_next <= 0.0:
            if tas_kt > wind_bound_kt:
                result = _fallback_result(
                    distance_nm=distance_nm,
                    tas_kt=tas_kt,
                    wind_bound_kt=wind_bound_kt,
                    temporal_cfg=temporal_cfg,
                    iterations=iteration,
                    diagnostic="dynamic travel-time solve used conservative fallback after non-finite speed",
                )
                _check_accepted_result_invariants(result, time_step_h=temporal_cfg.time_step_h)
                return result
            return _reject_result(
                iterations=iteration,
                diagnostic="dynamic travel-time solve failed: no finite conservative fallback",
            )

        if abs(tau_next - estimate) <= tolerance_h:
            bucket_count = max(1, int(math.ceil(tau_next / temporal_cfg.time_step_h)))
            tau_plan_h = bucket_count * temporal_cfg.time_step_h
            result = ModeledTraversalResult(
                tau_model_h=tau_next,
                bucket_count=bucket_count,
                tau_plan_h=tau_plan_h,
                converged=True,
                used_fallback=False,
                rejected=False,
                iterations=iteration,
                diagnostic="fixed-point convergence",
            )
            _check_accepted_result_invariants(result, time_step_h=temporal_cfg.time_step_h)
            return result

        estimate = tau_next

    if tas_kt > wind_bound_kt:
        result = _fallback_result(
            distance_nm=distance_nm,
            tas_kt=tas_kt,
            wind_bound_kt=wind_bound_kt,
            temporal_cfg=temporal_cfg,
            iterations=max_iterations,
            diagnostic="dynamic travel-time solve used conservative fallback after max_iterations",
        )
        _check_accepted_result_invariants(result, time_step_h=temporal_cfg.time_step_h)
        return result
    return _reject_result(
        iterations=max_iterations,
        diagnostic="dynamic travel-time solve failed: no finite conservative fallback",
    )
