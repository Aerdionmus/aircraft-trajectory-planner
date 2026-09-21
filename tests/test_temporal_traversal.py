from __future__ import annotations

import math

import pytest

from atp.core.geometry import Vec2
from atp.core.temporal import TemporalConfig
from atp.environment.wind import DynamicWindField, PeriodicWind
from atp.planning.temporal import solve_modeled_traversal_duration


class ConstantDynamicWind(DynamicWindField):
    def __init__(self, vector_kt: Vec2) -> None:
        self.vector_kt = vector_kt

    def at_time(self, x_nm: float, y_nm: float, altitude_ft: float, t_h: float) -> Vec2:
        return self.vector_kt

    def max_magnitude_kt(self) -> float:
        return self.vector_kt.norm()


class AlternatingWind(DynamicWindField):
    def __init__(self, amplitude_kt: float) -> None:
        self.amplitude_kt = amplitude_kt

    def at_time(self, x_nm: float, y_nm: float, altitude_ft: float, t_h: float) -> Vec2:
        if t_h < 0.5:
            return Vec2(0.0, 1.0) * self.amplitude_kt
        return Vec2(0.0, -1.0) * self.amplitude_kt

    def max_magnitude_kt(self) -> float:
        return abs(self.amplitude_kt)


def test_zero_wind_closed_form_case_converges():
    cfg = TemporalConfig(time_step_h=0.5)
    result = solve_modeled_traversal_duration(
        ConstantDynamicWind(Vec2(0.0, 0.0)),
        distance_nm=100.0,
        track_unit=Vec2(1.0, 0.0),
        tas_kt=50.0,
        departure_time_h=0.0,
        x_mid_nm=0.0,
        y_mid_nm=0.0,
        altitude_ft=0.0,
        temporal_cfg=cfg,
    )
    assert result.converged is True
    assert result.rejected is False
    assert result.tau_model_h == pytest.approx(100.0 / 50.0)
    assert result.tau_plan_h == pytest.approx(2.0)
    assert result.bucket_count == 4


def test_static_in_time_dynamic_wind_is_repeatable():
    cfg = TemporalConfig(time_step_h=1.0)
    wind = ConstantDynamicWind(Vec2(20.0, 0.0))
    a = solve_modeled_traversal_duration(
        wind,
        distance_nm=100.0,
        track_unit=Vec2(1.0, 0.0),
        tas_kt=60.0,
        departure_time_h=3.0,
        x_mid_nm=0.0,
        y_mid_nm=0.0,
        altitude_ft=1000.0,
        temporal_cfg=cfg,
    )
    b = solve_modeled_traversal_duration(
        wind,
        distance_nm=100.0,
        track_unit=Vec2(1.0, 0.0),
        tas_kt=60.0,
        departure_time_h=7.0,
        x_mid_nm=0.0,
        y_mid_nm=0.0,
        altitude_ft=1000.0,
        temporal_cfg=cfg,
    )
    assert a.tau_model_h == pytest.approx(b.tau_model_h)
    assert a.diagnostic == b.diagnostic


def test_periodic_wind_changes_with_departure_time():
    cfg = TemporalConfig(time_step_h=0.25)
    wind = PeriodicWind(
        centre_nm=Vec2(0.0, 0.0),
        base_vector_kt=Vec2(0.0, 0.0),
        amplitude_kt=50.0,
        period_h=4.0,
        altitude_scale_ft=1000.0,
    )
    a = solve_modeled_traversal_duration(
        wind,
        distance_nm=100.0,
        track_unit=Vec2(1.0, 0.0),
        tas_kt=60.0,
        departure_time_h=0.0,
        x_mid_nm=1.0,
        y_mid_nm=0.0,
        altitude_ft=0.0,
        temporal_cfg=cfg,
    )
    b = solve_modeled_traversal_duration(
        wind,
        distance_nm=100.0,
        track_unit=Vec2(1.0, 0.0),
        tas_kt=60.0,
        departure_time_h=2.0,
        x_mid_nm=1.0,
        y_mid_nm=0.0,
        altitude_ft=0.0,
        temporal_cfg=cfg,
    )
    assert a.tau_model_h != pytest.approx(b.tau_model_h)
    assert a.tau_model_h > 0.0
    assert b.tau_model_h > 0.0


def test_rejects_malformed_track_vectors():
    cfg = TemporalConfig(time_step_h=0.5)
    with pytest.raises(ValueError):
        solve_modeled_traversal_duration(
            ConstantDynamicWind(Vec2(0.0, 0.0)),
            distance_nm=10.0,
            track_unit=Vec2(0.0, 0.0),
            tas_kt=50.0,
            departure_time_h=0.0,
            x_mid_nm=0.0,
            y_mid_nm=0.0,
            altitude_ft=0.0,
            temporal_cfg=cfg,
        )
    with pytest.raises(ValueError):
        solve_modeled_traversal_duration(
            ConstantDynamicWind(Vec2(0.0, 0.0)),
            distance_nm=10.0,
            track_unit=Vec2(3.0, 4.0),
            tas_kt=50.0,
            departure_time_h=0.0,
            x_mid_nm=0.0,
            y_mid_nm=0.0,
            altitude_ft=0.0,
            temporal_cfg=cfg,
        )


def test_zero_distance_transition_has_zero_duration_result():
    cfg = TemporalConfig(time_step_h=0.5)
    result = solve_modeled_traversal_duration(
        ConstantDynamicWind(Vec2(0.0, 0.0)),
        distance_nm=0.0,
        track_unit=Vec2(1.0, 0.0),
        tas_kt=50.0,
        departure_time_h=0.0,
        x_mid_nm=0.0,
        y_mid_nm=0.0,
        altitude_ft=0.0,
        temporal_cfg=cfg,
    )
    assert result.tau_model_h == 0.0
    assert result.bucket_count == 0
    assert result.tau_plan_h == 0.0
    assert result.converged is True
    assert result.rejected is False


def test_accepted_result_invariants_hold_for_positive_distance():
    cfg = TemporalConfig(time_step_h=0.5)
    result = solve_modeled_traversal_duration(
        ConstantDynamicWind(Vec2(10.0, 0.0)),
        distance_nm=100.0,
        track_unit=Vec2(1.0, 0.0),
        tas_kt=80.0,
        departure_time_h=0.0,
        x_mid_nm=0.0,
        y_mid_nm=0.0,
        altitude_ft=0.0,
        temporal_cfg=cfg,
    )
    assert math.isfinite(result.tau_model_h) and result.tau_model_h > 0.0
    assert result.bucket_count >= 1
    assert math.isfinite(result.tau_plan_h) and result.tau_plan_h > 0.0
    assert result.tau_plan_h >= result.tau_model_h
    assert result.tau_plan_h == pytest.approx(result.bucket_count * cfg.time_step_h)


def test_fixed_point_converges_by_tolerance():
    cfg = TemporalConfig(time_step_h=0.5)
    wind = ConstantDynamicWind(Vec2(10.0, 0.0))
    result = solve_modeled_traversal_duration(
        wind,
        distance_nm=100.0,
        track_unit=Vec2(1.0, 0.0),
        tas_kt=80.0,
        departure_time_h=0.0,
        x_mid_nm=0.0,
        y_mid_nm=0.0,
        altitude_ft=0.0,
        temporal_cfg=cfg,
        tolerance_h=1e-12,
        max_iterations=50,
    )
    assert result.converged is True
    assert result.used_fallback is False
    assert result.rejected is False
    assert result.iterations >= 1


def test_max_iterations_uses_conservative_fallback_when_tas_exceeds_bound():
    cfg = TemporalConfig(time_step_h=0.5)
    wind = ConstantDynamicWind(Vec2(0.0, 0.0))
    result = solve_modeled_traversal_duration(
        wind,
        distance_nm=100.0,
        track_unit=Vec2(1.0, 0.0),
        tas_kt=100.0,
        departure_time_h=0.0,
        x_mid_nm=0.0,
        y_mid_nm=0.0,
        altitude_ft=0.0,
        temporal_cfg=cfg,
        initial_guess_h=0.25,
        tolerance_h=1e-12,
        max_iterations=1,
    )
    assert result.used_fallback is True
    assert result.rejected is False
    assert result.tau_model_h == pytest.approx(100.0 / (100.0 - 0.0))
    assert result.tau_plan_h == pytest.approx(1.0)


def test_no_finite_fallback_rejects_when_tas_le_wmax():
    cfg = TemporalConfig(time_step_h=0.5)
    wind = ConstantDynamicWind(Vec2(100.0, 0.0))
    result = solve_modeled_traversal_duration(
        wind,
        distance_nm=100.0,
        track_unit=Vec2(1.0, 0.0),
        tas_kt=40.0,
        departure_time_h=0.0,
        x_mid_nm=0.0,
        y_mid_nm=0.0,
        altitude_ft=0.0,
        temporal_cfg=cfg,
        initial_guess_h=0.25,
        tolerance_h=1e-12,
        max_iterations=1,
    )
    assert result.rejected is True
    assert result.used_fallback is False
    assert math.isinf(result.tau_model_h)
    assert math.isinf(result.tau_plan_h)
    assert "no finite conservative fallback" in result.diagnostic


def test_bucket_quantisation_and_exact_boundary_are_explicit():
    cfg = TemporalConfig(time_step_h=0.05)
    result_small = solve_modeled_traversal_duration(
        ConstantDynamicWind(Vec2(0.0, 0.0)),
        distance_nm=3.0,
        track_unit=Vec2(1.0, 0.0),
        tas_kt=100.0,
        departure_time_h=0.0,
        x_mid_nm=0.0,
        y_mid_nm=0.0,
        altitude_ft=0.0,
        temporal_cfg=cfg,
    )
    assert result_small.tau_model_h == pytest.approx(0.03)
    assert result_small.bucket_count == 1
    assert result_small.tau_plan_h == pytest.approx(0.05)

    result_exact = solve_modeled_traversal_duration(
        ConstantDynamicWind(Vec2(0.0, 0.0)),
        distance_nm=10.0,
        track_unit=Vec2(1.0, 0.0),
        tas_kt=50.0,
        departure_time_h=0.0,
        x_mid_nm=0.0,
        y_mid_nm=0.0,
        altitude_ft=0.0,
        temporal_cfg=cfg,
    )
    assert result_exact.tau_model_h == pytest.approx(0.2)
    assert result_exact.bucket_count == 4
    assert result_exact.tau_plan_h == pytest.approx(0.20)


def test_deterministic_repeatability():
    cfg = TemporalConfig(time_step_h=0.5)
    wind = AlternatingWind(40.0)
    inputs = dict(
        distance_nm=120.0,
        track_unit=Vec2(1.0, 0.0),
        tas_kt=70.0,
        departure_time_h=2.0,
        x_mid_nm=5.0,
        y_mid_nm=0.0,
        altitude_ft=2000.0,
        temporal_cfg=cfg,
    )
    a = solve_modeled_traversal_duration(wind, **inputs)
    b = solve_modeled_traversal_duration(wind, **inputs)
    assert a == b


def test_invalid_inputs_do_not_produce_valid_results():
    cfg = TemporalConfig(time_step_h=1.0)
    wind = ConstantDynamicWind(Vec2(10.0, 0.0))
    with pytest.raises(ValueError):
        solve_modeled_traversal_duration(
            wind,
            distance_nm=10.0,
            track_unit=Vec2(0.0, 0.0),
            tas_kt=50.0,
            departure_time_h=0.0,
            x_mid_nm=0.0,
            y_mid_nm=0.0,
            altitude_ft=0.0,
            temporal_cfg=cfg,
        )
    with pytest.raises(ValueError):
        solve_modeled_traversal_duration(
            wind,
            distance_nm=10.0,
            track_unit=Vec2(1.0, 0.0),
            tas_kt=float("nan"),
            departure_time_h=0.0,
            x_mid_nm=0.0,
            y_mid_nm=0.0,
            altitude_ft=0.0,
            temporal_cfg=cfg,
        )


def test_fallback_soundness_is_conservative():
    cfg = TemporalConfig(time_step_h=0.5)
    wind = ConstantDynamicWind(Vec2(0.0, 0.0))
    result = solve_modeled_traversal_duration(
        wind,
        distance_nm=200.0,
        track_unit=Vec2(1.0, 0.0),
        tas_kt=100.0,
        departure_time_h=0.0,
        x_mid_nm=0.0,
        y_mid_nm=0.0,
        altitude_ft=0.0,
        temporal_cfg=cfg,
        initial_guess_h=0.25,
        tolerance_h=1e-12,
        max_iterations=1,
    )
    assert result.used_fallback is True
    assert result.tau_model_h == pytest.approx(200.0 / 100.0)
    assert result.tau_model_h >= 0.0
