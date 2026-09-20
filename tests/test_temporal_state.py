from __future__ import annotations

import math

import pytest

from atp.core.temporal import TemporalConfig
from atp.planning.state import FlightState


def test_flight_state_k_participates_in_identity():
    a = FlightState(1, 2, 3, 4, 5, 0)
    b = FlightState(1, 2, 3, 4, 5, 0)
    c = FlightState(1, 2, 3, 4, 5, 1)
    assert a == b
    assert hash(a) == hash(b)
    assert a != c


def test_static_state_construction_remains_compatible():
    a = FlightState(11, 12, 13, 14)
    b = FlightState(11, 12, 13, 14, 0)
    c = FlightState(11, 12, 13, 14, 0, 0)
    assert a == b == c


def test_time_at_bucket_is_deterministic():
    cfg = TemporalConfig(time_origin_h=10.0, time_step_h=2.5)
    assert cfg.time_at_bucket(0) == pytest.approx(10.0)
    assert cfg.time_at_bucket(1) == pytest.approx(12.5)
    assert cfg.time_at_bucket(3) == pytest.approx(17.5)


def test_bucket_count_and_planned_duration_are_explicit_ceil_semantics():
    cfg = TemporalConfig(time_step_h=5.0)
    assert cfg.bucket_count_for_duration(0.0) == 0
    assert cfg.bucket_count_for_duration(0.1) == 1
    assert cfg.bucket_count_for_duration(5.0) == 1
    assert cfg.bucket_count_for_duration(11.0) == 3
    assert cfg.planned_duration_h(11.0) == pytest.approx(15.0)


def test_temporal_config_rejects_invalid_step():
    with pytest.raises(ValueError):
        TemporalConfig(time_step_h=0.0)
    with pytest.raises(ValueError):
        TemporalConfig(time_step_h=-1.0)
    with pytest.raises(ValueError):
        TemporalConfig(time_step_h=math.inf)
    with pytest.raises(ValueError):
        TemporalConfig(time_step_h=float("nan"))


def test_temporal_config_rejects_non_finite_values():
    with pytest.raises(ValueError):
        TemporalConfig(time_origin_h=math.inf, time_step_h=1.0)
    with pytest.raises(ValueError):
        TemporalConfig(time_origin_h=float("nan"), time_step_h=1.0)


def test_time_at_bucket_rejects_negative_non_integer_and_boolean_buckets():
    cfg = TemporalConfig(time_step_h=2.5)
    with pytest.raises(ValueError):
        cfg.time_at_bucket(-1)
    with pytest.raises(ValueError):
        cfg.time_at_bucket(1.5)
    with pytest.raises(ValueError):
        cfg.time_at_bucket(True)


def test_static_mode_requires_no_temporal_config():
    state = FlightState(2, 3, 4, 5)
    assert state.k == 0
    assert state.ih == 5


def test_temporal_values_are_not_wall_clock_dependent():
    cfg = TemporalConfig(time_origin_h=3.0, time_step_h=7.0)
    first = cfg.time_at_bucket(11)
    second = cfg.time_at_bucket(11)
    assert first == pytest.approx(second)
    assert first == pytest.approx(3.0 + 11.0 * 7.0)
