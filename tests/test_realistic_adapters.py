from __future__ import annotations

import pytest

from atp.aircraft.openap import OpenAPAdapterError, OpenAPProfile
from atp.api.serialization import trajectory_record
from atp.core.geometry import Vec2
from atp.environment.snapshot import default_weather_snapshot
from atp.environment.weather import ObservedWindField, WindObservation
from atp.planning.state import FlightState


def test_openap_profile_maps_to_existing_performance_contract() -> None:
    aircraft = OpenAPProfile(
        aircraft_id="A320",
        cruise_tas_kt=450.0,
        cruise_fuel_flow_kg_per_h=2400.0,
        max_climb_rate_fpm=1800.0,
        max_descent_rate_fpm=2200.0,
        service_ceiling_ft=41000.0,
    ).to_performance()

    assert aircraft.name == "openap:A320"
    assert aircraft.tas_kt(33000.0) == 450.0
    assert aircraft.service_ceiling_ft == 41000.0


def test_openap_profile_rejects_invalid_provider_data() -> None:
    with pytest.raises(OpenAPAdapterError):
        OpenAPProfile(
            aircraft_id="",
            cruise_tas_kt=450.0,
            cruise_fuel_flow_kg_per_h=2400.0,
            max_climb_rate_fpm=1800.0,
            max_descent_rate_fpm=2200.0,
            service_ceiling_ft=41000.0,
        ).to_performance()


def test_observed_wind_field_uses_nearest_spatiotemporal_observation() -> None:
    field = ObservedWindField(
        (
            WindObservation(0.0, 0.0, 30000.0, Vec2(20.0, 0.0), time_h=0.0),
            WindObservation(100.0, 0.0, 30000.0, Vec2(0.0, 40.0), time_h=2.0),
        )
    )

    assert field.at(1.0, 0.0, 30000.0) == Vec2(20.0, 0.0)
    assert field.at_time(99.0, 0.0, 30000.0, 2.0) == Vec2(0.0, 40.0)
    assert field.max_magnitude_kt() == 40.0


def test_snapshot_interpolates_position_altitude_and_time() -> None:
    field = default_weather_snapshot().field
    midpoint = field.at_time(140.0, 0.0, 24600.0, 2.0)
    origin = field.at_time(0.0, 0.0, 16400.0, 0.0)
    destination = field.at_time(280.0, 0.0, 32800.0, 4.0)

    assert origin != destination
    assert min(origin.x, destination.x) < midpoint.x < max(origin.x, destination.x)
    assert min(origin.y, destination.y) < midpoint.y < max(origin.y, destination.y)
    assert field.interpolation_method == "multilinear with boundary clamping"


def test_dense_telemetry_preserves_planner_endpoints_and_adds_samples() -> None:
    path = [
        FlightState(0, 0, 0, isp=0, k=0),
        FlightState(10, 0, 1, isp=0, k=2),
        FlightState(20, 0, 0, isp=0, k=4),
    ]
    dense = trajectory_record(
        path,
        cell_size_nm=10.0,
        flight_levels=(0, 100, 200),
        time_step_h=0.5,
        speed_set_kt=(330.0,),
        sampling_interval_nm=20.0,
    )

    assert len(dense) > len(path)
    assert (dense[0]["x"], dense[0]["y"]) == (0.0, 0.0)
    assert any(point["x"] == 100.0 and point["altitude_ft"] == 100.0 for point in dense)
    assert (dense[-1]["x"], dense[-1]["y"]) == (200.0, 0.0)
    climb = [point["altitude_ft"] for point in dense if point["flight_phase"] == "climb"]
    descent = [
        point["altitude_ft"] for point in dense if point["flight_phase"] == "descent"
    ]
    assert climb == sorted(climb)
    assert descent == sorted(descent, reverse=True)


def test_wind_triangle_headwind_and_tailwind_change_ground_speed() -> None:
    from atp.aircraft.kinematics import solve_ground_speed

    track = Vec2(0.0, 1.0)
    assert solve_ground_speed(400.0, Vec2(0.0, -50.0), track).ground_speed_kt == pytest.approx(350.0)
    assert solve_ground_speed(400.0, Vec2(0.0, 50.0), track).ground_speed_kt == pytest.approx(450.0)
