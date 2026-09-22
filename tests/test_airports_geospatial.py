from __future__ import annotations

import math

from fastapi.testclient import TestClient

from atp.data.airports import default_airport_database
from atp.geospatial.transform import GeospatialTransform
from atp.scenarios.geospatial import airport_scenario

from atp.api.app import app


def test_airport_database_has_real_provenance():
    db = default_airport_database()
    airport = db.lookup("VOMM")
    assert airport.icao == "VOMM"
    assert airport.iata == "MAA"
    assert airport.source == "OurAirports"
    assert airport.source_url.startswith("https://ourairports.com")


def test_geospatial_transform_round_trips():
    transform = GeospatialTransform(12.9900, 80.1696)
    error_nm = transform.validate_round_trip(12.9900, 80.1696)
    assert error_nm < 1e-9

    local = transform.to_local(19.0896, 72.8656)
    assert abs(local.x) > 0.0
    assert abs(local.y) > 0.0


def test_airport_pair_builds_local_scenario():
    scenario = airport_scenario("VOMM", "VABB")
    assert scenario.start != scenario.goal
    assert scenario.grid.cells_x > 0
    assert scenario.grid.cells_y > 0


def test_api_airport_resolution_and_plan_route():
    client = TestClient(app)
    airport = client.get("/api/airports/VOMM")
    assert airport.status_code == 200
    assert airport.json()["icao"] == "VOMM"

    response = client.post(
        "/api/plan",
        json={
            "mode": "static",
            "algorithm": "astar",
            "heuristic": "zero",
            "start_airport": "VOMM",
            "goal_airport": "VABB",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "solved"
    assert body["geographic"] is True
    assert body["airports"]["origin"]["icao"] == "VOMM"
    assert body["airports"]["destination"]["icao"] == "VABB"
    assert body["geography"]["departure_snap_error_nm"] == 0
    assert body["geography"]["arrival_snap_error_nm"] > 0
    assert {"latitude_deg", "longitude_deg"} <= body["trajectory"][0].keys()
    assert {"latitude_deg", "longitude_deg"} <= body["trajectory"][-1].keys()
    assert math.isclose(body["trajectory"][0]["latitude_deg"], 12.9900, abs_tol=1e-6)
    assert math.isclose(body["trajectory"][0]["longitude_deg"], 80.1696, abs_tol=1e-6)
    assert math.isclose(body["trajectory"][-1]["latitude_deg"], 19.0896, abs_tol=0.2)
    assert math.isclose(body["trajectory"][-1]["longitude_deg"], 72.8656, abs_tol=0.2)
    assert all(
        math.isfinite(point["latitude_deg"]) and math.isfinite(point["longitude_deg"])
        for point in body["trajectory"]
    )


def test_dynamic_airport_route_has_geographic_trajectory():
    client = TestClient(app)
    response = client.post(
        "/api/plan",
        json={
            "mode": "dynamic",
            "algorithm": "astar",
            "heuristic": "zero",
            "start_airport": "VOMM",
            "goal_airport": "VABB",
            "wind_amplitude_kt": 20,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["geographic"] is True
    assert body["environment"]["wind"]["amplitude_kt"] == 20
    assert all("latitude_deg" in point and "longitude_deg" in point for point in body["trajectory"])
