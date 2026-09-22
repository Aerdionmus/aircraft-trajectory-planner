"""Airport-to-grid adapter that keeps the planner planar while grounding real airports."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..core.geometry import Vec2
from ..data.airports import Airport, AirportDatabase, default_airport_database
from ..environment.airspace import GridSpec
from ..geospatial.transform import GeospatialTransform
from .spec import GridSpecDoc, ScenarioSpec


@dataclass(frozen=True, slots=True)
class AirportRoute:
    """Real-world airport pair mapped to planner coordinates."""

    origin: Airport
    destination: Airport
    reference: Airport
    origin_local_nm: Vec2
    destination_local_nm: Vec2
    grid_origin_local_nm: Vec2
    grid: GridSpecDoc
    start: tuple[int, int, int]
    goal: tuple[int, int, int]


def resolve_airport_pair(
    start_airport: str,
    goal_airport: str,
    *,
    reference_airport: str | None = None,
    cell_size_nm: float = 5.0,
    margin_cells: int = 4,
    database: AirportDatabase | None = None,
) -> AirportRoute:
    db = database or default_airport_database()
    start = db.lookup(start_airport)
    goal = db.lookup(goal_airport)
    if reference_airport is None:
        ref_lat = (start.latitude_deg + goal.latitude_deg) / 2.0
        ref_lon = (start.longitude_deg + goal.longitude_deg) / 2.0
        reference = Airport(
            icao="REF",
            iata=None,
            name="Midpoint reference",
            municipality="Local tangent plane",
            country=None,
            latitude_deg=ref_lat,
            longitude_deg=ref_lon,
            elevation_ft=None,
            airport_type="virtual_airport",
            source=db.source,
            source_url=db.source_url,
            source_date=db.source_date,
        )
    else:
        reference = db.lookup(reference_airport)
        ref_lat = reference.latitude_deg
        ref_lon = reference.longitude_deg

    transform = GeospatialTransform(ref_lat, ref_lon)
    origin_local = transform.to_local(start.latitude_deg, start.longitude_deg)
    destination_local = transform.to_local(goal.latitude_deg, goal.longitude_deg)
    min_x = min(origin_local.x, destination_local.x)
    max_x = max(origin_local.x, destination_local.x)
    min_y = min(origin_local.y, destination_local.y)
    max_y = max(origin_local.y, destination_local.y)
    cells_x = max(6, int(math.ceil((max_x - min_x) / cell_size_nm)) + margin_cells * 2)
    cells_y = max(6, int(math.ceil((max_y - min_y) / cell_size_nm)) + margin_cells * 2)

    start_ix = int(math.floor((origin_local.x - min_x) / cell_size_nm)) + margin_cells
    start_iy = int(math.floor((origin_local.y - min_y) / cell_size_nm)) + margin_cells
    goal_ix = int(math.floor((destination_local.x - min_x) / cell_size_nm)) + margin_cells
    goal_iy = int(math.floor((destination_local.y - min_y) / cell_size_nm)) + margin_cells
    grid_origin = Vec2(
        origin_local.x - start_ix * cell_size_nm,
        origin_local.y - start_iy * cell_size_nm,
    )

    grid = GridSpecDoc(
        cells_x=cells_x,
        cells_y=cells_y,
        cell_size_nm=cell_size_nm,
        flight_levels=[300],
    )
    return AirportRoute(
        origin=start,
        destination=goal,
        reference=reference,
        origin_local_nm=origin_local,
        destination_local_nm=destination_local,
        grid_origin_local_nm=grid_origin,
        grid=grid,
        start=(start_ix, start_iy, 0),
        goal=(goal_ix, goal_iy, 0),
    )


def airport_scenario(
    start_airport: str,
    goal_airport: str,
    *,
    reference_airport: str | None = None,
    cell_size_nm: float = 5.0,
    margin_cells: int = 4,
    database: AirportDatabase | None = None,
    name: str | None = None,
) -> ScenarioSpec:
    route = resolve_airport_pair(
        start_airport,
        goal_airport,
        reference_airport=reference_airport,
        cell_size_nm=cell_size_nm,
        margin_cells=margin_cells,
        database=database,
    )
    scenario_name = name or f"{route.origin.icao}-{route.destination.icao}"
    return ScenarioSpec(
        name=scenario_name,
        grid=route.grid,
        start=list(route.start),
        goal=list(route.goal),
        description=(
            f"Geographically grounded route from {route.origin.name} ({route.origin.icao}) "
            f"to {route.destination.name} ({route.destination.icao}) using a local WGS84 tangent-plane"
            f" anchor at {route.reference.name}."
        ),
    )
