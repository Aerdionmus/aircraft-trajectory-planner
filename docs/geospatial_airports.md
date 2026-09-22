# Geospatial airport grounding and provenance

The repository keeps the planner planar by design. The search, cost model, wind
field, and restriction logic continue to work in local nautical-mile coordinates
that are already proven by the existing M1-M5.3 tests.

The geospatial adapter lives in backend/domain modules only:

- airport data: `src/atp/data/airports.py`
- transform: `src/atp/geospatial/transform.py`
- scenario adapter: `src/atp/scenarios/geospatial.py`

## Data provenance

- source: OurAirports public airport dataset
- source URL: <https://ourairports.com/data/airports.csv>
- ingestion date: `2025-01-01`
- expected fields: `icao`, `iata`, `name`, `municipality`, `country`,
  `latitude_deg`, `longitude_deg`, `elevation_ft`, `airport_type`
- normalization assumptions: airport codes are upper-cased, optional strings are
  left as `None`, and every record retains the original published metadata.

## Execution model

1. Resolve a real airport code or location from the airport database.
2. Choose a local tangent-plane reference (usually the midpoint of origin and
   destination, unless a caller provides an explicit reference airport).
3. Convert each latitude/longitude pair into local east/north coordinates in NM.
4. Place the start and goal in a planner grid built on that local frame.
5. Run the existing planner exactly as before.
6. Convert the resulting local route back to geodetic coordinates only for
   geographic visualization and analytics overlays.

The transform uses the WGS84 ellipsoid (`a = 6378137 m`,
`f = 1 / 298.257223563`) and a first-order local ENU tangent plane. At the
reference latitude `phi0`, with WGS84 meridian and prime-vertical radii `M` and
`N`, forward conversion is:

```text
east_m  = (lambda - lambda0) * N(phi0) * cos(phi0)
north_m = (phi - phi0) * M(phi0)
local_nm = (east_m, north_m) / 1852
```

The inverse uses the corresponding local linearization:

```text
phi      = phi0 + north_m / M(phi0)
lambda   = lambda0 + east_m / (N(phi0) * cos(phi0))
```

This is appropriate for the route-sized local planning frames used here; it is
not a hard-coded latitude/longitude scale or a frontend approximation.

This intentionally does not rewrite A* or Dijkstra, and it never forces the
planner to operate directly on latitude/longitude. The geospatial conversion is a
boundary layer around an otherwise unchanged planner.

## Cesium runtime note

Cesium is initialized with `EllipsoidTerrainProvider`, no imagery provider,
`baseLayerPicker: false`, and the normal `Viewer` WebGL path. The frontend does
not perform a geographic fallback or hide initialization errors. In the current
headless browser runtime, a direct canvas probe returns no WebGL context and
Cesium reports `Error constructing CesiumWidget` / `WebGL initialization failed`.
That is an environment/software-rendering limitation, not a Cesium route or
coordinate configuration error. A normal WebGL-capable browser can use the same
viewer configuration and geographic `Cartesian3.fromDegrees` route path.

## Scope boundary

The geographic boundary is real airport data plus the WGS84 transform. The
default regression path remains synthetic, while the explicit M6.2/M6.3 modes
can use the pinned OpenAP A320 provider and local Open-Meteo weather snapshot.
The planner is still a grid-based research model, not an operational flight
planning system: climb/descent limits, sparse weather sampling, route
discretization, and fuel evaluation retain documented modelling assumptions.

OpenAP airport responses keep the discrete search path unchanged and densify
only the serialized telemetry at 20 NM intervals. Snapshot wind uses
multilinear interpolation across the sampled horizontal, altitude, and time
axes; the supplied snapshot has one north/south axis value, so that dimension
is clamped. Queries outside any sampled axis are clamped to its boundary rather
than extrapolated. Fuel and mass are integrated over the dense telemetry time
steps after search and are not planner state variables.
