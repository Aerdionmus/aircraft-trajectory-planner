"""Deterministic WGS84 local-tangent-plane transforms used as a backend adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..core.geometry import Vec2

EARTH_RADIUS_M = 6_378_137.0
EARTH_RADIUS_NM = EARTH_RADIUS_M / 1852.0


@dataclass(frozen=True, slots=True)
class WGS84:
    """Reference constants for the local tangent-plane adapter."""

    a_m: float = EARTH_RADIUS_M
    b_m: float = 6_356_752.314245179
    f: float = 1.0 / 298.257223563


WGS84 = WGS84()


def _radii_of_curvature(latitude_rad: float) -> tuple[float, float]:
    """Return meridian and prime-vertical radii for the WGS84 ellipsoid."""
    eccentricity_sq = WGS84.f * (2.0 - WGS84.f)
    sin_lat = math.sin(latitude_rad)
    denominator = 1.0 - eccentricity_sq * sin_lat * sin_lat
    prime_vertical = WGS84.a_m / math.sqrt(denominator)
    meridian = (
        WGS84.a_m * (1.0 - eccentricity_sq)
        / denominator ** 1.5
    )
    return meridian, prime_vertical


def geodetic_to_local_enu_m(
    latitude_deg: float,
    longitude_deg: float,
    *,
    reference_latitude_deg: float,
    reference_longitude_deg: float,
    reference_altitude_m: float = 0.0,
) -> tuple[float, float, float]:
    """Convert geodetic latitude/longitude to a local ENU frame in metres.

    The planner remains agnostic to WGS84 and still consumes a planar, local
    coordinate system. This adapter is a thin backend transform only.
    """
    lat_rad = math.radians(latitude_deg)
    lon_rad = math.radians(longitude_deg)
    ref_lat_rad = math.radians(reference_latitude_deg)
    ref_lon_rad = math.radians(reference_longitude_deg)

    dlat = lat_rad - ref_lat_rad
    dlon = lon_rad - ref_lon_rad
    cos_ref = math.cos(ref_lat_rad)
    meridian, prime_vertical = _radii_of_curvature(ref_lat_rad)
    east_m = dlon * prime_vertical * cos_ref
    north_m = dlat * meridian
    up_m = (reference_altitude_m - reference_altitude_m) + 0.0
    return east_m, north_m, up_m


def geodetic_to_local_nm(
    latitude_deg: float,
    longitude_deg: float,
    *,
    reference_latitude_deg: float,
    reference_longitude_deg: float,
) -> Vec2:
    east_m, north_m, _ = geodetic_to_local_enu_m(
        latitude_deg,
        longitude_deg,
        reference_latitude_deg=reference_latitude_deg,
        reference_longitude_deg=reference_longitude_deg,
    )
    return Vec2(east_m / 1852.0, north_m / 1852.0)


def local_to_geodetic(
    x_nm: float,
    y_nm: float,
    *,
    reference_latitude_deg: float,
    reference_longitude_deg: float,
) -> tuple[float, float]:
    """Inverse transform from local north/east NM to WGS84 latitude/longitude."""
    ref_lat_rad = math.radians(reference_latitude_deg)
    meridian, prime_vertical = _radii_of_curvature(ref_lat_rad)
    east_m = x_nm * 1852.0
    north_m = y_nm * 1852.0
    lat_rad = ref_lat_rad + north_m / meridian
    lon_rad = math.radians(reference_longitude_deg) + (
        east_m / (prime_vertical * math.cos(ref_lat_rad))
    )
    return math.degrees(lat_rad), math.degrees(lon_rad)


def local_to_geodetic_nm(
    x_nm: float,
    y_nm: float,
    *,
    reference_latitude_deg: float,
    reference_longitude_deg: float,
) -> tuple[float, float]:
    return local_to_geodetic(
        x_nm,
        y_nm,
        reference_latitude_deg=reference_latitude_deg,
        reference_longitude_deg=reference_longitude_deg,
    )


def round_trip_error_nm(
    latitude_deg: float,
    longitude_deg: float,
    *,
    reference_latitude_deg: float,
    reference_longitude_deg: float,
) -> float:
    ref_lat_rad = math.radians(reference_latitude_deg)
    meridian, prime_vertical = _radii_of_curvature(ref_lat_rad)
    local = geodetic_to_local_nm(
        latitude_deg,
        longitude_deg,
        reference_latitude_deg=reference_latitude_deg,
        reference_longitude_deg=reference_longitude_deg,
    )
    lat_back, lon_back = local_to_geodetic(
        local.x,
        local.y,
        reference_latitude_deg=reference_latitude_deg,
        reference_longitude_deg=reference_longitude_deg,
    )
    return math.hypot(
        math.radians(lat_back - latitude_deg) * meridian / 1852.0,
        math.radians(lon_back - longitude_deg)
        * prime_vertical
        * math.cos(math.radians(reference_latitude_deg))
        / 1852.0,
    )


@dataclass(frozen=True, slots=True)
class GeospatialTransform:
    """Geodetic adapter anchored on a local tangent-plane origin."""

    reference_latitude_deg: float
    reference_longitude_deg: float

    def to_local(self, latitude_deg: float, longitude_deg: float) -> Vec2:
        return geodetic_to_local_nm(
            latitude_deg,
            longitude_deg,
            reference_latitude_deg=self.reference_latitude_deg,
            reference_longitude_deg=self.reference_longitude_deg,
        )

    def to_geodetic(self, x_nm: float, y_nm: float) -> tuple[float, float]:
        return local_to_geodetic(
            x_nm,
            y_nm,
            reference_latitude_deg=self.reference_latitude_deg,
            reference_longitude_deg=self.reference_longitude_deg,
        )

    def validate_round_trip(self, latitude_deg: float, longitude_deg: float) -> float:
        return round_trip_error_nm(
            latitude_deg,
            longitude_deg,
            reference_latitude_deg=self.reference_latitude_deg,
            reference_longitude_deg=self.reference_longitude_deg,
        )
