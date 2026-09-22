"""Declarative scenario definitions and the built-in scenario library."""

from .geospatial import AirportRoute, airport_scenario, resolve_airport_pair

__all__ = [
    "AirportRoute",
    "airport_scenario",
    "resolve_airport_pair",
]
