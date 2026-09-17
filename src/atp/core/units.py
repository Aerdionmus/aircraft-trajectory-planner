"""Unit conventions for the whole package.

Every quantity crossing a module boundary carries its unit in the attribute or
argument name (``*_nm``, ``*_ft``, ``*_kt``, ``*_h``, ``*_kg``).  This module is
the single place where conversion factors live, so a unit bug can only ever be
introduced once.

Canonical internal units
------------------------
============  ==================================================
Quantity      Unit
============  ==================================================
Horizontal    nautical mile (NM)
Vertical      foot (ft)
Speed         knot (kt = NM/h)
Time          hour (h)
Mass / fuel   kilogram (kg)
Angle         radian internally, degrees only at I/O boundaries
Cost          abstract "cost units" (see :mod:`atp.planning.cost`)
============  ==================================================

The horizontal plane is a *local flat-earth projection*: ``x`` points east and
``y`` points north, both in NM from the south-west corner of the airspace.  No
spherical or geodetic model is used.  See ``docs/assumptions.md``.
"""

from __future__ import annotations

FT_PER_NM: float = 6076.115485564304
NM_PER_FT: float = 1.0 / FT_PER_NM

MIN_PER_H: float = 60.0
S_PER_H: float = 3600.0

#: Flight levels are quoted in hundreds of feet (FL300 == 30000 ft).
FT_PER_FLIGHT_LEVEL: float = 100.0


def nm_to_ft(distance_nm: float) -> float:
    return distance_nm * FT_PER_NM


def ft_to_nm(distance_ft: float) -> float:
    return distance_ft * NM_PER_FT


def flight_level_to_ft(flight_level: float) -> float:
    return flight_level * FT_PER_FLIGHT_LEVEL


def ft_to_flight_level(altitude_ft: float) -> float:
    return altitude_ft / FT_PER_FLIGHT_LEVEL


def hours_to_minutes(hours: float) -> float:
    return hours * MIN_PER_H


def hours_to_seconds(hours: float) -> float:
    return hours * S_PER_H


def fpm_to_ft_per_hour(rate_fpm: float) -> float:
    return rate_fpm * MIN_PER_H
