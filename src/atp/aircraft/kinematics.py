"""Wind triangle.

The aircraft is a point mass flying at a commanded true airspeed (TAS).  To
maintain a *ground track* ``t_hat`` in the presence of a wind vector ``w`` it
must crab into the wind.  Decomposing the wind into components along and across
the track:

    w_along = w . t_hat
    w_cross = |w - w_along * t_hat|

the achievable ground speed is

    GS = w_along + sqrt(TAS^2 - w_cross^2)

The track is unflyable when ``w_cross > TAS`` (the crosswind cannot be
cancelled) or when the resulting ``GS <= 0`` (the aircraft is blown backwards).
Both cases are reported rather than silently clamped, because a planner that
quietly clamps infeasible legs produces trajectories that cannot be flown.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..core.geometry import Vec2


@dataclass(frozen=True, slots=True)
class GroundSpeedSolution:
    feasible: bool
    ground_speed_kt: float
    drift_angle_deg: float
    reason: str = ""


def solve_ground_speed(
    tas_kt: float, wind_kt: Vec2, track_unit: Vec2
) -> GroundSpeedSolution:
    """Solve the wind triangle for a commanded track.

    ``track_unit`` must be a unit vector; ``tas_kt`` must be positive.
    """
    if tas_kt <= 0.0:
        return GroundSpeedSolution(False, 0.0, 0.0, "non-positive TAS")

    along = wind_kt.dot(track_unit)
    cross_vec = wind_kt - track_unit * along
    cross = cross_vec.norm()

    if cross > tas_kt:
        return GroundSpeedSolution(
            False, 0.0, 0.0, f"crosswind {cross:.1f} kt exceeds TAS {tas_kt:.1f} kt"
        )

    gs = along + math.sqrt(max(0.0, tas_kt * tas_kt - cross * cross))
    if gs <= 1e-6:
        return GroundSpeedSolution(
            False, 0.0, 0.0, f"non-positive ground speed ({gs:.2f} kt)"
        )

    drift_deg = math.degrees(math.asin(max(-1.0, min(1.0, cross / tas_kt))))
    return GroundSpeedSolution(True, gs, drift_deg)


def max_possible_ground_speed_kt(max_tas_kt: float, max_wind_kt: float) -> float:
    """Sound upper bound on ground speed anywhere in the airspace.

    Used by the admissible heuristic.  The bound ``TAS + |w|_max`` is attained
    only with a pure tailwind, so it is valid but optimistic -- exactly what an
    admissible heuristic requires.
    """
    return max_tas_kt + max(0.0, max_wind_kt)
