"""Risk- and fuel-aware aircraft trajectory planning with A* heuristic search.

This is a **simulation and teaching** package.  It models a synthetic airspace
with synthetic wind, risk and restriction data and a heavily simplified aircraft
performance model.  It is not flight-planning software, is not validated against
any operational system, and carries no airworthiness or certification claim.
See ``docs/assumptions.md`` for the full list of modelling simplifications.
"""

from .data.airports import Airport, AirportDatabase, default_airport_database
from .environment.airspace import Airspace, GridSpec, GridState
from .geospatial.transform import GeospatialTransform, geodetic_to_local_nm, local_to_geodetic_nm
from .planning.astar import SearchResult, SearchStatus, astar, dijkstra
from .planning.cost import CostModel, CostWeights
from .planning.problem import GoalSpec, TrajectoryPlanningProblem
from .scenarios.spec import ScenarioSpec, build_scenario

__version__ = "0.1.0"

__all__ = [
    "Airport",
    "AirportDatabase",
    "default_airport_database",
    "Airspace",
    "GridSpec",
    "GridState",
    "GeospatialTransform",
    "geodetic_to_local_nm",
    "local_to_geodetic_nm",
    "SearchResult",
    "SearchStatus",
    "astar",
    "dijkstra",
    "CostModel",
    "CostWeights",
    "GoalSpec",
    "TrajectoryPlanningProblem",
    "ScenarioSpec",
    "build_scenario",
    "__version__",
]
