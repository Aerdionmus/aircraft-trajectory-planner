"""Discretised 2.5D airspace.

The airspace is a uniform horizontal grid of square cells replicated over a
small, explicitly enumerated set of flight levels ("2.5D": continuous-ish
laterally, discrete vertically).  A state is the integer triple
``(ix, iy, il)`` and is located at the *centre* of its cell.

Design notes
------------
* The grid is the only discretisation in the system.  Everything else (wind,
  risk, restrictions) is evaluated continuously and only sampled at grid
  resolution, so raising the resolution strictly improves fidelity.
* Cell blocking is evaluated lazily and cached: for a 200x200x4 airspace the
  eager version costs a few hundred thousand geometry queries at start-up for
  no benefit when A* only ever touches a fraction of the grid.
* Grids introduce a well-known discretisation bias (paths are restricted to the
  connectivity directions).  With 8-connectivity the worst-case length
  overestimate versus a straight line is ~8%.  Any-angle post-smoothing is
  listed as deferred work in ``docs/architecture.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator, NamedTuple

from ..core.geometry import Vec2
from ..core.units import flight_level_to_ft
from .restrictions import RestrictionSet
from .risk import ConstantRisk, RiskField
from .wind import WindField, ZeroWind


class GridState(NamedTuple):
    """Planner state: horizontal cell indices plus a flight-level index."""

    ix: int
    iy: int
    il: int


#: Horizontal moves, ordered deterministically (N, NE, E, ... ) so that search
#: expansion order is reproducible across runs and platforms.
MOVES_4: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (0, -1), (-1, 0))
MOVES_8: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)
#: 16-connectivity adds knight-like moves, halving the worst-case heading error
#: from 22.5 deg to 11.25 deg at roughly double the branching factor.
MOVES_16: tuple[tuple[int, int], ...] = MOVES_8 + (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)

CONNECTIVITY: dict[int, tuple[tuple[int, int], ...]] = {
    4: MOVES_4,
    8: MOVES_8,
    16: MOVES_16,
}


@dataclass(frozen=True)
class GridSpec:
    """Geometry of the discretisation."""

    cells_x: int
    cells_y: int
    cell_size_nm: float
    flight_levels: tuple[int, ...]  # e.g. (280, 300, 320, 340), in hundreds of ft
    connectivity: int = 8

    def __post_init__(self) -> None:
        if self.cells_x <= 0 or self.cells_y <= 0:
            raise ValueError("grid must have at least one cell in each axis")
        if self.cell_size_nm <= 0:
            raise ValueError("cell_size_nm must be positive")
        if not self.flight_levels:
            raise ValueError("at least one flight level is required")
        if tuple(sorted(self.flight_levels)) != tuple(self.flight_levels):
            raise ValueError("flight_levels must be given in ascending order")
        if self.connectivity not in CONNECTIVITY:
            raise ValueError(f"connectivity must be one of {sorted(CONNECTIVITY)}")

    @property
    def width_nm(self) -> float:
        return self.cells_x * self.cell_size_nm

    @property
    def height_nm(self) -> float:
        return self.cells_y * self.cell_size_nm

    @property
    def num_levels(self) -> int:
        return len(self.flight_levels)

    @property
    def moves(self) -> tuple[tuple[int, int], ...]:
        return CONNECTIVITY[self.connectivity]

    def altitude_ft(self, il: int) -> float:
        return flight_level_to_ft(self.flight_levels[il])

    def in_bounds(self, state: GridState) -> bool:
        return (
            0 <= state.ix < self.cells_x
            and 0 <= state.iy < self.cells_y
            and 0 <= state.il < self.num_levels
        )

    def centre_nm(self, state: GridState) -> Vec2:
        return Vec2(
            (state.ix + 0.5) * self.cell_size_nm,
            (state.iy + 0.5) * self.cell_size_nm,
        )

    def cell_for_point(self, point_nm: Vec2) -> tuple[int, int]:
        """Nearest in-bounds cell for a continuous position (clamped)."""
        ix = int(math.floor(point_nm.x / self.cell_size_nm))
        iy = int(math.floor(point_nm.y / self.cell_size_nm))
        return (
            max(0, min(self.cells_x - 1, ix)),
            max(0, min(self.cells_y - 1, iy)),
        )

    def level_for_altitude(self, altitude_ft: float) -> int:
        """Index of the nearest enumerated flight level."""
        return min(
            range(self.num_levels),
            key=lambda il: abs(self.altitude_ft(il) - altitude_ft),
        )

    def states(self) -> Iterator[GridState]:
        for il in range(self.num_levels):
            for iy in range(self.cells_y):
                for ix in range(self.cells_x):
                    yield GridState(ix, iy, il)

    @property
    def size(self) -> int:
        return self.cells_x * self.cells_y * self.num_levels


@dataclass
class Airspace:
    """Grid geometry bundled with the environmental models defined on it."""

    spec: GridSpec
    wind: WindField = field(default_factory=ZeroWind)
    restrictions: RestrictionSet = field(default_factory=RestrictionSet)
    risk: RiskField = field(default_factory=lambda: ConstantRisk(0.0))
    name: str = "airspace"
    _blocked_cache: dict[GridState, bool] = field(
        default_factory=dict, repr=False, compare=False
    )

    # -- geometry passthrough ------------------------------------------------
    def centre_nm(self, state: GridState) -> Vec2:
        return self.spec.centre_nm(state)

    def altitude_ft(self, state: GridState) -> float:
        return self.spec.altitude_ft(state.il)

    def in_bounds(self, state: GridState) -> bool:
        return self.spec.in_bounds(state)

    # -- environment queries -------------------------------------------------
    def wind_at(self, state: GridState) -> Vec2:
        p = self.centre_nm(state)
        return self.wind.at(p.x, p.y, self.altitude_ft(state))

    def is_blocked(self, state: GridState) -> bool:
        """True if the cell centre lies inside a hard restriction.

        Cell-centre sampling can miss a restriction smaller than a cell.  The
        transition-level test in :class:`~atp.planning.cost.CostModel` is the
        authoritative one; this is a fast pre-filter.
        """
        cached = self._blocked_cache.get(state)
        if cached is None:
            cached = self.restrictions.point_blocked(
                self.centre_nm(state), self.altitude_ft(state)
            )
            self._blocked_cache[state] = cached
        return cached

    def blocked_cell_count(self) -> int:
        """Diagnostic only: O(grid size). Not used inside the search loop."""
        return sum(1 for s in self.spec.states() if self.is_blocked(s))
