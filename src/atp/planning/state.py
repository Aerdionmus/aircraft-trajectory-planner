"""Heading-augmented planner state.

Milestone 1's state is :class:`~atp.environment.airspace.GridState`, the triple
``(ix, iy, il)``.  Milestone 2 adds a heading index, giving::

    FlightState(ix, iy, il, ih)

``ih`` is an **index into** :attr:`~atp.environment.airspace.GridSpec.moves`,
denoting the ground track just flown to arrive at the cell -- not an
independent angular discretisation.

Why index the move set
----------------------
On a grid the set of representable ground tracks is exactly the move set.  An
independent heading grid (say 36 bins of 10 degrees) would leave most bins
unreachable, would need a rounding step on every transition, and would require
that rounding error to be bounded before any admissibility claim could be made.
Indexing the move set gives **zero heading-representation error by
construction**, makes wraparound ordinary modular arithmetic on indices, and
keeps the state a hashable ``NamedTuple`` that ``planning/astar.py`` can consume
without modification.

``MOVES_16`` is not uniformly spaced
------------------------------------
Its knight moves lie at ``atan2(1, 2) = 26.565`` degrees, so consecutive
headings alternate 26.565 and 18.435 degree steps rather than sitting at
``360 / 16 = 22.5``.  :class:`TurnTable` therefore computes every angle from the
actual move vectors and never from ``2 pi |i - j| / K``.

Projection
----------
:func:`cell_of` is the projection ``pi: FlightState -> GridState`` used by every
environment, restriction, risk, blocking-cache and evaluation call, so nothing
below ``planning/`` learns about heading.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from ..core.geometry import Vec2, angle_between
from ..environment.airspace import GridState

#: Sentinel heading index for a state with no incoming leg: the start state of a
#: scenario that does not declare a departure heading.  No turn is ever charged
#: out of it and no gate is ever applied to it.
NO_HEADING: int = -1


class FlightState(NamedTuple):
    """Horizontal cell, flight level, and the index of the move just flown."""

    ix: int
    iy: int
    il: int
    ih: int = NO_HEADING

    @property
    def cell(self) -> GridState:
        """The projection ``pi`` onto the Milestone 1 state."""
        return GridState(self.ix, self.iy, self.il)

    @property
    def has_heading(self) -> bool:
        return self.ih != NO_HEADING


def cell_of(state) -> GridState:
    """``pi(state)``, accepting either a :class:`FlightState` or a
    :class:`~atp.environment.airspace.GridState`."""
    return GridState(state.ix, state.iy, state.il)


class TurnTable:
    """Angles between every ordered pair of moves in a connectivity set.

    Built once per grid.  ``angle_rad[i][j]`` is the unsigned angle between the
    ground tracks of ``moves[i]`` and ``moves[j]``; ``length_nm[i]`` is the
    horizontal ground length of ``moves[i]`` at the grid's cell size.
    """

    __slots__ = ("moves", "unit", "angle_rad", "length_nm", "cell_size_nm")

    def __init__(
        self, moves: tuple[tuple[int, int], ...], cell_size_nm: float
    ) -> None:
        self.moves = moves
        self.cell_size_nm = cell_size_nm
        self.unit: tuple[Vec2, ...] = tuple(
            Vec2(float(dx), float(dy)).normalized() for dx, dy in moves
        )
        self.length_nm: tuple[float, ...] = tuple(
            math.hypot(dx, dy) * cell_size_nm for dx, dy in moves
        )
        self.angle_rad: tuple[tuple[float, ...], ...] = tuple(
            tuple(angle_between(a, b) for b in self.unit) for a in self.unit
        )

    def __len__(self) -> int:
        return len(self.moves)

    def track_unit(self, index: int) -> Vec2:
        return self.unit[index]

    def index_of(self, move: tuple[int, int]) -> int:
        """Index of an exact move vector; raises if it is not in the set."""
        return self.moves.index(move)

    def nearest_index_for_bearing(self, bearing_deg: float) -> int:
        """Move whose ground track is closest to a true bearing (0 = north,
        clockwise).  Used only at the scenario I/O boundary."""
        target = Vec2(
            math.sin(math.radians(bearing_deg)), math.cos(math.radians(bearing_deg))
        )
        return min(
            range(len(self.unit)), key=lambda i: angle_between(self.unit[i], target)
        )
