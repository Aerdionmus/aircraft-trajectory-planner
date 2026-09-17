"""Text rendering of an airspace and a trajectory.

Deliberately the *only* visualisation in the MVP, and it depends on nothing.
Plotting is a presentation concern that tends to leak into model code; keeping
the sole renderer a pure ``(airspace, path) -> str`` function makes it
impossible for the planner to acquire a plotting dependency by accident, and
lets the renderer itself be unit tested.

Matplotlib output is deferred work (see ``docs/architecture.md``).
"""

from __future__ import annotations

from ..environment.airspace import Airspace, GridState

BLOCKED = "#"
FREE = "."
RISKY = ":"
HIGH_RISK = "*"
PATH = "o"
START = "S"
GOAL = "G"


def render_ascii(
    airspace: Airspace,
    path: list[GridState] | None = None,
    *,
    level: int = 0,
    width: int = 78,
    height: int = 36,
    risk_threshold: float = 0.15,
) -> str:
    """Render one flight level as a character grid.

    The airspace is down-sampled to at most ``width`` x ``height`` characters.
    Down-sampling is conservative for obstacles (a character is blocked if any
    covered cell is blocked) so a rendered route never looks safer than it is.
    """
    spec = airspace.spec
    cols = min(width, spec.cells_x)
    rows = min(height, spec.cells_y)
    sx = spec.cells_x / cols
    sy = spec.cells_y / rows

    max_risk = max(airspace.risk.max_density(), 1e-9)
    canvas = [[FREE for _ in range(cols)] for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            ix = min(spec.cells_x - 1, int((c + 0.5) * sx))
            iy = min(spec.cells_y - 1, int((r + 0.5) * sy))
            state = GridState(ix, iy, level)
            if airspace.is_blocked(state):
                canvas[r][c] = BLOCKED
                continue
            point = airspace.centre_nm(state)
            density = airspace.risk.density_at(
                point.x, point.y, airspace.altitude_ft(state)
            )
            ratio = density / max_risk
            if ratio >= 0.55:
                canvas[r][c] = HIGH_RISK
            elif ratio >= risk_threshold:
                canvas[r][c] = RISKY

    def to_cell(state: GridState) -> tuple[int, int]:
        return (
            min(rows - 1, int(state.iy / sy)),
            min(cols - 1, int(state.ix / sx)),
        )

    if path:
        for state in path:
            r, c = to_cell(state)
            canvas[r][c] = PATH
        r, c = to_cell(path[0])
        canvas[r][c] = START
        r, c = to_cell(path[-1])
        canvas[r][c] = GOAL

    # Row 0 is the south edge; print north-up.
    body = "\n".join("".join(row) for row in reversed(canvas))
    legend = (
        f"level FL{spec.flight_levels[level]}  "
        f"{spec.cells_x}x{spec.cells_y} cells @ {spec.cell_size_nm:g} NM  "
        f"[{BLOCKED}]restricted [{RISKY}/{HIGH_RISK}]risk "
        f"[{PATH}]route [{START}]origin [{GOAL}]destination"
    )
    return f"{body}\n{legend}"
