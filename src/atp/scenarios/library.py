"""Built-in scenarios.

Each entry is a plain :class:`~atp.scenarios.spec.ScenarioSpec`, i.e. the same
thing a JSON file produces, so the library is a convenience and never a second
code path.  They are ordered by what they are designed to isolate:

``empty-cruise``        sanity check -- the optimum is the straight line.
``two-no-fly``          hard constraint avoidance, no environmental effects.
``jetstream``           wind-driven lateral deviation; the shortest route is
                        not the fastest.
``layered-jetstream``   the favourable wind exists only at one level, so the
                        optimum requires a level change.
``convective-risk``     soft risk trade-off: the cheapest route depends on the
                        risk price, not on the geometry.
``corridor-charge``     soft restriction: penalty versus detour.
``dense-restrictions``  search-effort stress case.

``random_scenario`` produces reproducible pseudo-random instances from an
integer seed for scaling experiments.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable

from .spec import GridSpecDoc, ScenarioSpec

BASE_WEIGHTS: dict[str, float] = {
    "time_cost_per_hour": 3000.0,
    "fuel_cost_per_kg": 0.8,
    "distance_cost_per_nm": 2.0,
    "risk_cost_per_exposure": 0.0,
    "restriction_penalty_scale": 1.0,
}

RISK_WEIGHTS: dict[str, float] = dict(BASE_WEIGHTS, risk_cost_per_exposure=15000.0)


def _grid(cells: int = 60, cell_nm: float = 5.0, levels: tuple[int, ...] = (300,)) -> GridSpecDoc:
    return GridSpecDoc(
        cells_x=cells, cells_y=cells, cell_size_nm=cell_nm, flight_levels=list(levels)
    )


def empty_cruise() -> ScenarioSpec:
    return ScenarioSpec(
        name="empty-cruise",
        description="No wind, no restrictions, no risk. Optimal route is the "
        "straight line; used to verify the planner and the baseline agree.",
        grid=_grid(),
        start=[2, 2, 0],
        goal=[57, 57, 0],
        weights=dict(BASE_WEIGHTS),
    )


def two_no_fly() -> ScenarioSpec:
    return ScenarioSpec(
        name="two-no-fly",
        description="Two prohibited areas straddling the direct track.",
        grid=_grid(),
        start=[2, 2, 0],
        goal=[57, 57, 0],
        restrictions=[
            {
                "type": "circle",
                "id": "P-101",
                "centre_nm": [110.0, 90.0],
                "radius_nm": 35.0,
            },
            {
                "type": "polygon",
                "id": "P-102",
                "vertices_nm": [
                    [170.0, 175.0],
                    [235.0, 175.0],
                    [235.0, 220.0],
                    [170.0, 220.0],
                ],
            },
        ],
        weights=dict(BASE_WEIGHTS),
    )


def jetstream() -> ScenarioSpec:
    return ScenarioSpec(
        name="jetstream",
        description="A strong vortex centred on the direct track: flying through "
        "it gives pure crosswind, so the minimum-time route bows into the "
        "tailwind and is longer than the shortest route.",
        grid=_grid(),
        start=[2, 30, 0],
        goal=[57, 30, 0],
        wind=[
            {
                "type": "vortex",
                "centre_nm": [150.0, 152.5],
                "peak_speed_kt": 130.0,
                "core_radius_nm": 70.0,
                "clockwise": True,
            }
        ],
        weights=dict(BASE_WEIGHTS),
    )


def layered_jetstream() -> ScenarioSpec:
    return ScenarioSpec(
        name="layered-jetstream",
        description="A 90 kt tailwind exists only above FL320, so the "
        "minimum-cost trajectory must climb even though climbing costs fuel.",
        # 12 NM cells: at 450 kt a cell takes 96 s, which is enough to change
        # flight level within the 1800 fpm climb limit. With 5 NM cells every
        # level change would be correctly rejected as unflyable.
        grid=_grid(cell_nm=12.0, levels=(280, 300, 320, 340)),
        start=[2, 30, 0],
        goal=[57, 30, 0],
        wind=[
            {
                "type": "layered",
                "layers": [
                    {"upper_ft": 31000.0, "direction_from_deg": 90.0, "speed_kt": 25.0},
                    {"upper_ft": 33000.0, "direction_from_deg": 270.0, "speed_kt": 90.0},
                ],
                "above_kt": [70.0, 0.0],
            }
        ],
        weights=dict(BASE_WEIGHTS),
    )


def convective_risk() -> ScenarioSpec:
    return ScenarioSpec(
        name="convective-risk",
        description="Three convective cells on the direct track, priced as risk "
        "rather than prohibited. Route shape depends on risk_cost_per_exposure.",
        grid=_grid(),
        start=[2, 2, 0],
        goal=[57, 57, 0],
        risk=[
            {"type": "gaussian", "centre_nm": [120.0, 120.0], "peak": 1.0, "sigma_nm": 25.0},
            {"type": "gaussian", "centre_nm": [180.0, 160.0], "peak": 0.8, "sigma_nm": 20.0},
            {"type": "gaussian", "centre_nm": [90.0, 170.0], "peak": 0.6, "sigma_nm": 18.0},
        ],
        weights=dict(RISK_WEIGHTS),
    )


def corridor_charge() -> ScenarioSpec:
    return ScenarioSpec(
        name="corridor-charge",
        description="A chargeable corridor across the direct track: the planner "
        "trades the per-NM penalty against the detour distance.",
        grid=_grid(),
        start=[2, 2, 0],
        goal=[57, 57, 0],
        restrictions=[
            {
                "type": "corridor",
                "id": "C-200",
                "start_nm": [40.0, 200.0],
                "end_nm": [260.0, 90.0],
                "half_width_nm": 25.0,
                "hard": False,
                "penalty_per_nm": 900.0,
            }
        ],
        weights=dict(BASE_WEIGHTS),
    )


def dense_restrictions() -> ScenarioSpec:
    circles: list[dict[str, Any]] = []
    for k in range(9):
        angle = 2.0 * math.pi * k / 9.0
        circles.append(
            {
                "type": "circle",
                "id": f"D-{k:02d}",
                "centre_nm": [
                    150.0 + 85.0 * math.cos(angle),
                    150.0 + 85.0 * math.sin(angle),
                ],
                "radius_nm": 26.0,
            }
        )
    return ScenarioSpec(
        name="dense-restrictions",
        description="Nine prohibited areas in a ring. Search-effort stress case.",
        grid=_grid(cells=100, cell_nm=3.0),
        start=[3, 3, 0],
        goal=[96, 96, 0],
        restrictions=circles,
        weights=dict(BASE_WEIGHTS),
    )


SCENARIO_LIBRARY: dict[str, Callable[[], ScenarioSpec]] = {
    "empty-cruise": empty_cruise,
    "two-no-fly": two_no_fly,
    "jetstream": jetstream,
    "layered-jetstream": layered_jetstream,
    "convective-risk": convective_risk,
    "corridor-charge": corridor_charge,
    "dense-restrictions": dense_restrictions,
}


def get_scenario(name: str) -> ScenarioSpec:
    if name not in SCENARIO_LIBRARY:
        raise KeyError(
            f"unknown scenario {name!r}; available: {sorted(SCENARIO_LIBRARY)}"
        )
    return SCENARIO_LIBRARY[name]()


def random_scenario(
    seed: int,
    *,
    cells: int = 60,
    cell_size_nm: float = 5.0,
    num_restrictions: int = 6,
    num_hazards: int = 3,
    max_wind_kt: float = 80.0,
    weights: dict[str, float] | None = None,
) -> ScenarioSpec:
    """Reproducible random instance.

    Uses a private :class:`random.Random` seeded explicitly, never the global
    RNG, so generating a scenario cannot perturb anything else in the process.
    Start and goal are placed in opposite corners and restrictions are rejected
    if they would cover either, guaranteeing a feasible instance exists.
    """
    rng = random.Random(seed)
    extent = cells * cell_size_nm
    start = [2, 2, 0]
    goal = [cells - 3, cells - 3, 0]
    start_xy = ((start[0] + 0.5) * cell_size_nm, (start[1] + 0.5) * cell_size_nm)
    goal_xy = ((goal[0] + 0.5) * cell_size_nm, (goal[1] + 0.5) * cell_size_nm)

    restrictions: list[dict[str, Any]] = []
    attempts = 0
    while len(restrictions) < num_restrictions and attempts < 1000:
        attempts += 1
        cx = rng.uniform(0.15 * extent, 0.85 * extent)
        cy = rng.uniform(0.15 * extent, 0.85 * extent)
        radius = rng.uniform(0.04 * extent, 0.10 * extent)
        clearance = radius + 2.0 * cell_size_nm
        if math.dist((cx, cy), start_xy) < clearance:
            continue
        if math.dist((cx, cy), goal_xy) < clearance:
            continue
        restrictions.append(
            {
                "type": "circle",
                "id": f"RND-{len(restrictions):02d}",
                "centre_nm": [round(cx, 3), round(cy, 3)],
                "radius_nm": round(radius, 3),
            }
        )

    hazards: list[dict[str, Any]] = [
        {
            "type": "gaussian",
            "centre_nm": [
                round(rng.uniform(0.1 * extent, 0.9 * extent), 3),
                round(rng.uniform(0.1 * extent, 0.9 * extent), 3),
            ],
            "peak": round(rng.uniform(0.3, 1.2), 3),
            "sigma_nm": round(rng.uniform(0.03 * extent, 0.08 * extent), 3),
        }
        for _ in range(num_hazards)
    ]

    wind = [
        {
            "type": "vortex",
            "centre_nm": [
                round(rng.uniform(0.2 * extent, 0.8 * extent), 3),
                round(rng.uniform(0.2 * extent, 0.8 * extent), 3),
            ],
            "peak_speed_kt": round(rng.uniform(0.25, 1.0) * max_wind_kt, 3),
            "core_radius_nm": round(rng.uniform(0.08, 0.2) * extent, 3),
            "clockwise": rng.random() < 0.5,
        }
    ]

    return ScenarioSpec(
        name=f"random-{seed}",
        description=f"Randomly generated instance, seed={seed}.",
        grid=GridSpecDoc(
            cells_x=cells,
            cells_y=cells,
            cell_size_nm=cell_size_nm,
            flight_levels=[300],
        ),
        start=start,
        goal=goal,
        wind=wind,
        restrictions=restrictions,
        risk=hazards,
        weights=dict(weights or RISK_WEIGHTS),
    )
