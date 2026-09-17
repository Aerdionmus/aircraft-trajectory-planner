"""Declarative scenario specification.

A scenario is fully described by a JSON document.  Nothing in a scenario is
implicit, and no scenario contains code, which gives three properties the
project needs:

* experiments are reproducible from a file under version control;
* the environment models can be swapped without touching the planner;
* a scenario can be diffed and reviewed like any other artefact.

Unknown keys are rejected rather than ignored, so a typo in a config fails
loudly instead of silently planning in the wrong airspace.

Schema versions
---------------
Version 1 is the Milestone 1 document.  It is still accepted verbatim and is
**migrated by forcing** ``turn_model="none"``, which reproduces Milestone 1
behaviour exactly -- a document written before turn dynamics existed cannot be
assumed to have been designed for them (several Milestone 1 scenarios use cell
sizes too small for the aircraft to turn at all; see ``docs/turn_model.md``).

A document with no ``schema_version`` at all is treated as version 1, for the
same reason.

Version 2 adds ``turn_model``, ``max_bank_deg``, ``max_turn_deg``,
``start_heading_deg`` and ``goal_heading_deg``.  A version 2 document that does
not name a turn model gets ``"gate+cost"``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from ..aircraft.performance import AIRCRAFT_LIBRARY, AircraftPerformance
from ..core.geometry import Vec2, met_wind_to_vector
from ..environment.airspace import Airspace, GridSpec, GridState
from ..environment.restrictions import (
    CircularRestriction,
    CorridorRestriction,
    PolygonRestriction,
    RestrictedRegion,
    RestrictionSet,
)
from ..environment.risk import (
    CompositeRisk,
    ConstantRisk,
    GaussianHazard,
    ProximityRisk,
    RiskField,
)
from ..environment.wind import (
    CompositeWind,
    LayeredWind,
    UniformWind,
    VortexWind,
    WindField,
    ZeroWind,
)
from ..planning.cost import TURN_MODELS, CostModel, CostWeights
from ..planning.problem import GoalSpec, TrajectoryPlanningProblem
from ..planning.state import TurnTable

SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = (1, 2)
#: Turn model applied to a version 2 document that does not name one.
DEFAULT_V2_TURN_MODEL = "gate+cost"


class ScenarioError(ValueError):
    """Raised for any malformed scenario document."""


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ScenarioError(f"{context}: missing required key {key!r}")
    return mapping[key]


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ScenarioError(f"{context}: unknown key(s) {sorted(unknown)}")


@dataclass(frozen=True)
class GridSpecDoc:
    cells_x: int
    cells_y: int
    cell_size_nm: float
    flight_levels: list[int]
    connectivity: int = 8

    def build(self) -> GridSpec:
        return GridSpec(
            cells_x=self.cells_x,
            cells_y=self.cells_y,
            cell_size_nm=self.cell_size_nm,
            flight_levels=tuple(self.flight_levels),
            connectivity=self.connectivity,
        )


@dataclass(frozen=True)
class ScenarioSpec:
    """In-memory form of a scenario document."""

    name: str
    grid: GridSpecDoc
    start: list[int]
    goal: list[int]
    aircraft: str = "medium-twin-jet"
    match_goal_level: bool = False
    wind: list[dict[str, Any]] = field(default_factory=list)
    restrictions: list[dict[str, Any]] = field(default_factory=list)
    risk: list[dict[str, Any]] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    allow_level_change_with_move: bool = True
    allow_pure_level_change: bool = False
    integration_samples: int = 6
    #: Milestone 2.  Defaults to ``"none"`` so that a scenario constructed in
    #: Python behaves exactly as it did in Milestone 1 unless it opts in.
    turn_model: str = "none"
    #: Overrides the aircraft's own bank limit when set.
    max_bank_deg: float | None = None
    max_turn_deg: float = 180.0
    #: True bearings (0 = north, clockwise), snapped to the nearest grid move.
    start_heading_deg: float | None = None
    goal_heading_deg: float | None = None
    description: str = ""
    schema_version: int = SCHEMA_VERSION

    # -- serialisation -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @staticmethod
    def from_dict(doc: dict[str, Any]) -> "ScenarioSpec":
        allowed = {
            "name",
            "grid",
            "start",
            "goal",
            "aircraft",
            "match_goal_level",
            "wind",
            "restrictions",
            "risk",
            "weights",
            "allow_level_change_with_move",
            "allow_pure_level_change",
            "integration_samples",
            "turn_model",
            "max_bank_deg",
            "max_turn_deg",
            "start_heading_deg",
            "goal_heading_deg",
            "description",
            "schema_version",
        }
        _reject_unknown(doc, allowed, "scenario")
        # An absent version means version 1: a document written without a
        # version predates Milestone 2, and silently switching turn dynamics on
        # underneath it would be the wrong default.
        version = doc.get("schema_version", 1)
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ScenarioError(
                f"scenario schema version {version} is not supported "
                f"(expected one of {list(SUPPORTED_SCHEMA_VERSIONS)})"
            )
        if version == 1:
            # Migration: a Milestone 1 document predates turn dynamics and is
            # not assumed to be flyable under them.
            turn_model = "none"
        else:
            turn_model = doc.get("turn_model", DEFAULT_V2_TURN_MODEL)
        if turn_model not in TURN_MODELS:
            raise ScenarioError(
                f"scenario: unknown turn_model {turn_model!r}; "
                f"available: {list(TURN_MODELS)}"
            )
        grid_doc = _require(doc, "grid", "scenario")
        _reject_unknown(
            grid_doc,
            {"cells_x", "cells_y", "cell_size_nm", "flight_levels", "connectivity"},
            "scenario.grid",
        )
        grid = GridSpecDoc(
            cells_x=_require(grid_doc, "cells_x", "scenario.grid"),
            cells_y=_require(grid_doc, "cells_y", "scenario.grid"),
            cell_size_nm=_require(grid_doc, "cell_size_nm", "scenario.grid"),
            flight_levels=list(_require(grid_doc, "flight_levels", "scenario.grid")),
            connectivity=grid_doc.get("connectivity", 8),
        )
        return ScenarioSpec(
            name=_require(doc, "name", "scenario"),
            grid=grid,
            start=list(_require(doc, "start", "scenario")),
            goal=list(_require(doc, "goal", "scenario")),
            aircraft=doc.get("aircraft", "medium-twin-jet"),
            match_goal_level=doc.get("match_goal_level", False),
            wind=list(doc.get("wind", [])),
            restrictions=list(doc.get("restrictions", [])),
            risk=list(doc.get("risk", [])),
            weights=dict(doc.get("weights", {})),
            allow_level_change_with_move=doc.get("allow_level_change_with_move", True),
            allow_pure_level_change=doc.get("allow_pure_level_change", False),
            integration_samples=doc.get("integration_samples", 6),
            turn_model=turn_model,
            max_bank_deg=(
                None if doc.get("max_bank_deg") is None else float(doc["max_bank_deg"])
            ),
            max_turn_deg=float(doc.get("max_turn_deg", 180.0)),
            start_heading_deg=(
                None
                if doc.get("start_heading_deg") is None
                else float(doc["start_heading_deg"])
            ),
            goal_heading_deg=(
                None
                if doc.get("goal_heading_deg") is None
                else float(doc["goal_heading_deg"])
            ),
            description=doc.get("description", ""),
        )

    @staticmethod
    def from_json(text: str) -> "ScenarioSpec":
        return ScenarioSpec.from_dict(json.loads(text))

    @staticmethod
    def load(path: str | Path) -> "ScenarioSpec":
        return ScenarioSpec.from_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")


# -- component factories -----------------------------------------------------
def _vec(doc: dict[str, Any], key: str, context: str) -> Vec2:
    value = _require(doc, key, context)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ScenarioError(f"{context}.{key}: expected [x, y]")
    return Vec2(float(value[0]), float(value[1]))


def build_wind(docs: list[dict[str, Any]]) -> WindField:
    fields: list[WindField] = []
    for i, doc in enumerate(docs):
        ctx = f"wind[{i}]"
        kind = _require(doc, "type", ctx)
        if kind == "zero":
            fields.append(ZeroWind())
        elif kind == "uniform":
            if "direction_from_deg" in doc:
                vector = met_wind_to_vector(
                    float(doc["direction_from_deg"]),
                    float(_require(doc, "speed_kt", ctx)),
                )
            else:
                vector = _vec(doc, "vector_kt", ctx)
            fields.append(UniformWind(vector_kt=vector))
        elif kind == "layered":
            layers = []
            for j, layer in enumerate(_require(doc, "layers", ctx)):
                lctx = f"{ctx}.layers[{j}]"
                if "direction_from_deg" in layer:
                    vector = met_wind_to_vector(
                        float(layer["direction_from_deg"]),
                        float(_require(layer, "speed_kt", lctx)),
                    )
                else:
                    vector = _vec(layer, "vector_kt", lctx)
                layers.append((float(_require(layer, "upper_ft", lctx)), vector))
            fields.append(
                LayeredWind(
                    layers=tuple(layers),
                    above=_vec(doc, "above_kt", ctx) if "above_kt" in doc else Vec2(0.0, 0.0),
                )
            )
        elif kind == "vortex":
            fields.append(
                VortexWind(
                    centre_nm=_vec(doc, "centre_nm", ctx),
                    peak_speed_kt=float(_require(doc, "peak_speed_kt", ctx)),
                    core_radius_nm=float(_require(doc, "core_radius_nm", ctx)),
                    clockwise=bool(doc.get("clockwise", False)),
                )
            )
        else:
            raise ScenarioError(f"{ctx}: unknown wind type {kind!r}")
    if not fields:
        return ZeroWind()
    if len(fields) == 1:
        return fields[0]
    return CompositeWind(fields=tuple(fields))


def build_restrictions(docs: list[dict[str, Any]]) -> RestrictionSet:
    regions: list[RestrictedRegion] = []
    for i, doc in enumerate(docs):
        ctx = f"restrictions[{i}]"
        kind = _require(doc, "type", ctx)
        common = {
            "region_id": doc.get("id", f"R{i}"),
            "lower_ft": float(doc.get("lower_ft", 0.0)),
            "upper_ft": float(doc.get("upper_ft", 60000.0)),
            "hard": bool(doc.get("hard", True)),
            "penalty_per_nm": float(doc.get("penalty_per_nm", 0.0)),
        }
        if kind == "circle":
            regions.append(
                CircularRestriction(
                    centre_nm=_vec(doc, "centre_nm", ctx),
                    radius_nm=float(_require(doc, "radius_nm", ctx)),
                    **common,
                )
            )
        elif kind == "polygon":
            vertices = tuple(
                Vec2(float(v[0]), float(v[1]))
                for v in _require(doc, "vertices_nm", ctx)
            )
            if len(vertices) < 3:
                raise ScenarioError(f"{ctx}: polygon needs at least 3 vertices")
            regions.append(PolygonRestriction(vertices_nm=vertices, **common))
        elif kind == "corridor":
            regions.append(
                CorridorRestriction(
                    start_nm=_vec(doc, "start_nm", ctx),
                    end_nm=_vec(doc, "end_nm", ctx),
                    half_width_nm=float(_require(doc, "half_width_nm", ctx)),
                    **common,
                )
            )
        else:
            raise ScenarioError(f"{ctx}: unknown restriction type {kind!r}")
    return RestrictionSet(regions=tuple(regions))


def build_risk(docs: list[dict[str, Any]], restrictions: RestrictionSet) -> RiskField:
    fields: list[RiskField] = []
    for i, doc in enumerate(docs):
        ctx = f"risk[{i}]"
        kind = _require(doc, "type", ctx)
        if kind == "constant":
            fields.append(ConstantRisk(density=float(doc.get("density", 0.0))))
        elif kind == "gaussian":
            fields.append(
                GaussianHazard(
                    centre_nm=_vec(doc, "centre_nm", ctx),
                    peak=float(_require(doc, "peak", ctx)),
                    sigma_nm=float(_require(doc, "sigma_nm", ctx)),
                    lower_ft=float(doc.get("lower_ft", 0.0)),
                    upper_ft=float(doc.get("upper_ft", 60000.0)),
                )
            )
        elif kind == "proximity":
            region_id = _require(doc, "region_id", ctx)
            matches = [r for r in restrictions.regions if r.region_id == region_id]
            if not matches:
                raise ScenarioError(f"{ctx}: no restriction with id {region_id!r}")
            fields.append(
                ProximityRisk(
                    region=matches[0],
                    peak=float(_require(doc, "peak", ctx)),
                    buffer_nm=float(_require(doc, "buffer_nm", ctx)),
                )
            )
        else:
            raise ScenarioError(f"{ctx}: unknown risk type {kind!r}")
    if not fields:
        return ConstantRisk(0.0)
    if len(fields) == 1:
        return fields[0]
    return CompositeRisk(fields=tuple(fields))


def build_weights(doc: dict[str, float]) -> CostWeights:
    allowed = {
        "time_cost_per_hour",
        "fuel_cost_per_kg",
        "distance_cost_per_nm",
        "risk_cost_per_exposure",
        "restriction_penalty_scale",
    }
    _reject_unknown(doc, allowed, "weights")
    return CostWeights(**{k: float(v) for k, v in doc.items()})


def build_aircraft(name: str) -> AircraftPerformance:
    if name not in AIRCRAFT_LIBRARY:
        raise ScenarioError(
            f"unknown aircraft {name!r}; available: {sorted(AIRCRAFT_LIBRARY)}"
        )
    return AIRCRAFT_LIBRARY[name]


@dataclass
class BuiltScenario:
    """Everything a planner run needs, assembled from a :class:`ScenarioSpec`."""

    spec: ScenarioSpec
    airspace: Airspace
    aircraft: AircraftPerformance
    cost_model: CostModel
    problem: TrajectoryPlanningProblem


def _state(triple: list[int], context: str) -> GridState:
    if len(triple) != 3:
        raise ScenarioError(f"{context}: expected [ix, iy, level_index]")
    return GridState(int(triple[0]), int(triple[1]), int(triple[2]))


def _heading_index(
    bearing_deg: float | None, table: TurnTable
) -> int | None:
    if bearing_deg is None:
        return None
    return table.nearest_index_for_bearing(bearing_deg)


def build_scenario(spec: ScenarioSpec) -> BuiltScenario:
    grid = spec.grid.build()
    restrictions = build_restrictions(spec.restrictions)
    airspace = Airspace(
        spec=grid,
        wind=build_wind(spec.wind),
        restrictions=restrictions,
        risk=build_risk(spec.risk, restrictions),
        name=spec.name,
    )
    aircraft = build_aircraft(spec.aircraft)
    if spec.max_bank_deg is not None:
        if not 0.0 < spec.max_bank_deg < 90.0:
            raise ScenarioError("scenario.max_bank_deg must lie in (0, 90)")
        aircraft = replace(aircraft, max_bank_deg=float(spec.max_bank_deg))
    if not 0.0 < spec.max_turn_deg <= 180.0:
        raise ScenarioError("scenario.max_turn_deg must lie in (0, 180]")
    cost_model = CostModel(
        airspace,
        aircraft,
        build_weights(spec.weights),
        integration_samples=spec.integration_samples,
        turn_model=spec.turn_model,
        max_turn_deg=spec.max_turn_deg,
    )
    table = TurnTable(grid.moves, grid.cell_size_nm)
    problem = TrajectoryPlanningProblem(
        airspace,
        cost_model,
        start=_state(spec.start, "scenario.start"),
        goal=GoalSpec(
            _state(spec.goal, "scenario.goal"),
            spec.match_goal_level,
            _heading_index(spec.goal_heading_deg, table),
        ),
        allow_level_change_with_move=spec.allow_level_change_with_move,
        allow_pure_level_change=spec.allow_pure_level_change,
        start_heading_index=_heading_index(spec.start_heading_deg, table),
    )
    return BuiltScenario(spec, airspace, aircraft, cost_model, problem)
