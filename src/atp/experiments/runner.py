"""Deterministic planner runs and experiment matrices.

Two layers:

``run_plan``
    One (scenario, planner, heuristic) run.  Returns a
    :class:`~atp.evaluation.metrics.PlanReport` in which the trajectory has been
    re-scored by the cost model, independently of the search.

``run_matrix``
    The cross product of scenarios x planners x heuristics, written to CSV and
    JSON.  Determinism is a property of the whole stack (fixed successor order,
    deterministic tie-breaking, explicitly seeded scenario RNG), so re-running
    the same matrix reproduces the results byte-for-byte apart from the timing
    columns.  ``tests/test_determinism.py`` asserts this.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Sequence

from ..baselines.analytic import plan_analytic_direct
from ..baselines.direct import plan_direct_route
from ..evaluation.metrics import PlanReport, evaluate_trajectory
from ..planning.astar import SearchStatistics, SearchStatus, astar
from ..planning.heuristics import build_heuristic
from ..scenarios.library import get_scenario, random_scenario
from ..scenarios.spec import BuiltScenario, ScenarioSpec, build_scenario

#: Columns excluded from the deterministic-comparison view (wall clock).
NON_DETERMINISTIC_COLUMNS = {"runtime_s"}


def run_plan(
    built: BuiltScenario,
    *,
    heuristic: str = "optimistic",
    weight: float = 1.0,
    max_expansions: int | None = None,
    time_limit_s: float | None = None,
) -> PlanReport:
    """Run A* (or Dijkstra, via ``heuristic="zero"``) on a built scenario."""
    h = build_heuristic(heuristic, built.problem, weight=weight)
    result = astar(
        built.problem,
        h,
        max_expansions=max_expansions,
        time_limit_s=time_limit_s,
        reopen_closed=not h.consistent,
    )
    evaluation = evaluate_trajectory(
        built.cost_model, list(result.path), start_move=built.problem.start_move
    )
    planner = "dijkstra" if heuristic == "zero" and weight == 1.0 else "astar"
    return PlanReport(
        scenario=built.spec.name,
        planner=planner,
        heuristic=h.name,
        turn_model=built.spec.turn_model,
        status=result.status.value,
        search_cost=result.cost,
        evaluation=evaluation,
        statistics=result.statistics,
        heuristic_info=h.describe(),
        path=list(result.path),
    )


def run_baseline(built: BuiltScenario, *, analytic: bool = False) -> PlanReport:
    """A reference trajectory, scored with the same cost model as any plan.

    ``analytic=True`` selects the corner-free straight-line reference instead of
    the Bresenham staircase; see :mod:`atp.baselines.analytic` for why both are
    reported once turns are modelled.
    """
    baseline = (
        plan_analytic_direct(built.problem)
        if analytic
        else plan_direct_route(built.problem)
    )
    if not analytic:
        baseline.evaluation = evaluate_trajectory(
            built.cost_model, baseline.path, start_move=built.problem.start_move
        )
    status = (
        SearchStatus.SOLVED.value
        if baseline.evaluation.feasible
        else "constraint_violation"
    )
    return PlanReport(
        scenario=built.spec.name,
        planner=baseline.name,
        heuristic="none",
        turn_model=built.spec.turn_model,
        status=status,
        search_cost=baseline.evaluation.comparable_cost,
        evaluation=baseline.evaluation,
        statistics=SearchStatistics(),
        heuristic_info={"name": "none", "admissible": True, "consistent": True},
        path=list(baseline.path),
    )


@dataclass
class ExperimentSpec:
    """Declarative description of an experiment matrix."""

    name: str = "default"
    scenarios: list[str] = field(default_factory=lambda: ["empty-cruise"])
    random_seeds: list[int] = field(default_factory=list)
    heuristics: list[str] = field(default_factory=lambda: ["zero", "optimistic"])
    weights: list[float] = field(default_factory=lambda: [1.0])
    include_baseline: bool = True
    #: Also report the corner-free straight-line reference (Milestone 2).
    include_analytic_baseline: bool = False
    max_expansions: int | None = None
    time_limit_s: float | None = None
    #: Milestone 2 ablation axes.  An empty list means "leave the scenario's own
    #: value alone", so an unmodified Milestone 1 experiment document produces
    #: an unmodified Milestone 1 matrix.
    turn_models: list[str] = field(default_factory=list)
    bank_angles_deg: list[float] = field(default_factory=list)
    connectivities: list[int] = field(default_factory=list)
    #: Milestone 3 ablation axis.  Each entry is a list of planning TAS values
    #: that *replaces* the scenario's own speed envelope, keeping its operating
    #: limits.  A single-element list is a fixed-speed arm; the whole list is
    #: the free-choice arm.  Empty means "leave the scenario's envelope alone",
    #: so an unmodified Milestone 1 or 2 experiment document is unaffected.
    speed_sets: list[list[float]] = field(default_factory=list)

    @staticmethod
    def from_dict(doc: dict) -> "ExperimentSpec":
        unknown = set(doc) - set(ExperimentSpec().__dict__)
        if unknown:
            raise ValueError(f"experiment: unknown key(s) {sorted(unknown)}")
        return ExperimentSpec(**doc)

    @staticmethod
    def load(path: str | Path) -> "ExperimentSpec":
        return ExperimentSpec.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )


def _iter_specs(spec: ExperimentSpec) -> Iterable[ScenarioSpec]:
    for name in spec.scenarios:
        yield get_scenario(name)
    for seed in spec.random_seeds:
        yield random_scenario(seed)


def _variants(spec: ExperimentSpec, base: ScenarioSpec) -> Iterable[ScenarioSpec]:
    """Expand a scenario over the Milestone 2 ablation axes.

    Every axis defaults to empty, which yields ``base`` unchanged exactly once.
    """
    turn_models = spec.turn_models or [base.turn_model]
    banks: list[float | None] = list(spec.bank_angles_deg) or [base.max_bank_deg]
    connectivities = spec.connectivities or [base.grid.connectivity]
    speed_sets: list[list[float] | None] = list(spec.speed_sets) or [None]
    for turn_model in turn_models:
        for bank in banks:
            for connectivity in connectivities:
                for speeds in speed_sets:
                    grid = replace(base.grid, connectivity=connectivity)
                    suffix = []
                    if len(turn_models) > 1:
                        suffix.append(turn_model)
                    if len(banks) > 1:
                        suffix.append(f"bank{bank:g}")
                    if len(connectivities) > 1:
                        suffix.append(f"c{connectivity}")
                    if speeds is not None and len(speed_sets) > 1:
                        suffix.append("sp" + "/".join(f"{v:g}" for v in speeds))
                    name = base.name + ("[" + ",".join(suffix) + "]" if suffix else "")
                    yield replace(
                        base,
                        name=name,
                        grid=grid,
                        turn_model=turn_model,
                        max_bank_deg=bank,
                        speed_envelope=_substitute_speeds(base, speeds),
                        start_speed_kt=(
                            base.start_speed_kt
                            if speeds is None or base.start_speed_kt in speeds
                            else None
                        ),
                    )


def _substitute_speeds(
    base: ScenarioSpec, speeds: list[float] | None
) -> dict | None:
    """Replace a scenario's planning speeds, keeping its operating limits.

    The envelope's own ``cruise_tas_kt`` is set to the speed in the new set
    closest to the base envelope's cruise speed, ties going to the slower, so
    the reference selectable speed moves as little as the substitution allows
    and the choice is deterministic.  This is only the *default selectable*
    speed; the speed the fuel-flow law is normalised at belongs to the aircraft
    and does not move at all (see
    :attr:`atp.aircraft.performance.AircraftPerformance.reference_tas_kt`),
    which is what keeps a fixed-speed sweep a controlled comparison.

    Substituting speeds into a scenario that declares no envelope is refused
    rather than guessed at: the operating limits would have to be invented.
    """
    if speeds is None:
        return base.speed_envelope
    if not speeds:
        raise ValueError("a speed set must contain at least one speed")
    if base.speed_envelope is None:
        raise ValueError(
            f"scenario {base.name!r} declares no speed_envelope, so the "
            "speed_sets axis has no operating limits to preserve"
        )
    ordered = sorted(float(v) for v in speeds)
    anchor = float(base.speed_envelope.get("cruise_tas_kt", ordered[0]))
    cruise = min(ordered, key=lambda v: (abs(v - anchor), v))
    return dict(
        base.speed_envelope, planning_tas_kt=ordered, cruise_tas_kt=cruise
    )


def run_matrix(spec: ExperimentSpec) -> list[PlanReport]:
    reports: list[PlanReport] = []
    for base_spec in _iter_specs(spec):
        for scenario_spec in _variants(spec, base_spec):
            built = build_scenario(scenario_spec)
            if spec.include_baseline:
                reports.append(run_baseline(built))
            if spec.include_analytic_baseline:
                reports.append(
                    run_baseline(build_scenario(scenario_spec), analytic=True)
                )
            for heuristic in spec.heuristics:
                for weight in spec.weights:
                    if heuristic == "zero" and weight != 1.0:
                        continue  # weighting a zero heuristic is a no-op
                    # Rebuild so that the blocked-cell cache is not shared
                    # between runs and timing comparisons stay fair.
                    fresh = build_scenario(scenario_spec)
                    reports.append(
                        run_plan(
                            fresh,
                            heuristic=heuristic,
                            weight=weight,
                            max_expansions=spec.max_expansions,
                            time_limit_s=spec.time_limit_s,
                        )
                    )
    return reports


def environment_metadata() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
    }


def write_results(
    reports: Sequence[PlanReport],
    output_dir: str | Path,
    *,
    name: str = "experiment",
    metadata: dict | None = None,
) -> tuple[Path, Path]:
    """Write ``<name>.csv`` and ``<name>.json``; returns both paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r.as_row() for r in reports]

    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    csv_path = out / f"{name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})

    json_path = out / f"{name}.json"
    json_path.write_text(
        json.dumps(
            {
                "name": name,
                "environment": environment_metadata(),
                "metadata": metadata or {},
                "results": rows,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path


def summarise(reports: Sequence[PlanReport]) -> str:
    """Fixed-width text table for the terminal."""
    # `cost` is comparable_cost: inf for an infeasible trajectory, whose
    # priced total would otherwise cover only its feasible segments.
    header = (
        f"{'scenario':<24} {'planner':<15} {'heuristic':<20} {'turn':<9} "
        f"{'status':<20} {'cost':>12} {'dist_nm':>9} {'time_min':>9} "
        f"{'fuel_kg':>9} {'corners':>8} {'charged':>8} {'hdg_deg':>8} "
        f"{'expand':>8}"
    )
    lines = [header, "-" * len(header)]
    partial = False
    for r in reports:
        e = r.evaluation
        mark = ""
        if e.unpriced_segments:
            partial = True
            mark = f"  (*{e.unpriced_segments} segment(s) unpriced)"
        lines.append(
            f"{r.scenario:<24} {r.planner:<15} {r.heuristic:<20} "
            f"{r.turn_model:<9} {r.status:<20} "
            f"{e.comparable_cost:>12.1f} {e.distance_nm:>9.1f} {e.time_min:>9.1f} "
            f"{e.fuel_kg:>9.1f} {e.geometric_corners:>8d} {e.num_turns:>8d} "
            f"{e.total_heading_change_deg:>8.1f} {r.statistics.expansions:>8d}{mark}"
        )
    if partial:
        lines.append(
            "(*) the trajectory violates a hard constraint; its distance, time "
            "and fuel cover only the feasible segments and are not comparable."
        )
    return "\n".join(lines)
