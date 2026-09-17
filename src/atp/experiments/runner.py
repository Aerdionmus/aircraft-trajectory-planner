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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

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
    evaluation = evaluate_trajectory(built.cost_model, list(result.path))
    planner = "dijkstra" if heuristic == "zero" and weight == 1.0 else "astar"
    return PlanReport(
        scenario=built.spec.name,
        planner=planner,
        heuristic=h.name,
        status=result.status.value,
        search_cost=result.cost,
        evaluation=evaluation,
        statistics=result.statistics,
        heuristic_info=h.describe(),
        path=list(result.path),
    )


def run_baseline(built: BuiltScenario) -> PlanReport:
    """Direct-route baseline, scored with the same cost model."""
    baseline = plan_direct_route(built.problem)
    status = (
        SearchStatus.SOLVED.value
        if baseline.evaluation.feasible
        else "constraint_violation"
    )
    return PlanReport(
        scenario=built.spec.name,
        planner=baseline.name,
        heuristic="none",
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
    max_expansions: int | None = None
    time_limit_s: float | None = None

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


def run_matrix(spec: ExperimentSpec) -> list[PlanReport]:
    reports: list[PlanReport] = []
    for scenario_spec in _iter_specs(spec):
        built = build_scenario(scenario_spec)
        if spec.include_baseline:
            reports.append(run_baseline(built))
        for heuristic in spec.heuristics:
            for weight in spec.weights:
                if heuristic == "zero" and weight != 1.0:
                    continue  # weighting a zero heuristic is a no-op
                # Rebuild so that the blocked-cell cache is not shared between
                # runs and timing comparisons stay fair.
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
        f"{'scenario':<20} {'planner':<12} {'heuristic':<20} {'status':<20} "
        f"{'cost':>12} {'dist_nm':>9} {'time_min':>9} {'fuel_kg':>9} {'expand':>8}"
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
            f"{r.scenario:<20} {r.planner:<12} {r.heuristic:<20} {r.status:<20} "
            f"{e.comparable_cost:>12.1f} {e.distance_nm:>9.1f} {e.time_min:>9.1f} "
            f"{e.fuel_kg:>9.1f} {r.statistics.expansions:>8d}{mark}"
        )
    if partial:
        lines.append(
            "(*) the trajectory violates a hard constraint; its distance, time "
            "and fuel cover only the feasible segments and are not comparable."
        )
    return "\n".join(lines)
