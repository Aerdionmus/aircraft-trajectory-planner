"""Trajectory evaluation.

Metrics are recomputed from the state sequence rather than accumulated during
search.  That is intentional: it means a path produced by A*, by Dijkstra, or by
the direct-route baseline is scored by exactly the same code, and a bug in the
search cannot flatter its own results.

Infeasible trajectories
-----------------------
A segment that violates a hard constraint has no defined cost under this model,
so it contributes nothing to the totals.  The consequence matters for
experiments: the distance/time/fuel/cost totals of an infeasible trajectory
cover only its **feasible** segments and are therefore *not* comparable with
those of a feasible one.  ``unpriced_segments`` records how many were dropped
and :attr:`TrajectoryEvaluation.comparable_cost` is ``inf`` unless the whole
trajectory is feasible.  Comparisons between planners must use
``comparable_cost``, never ``cost.total``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..core.units import hours_to_minutes
from ..environment.airspace import GridState
from ..planning.astar import SearchStatistics
from ..planning.cost import CostBreakdown, CostModel, SegmentMetrics


@dataclass
class TrajectoryEvaluation:
    feasible: bool
    num_waypoints: int = 0
    distance_nm: float = 0.0
    time_h: float = 0.0
    fuel_kg: float = 0.0
    risk_exposure: float = 0.0
    max_segment_mean_risk_density: float = 0.0
    soft_penalty: float = 0.0
    level_changes: int = 0
    total_climb_ft: float = 0.0
    #: Segments dropped from the totals because they violate a hard constraint.
    unpriced_segments: int = 0
    mean_ground_speed_kt: float = 0.0
    cost: CostBreakdown = field(default_factory=CostBreakdown)
    hard_violations: list[str] = field(default_factory=list)
    infeasible_reason: str = ""

    @property
    def time_min(self) -> float:
        return hours_to_minutes(self.time_h)

    @property
    def comparable_cost(self) -> float:
        """Cost usable for planner comparison: ``inf`` if any segment is
        infeasible, because the priced total then covers only part of the
        trajectory."""
        return self.cost.total if self.feasible else math.inf

    def as_dict(self) -> dict[str, object]:
        return {
            "feasible": self.feasible,
            "num_waypoints": self.num_waypoints,
            "distance_nm": self.distance_nm,
            "time_h": self.time_h,
            "time_min": self.time_min,
            "fuel_kg": self.fuel_kg,
            "risk_exposure": self.risk_exposure,
            "max_segment_mean_risk_density": self.max_segment_mean_risk_density,
            "unpriced_segments": self.unpriced_segments,
            "comparable_cost": self.comparable_cost,
            "soft_penalty": self.soft_penalty,
            "level_changes": self.level_changes,
            "total_climb_ft": self.total_climb_ft,
            "mean_ground_speed_kt": self.mean_ground_speed_kt,
            "hard_violations": list(self.hard_violations),
            "infeasible_reason": self.infeasible_reason,
            **{f"cost_{k}": v for k, v in self.cost.as_dict().items()},
        }


def evaluate_trajectory(
    cost_model: CostModel, path: list[GridState]
) -> TrajectoryEvaluation:
    """Re-evaluate a state sequence against the cost model."""
    if len(path) < 2:
        return TrajectoryEvaluation(
            feasible=len(path) == 1,
            num_waypoints=len(path),
            infeasible_reason="" if path else "empty path",
        )

    total = SegmentMetrics(feasible=True)
    breakdown = CostBreakdown()
    violations: list[str] = []
    level_changes = 0
    climb_ft = 0.0
    reason = ""

    for a, b in zip(path, path[1:]):
        metrics = cost_model.evaluate(a, b)
        if not metrics.feasible:
            violations.append(f"{a}->{b}: {metrics.infeasible_reason}")
            reason = reason or metrics.infeasible_reason
            continue
        total = total + metrics
        breakdown = breakdown + cost_model.price(metrics)
        if a.il != b.il:
            level_changes += 1
            alt_delta = cost_model.airspace.spec.altitude_ft(
                b.il
            ) - cost_model.airspace.spec.altitude_ft(a.il)
            if alt_delta > 0:
                climb_ft += alt_delta

    return TrajectoryEvaluation(
        feasible=not violations,
        num_waypoints=len(path),
        distance_nm=total.ground_distance_nm,
        time_h=total.time_h,
        fuel_kg=total.fuel_kg,
        risk_exposure=total.risk_exposure,
        max_segment_mean_risk_density=total.mean_risk_density,
        unpriced_segments=len(violations),
        soft_penalty=total.restriction_penalty,
        level_changes=level_changes,
        total_climb_ft=climb_ft,
        mean_ground_speed_kt=total.ground_speed_kt,
        cost=breakdown,
        hard_violations=violations,
        infeasible_reason=reason,
    )


@dataclass
class PlanReport:
    """One planner run: what was asked, what came back, how hard it was."""

    scenario: str
    planner: str
    heuristic: str
    status: str
    search_cost: float
    evaluation: TrajectoryEvaluation
    statistics: SearchStatistics
    heuristic_info: dict[str, object] = field(default_factory=dict)
    #: The planned state sequence. Kept out of :meth:`as_row` because tabular
    #: results files should stay one row per run.
    path: list[GridState] = field(default_factory=list)

    def as_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "scenario": self.scenario,
            "planner": self.planner,
            "heuristic": self.heuristic,
            "status": self.status,
            "search_cost": self.search_cost,
        }
        row.update(self.evaluation.as_dict())
        row.update(self.statistics.as_dict())
        row.update(
            {f"heuristic_{k}": v for k, v in self.heuristic_info.items() if k != "name"}
        )
        return row
