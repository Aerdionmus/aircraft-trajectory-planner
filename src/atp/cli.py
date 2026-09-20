"""Command-line entry point: ``python -m atp ...`` or ``atp ...``.

The CLI is a thin adapter over the library.  It contains no planning logic, so
every capability it exposes is reachable (and testable) from Python directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .experiments.runner import (
    ExperimentSpec,
    run_baseline,
    run_matrix,
    run_plan,
    summarise,
    write_results,
)
from .planning.cost import TURN_MODELS
from .scenarios.library import SCENARIO_LIBRARY, get_scenario, random_scenario
from .scenarios.spec import ScenarioSpec, build_scenario
from .visualization.ascii_map import render_ascii


def _load_scenario(args: argparse.Namespace) -> ScenarioSpec:
    if args.scenario_file:
        spec = ScenarioSpec.load(args.scenario_file)
    elif args.seed is not None:
        spec = random_scenario(args.seed)
    else:
        spec = get_scenario(args.scenario)
    return _apply_overrides(spec, args)


def _apply_overrides(spec: ScenarioSpec, args: argparse.Namespace) -> ScenarioSpec:
    """Apply the Milestone 2 command-line overrides, if any were given."""
    overrides: dict[str, object] = {}
    if getattr(args, "turn_model", None) is not None:
        overrides["turn_model"] = args.turn_model
    if getattr(args, "bank_deg", None) is not None:
        overrides["max_bank_deg"] = args.bank_deg
    if getattr(args, "max_turn_deg", None) is not None:
        overrides["max_turn_deg"] = args.max_turn_deg
    if getattr(args, "start_heading", None) is not None:
        overrides["start_heading_deg"] = args.start_heading
    if getattr(args, "goal_heading", None) is not None:
        overrides["goal_heading_deg"] = args.goal_heading
    if getattr(args, "speeds", None):
        if spec.speed_envelope is None:
            raise SystemExit(
                f"scenario {spec.name!r} declares no speed_envelope, so --speeds "
                "has no operating limits to preserve; pick a Milestone 3 scenario "
                "or a scenario file that declares one"
            )
        speeds = sorted(float(v) for v in args.speeds)
        anchor = float(spec.speed_envelope.get("cruise_tas_kt", speeds[0]))
        overrides["speed_envelope"] = dict(
            spec.speed_envelope,
            planning_tas_kt=speeds,
            cruise_tas_kt=min(speeds, key=lambda v: (abs(v - anchor), v)),
        )
        if spec.start_speed_kt is not None and spec.start_speed_kt not in speeds:
            overrides["start_speed_kt"] = None
    return replace(spec, **overrides) if overrides else spec


def _add_scenario_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scenario",
        default="empty-cruise",
        help="built-in scenario name (see `atp scenarios`)",
    )
    parser.add_argument(
        "--scenario-file", type=Path, help="path to a scenario JSON document"
    )
    parser.add_argument(
        "--seed", type=int, help="generate a reproducible random scenario instead"
    )
    parser.add_argument(
        "--turn-model",
        choices=list(TURN_MODELS),
        help="override the scenario's turn model ('none' is Milestone 1 behaviour)",
    )
    parser.add_argument(
        "--bank-deg", type=float, help="override the aircraft's maximum bank angle"
    )
    parser.add_argument(
        "--max-turn-deg", type=float, help="hard cap on the heading change per corner"
    )
    parser.add_argument(
        "--start-heading", type=float, help="departure true bearing (0 = north)"
    )
    parser.add_argument(
        "--goal-heading", type=float, help="required arrival true bearing"
    )
    parser.add_argument(
        "--speeds",
        type=float,
        nargs="+",
        metavar="TAS_KT",
        help=(
            "replace the scenario's planning speeds (TAS in knots), keeping its "
            "operating limits; one value pins a fixed-speed arm"
        ),
    )


def cmd_scenarios(args: argparse.Namespace) -> int:
    for name in sorted(SCENARIO_LIBRARY):
        spec = get_scenario(name)
        print(f"{name:<22} {spec.description}")
    return 0


def cmd_export_scenario(args: argparse.Namespace) -> int:
    spec = _load_scenario(args)
    if args.output:
        spec.save(args.output)
        print(f"wrote {args.output}")
    else:
        print(spec.to_json())
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    spec = _load_scenario(args)
    built = build_scenario(spec)
    report = run_plan(
        built,
        heuristic=args.heuristic,
        weight=args.weight,
        max_expansions=args.max_expansions,
        time_limit_s=args.time_limit,
    )
    reports = [report]
    if args.with_baseline:
        reports.insert(0, run_baseline(build_scenario(spec)))
    if args.with_analytic_baseline:
        reports.insert(0, run_baseline(build_scenario(spec), analytic=True))
    print(summarise(reports))
    if args.render:
        print()
        print(
            render_ascii(
                built.airspace, report.path or None, level=built.problem.start.il
            )
        )
    if args.json:
        print(json.dumps([r.as_row() for r in reports], indent=2, default=str))
    return 0 if report.status == "solved" else 1



def cmd_baseline(args: argparse.Namespace) -> int:
    built = build_scenario(_load_scenario(args))
    report = run_baseline(built)
    print(summarise([report]))
    return 0 if report.evaluation.feasible else 1


def cmd_experiment(args: argparse.Namespace) -> int:
    spec = (
        ExperimentSpec.load(args.config)
        if args.config
        else ExperimentSpec(
            name="cli",
            scenarios=sorted(SCENARIO_LIBRARY),
            heuristics=["zero", "euclidean", "optimistic", "octile", "manhattan"],
        )
    )
    reports = run_matrix(spec)
    print(summarise(reports))
    csv_path, json_path = write_results(reports, args.output_dir, name=spec.name)
    print(f"\nwrote {csv_path}\nwrote {json_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atp",
        description=(
            "Risk- and fuel-aware aircraft trajectory planning with A* search. "
            "Research/teaching simulator -- not for operational use."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("scenarios", help="list built-in scenarios").set_defaults(
        func=cmd_scenarios
    )

    p_export = sub.add_parser(
        "export-scenario", help="print or save a scenario as JSON"
    )
    _add_scenario_args(p_export)
    p_export.add_argument("--output", type=Path)
    p_export.set_defaults(func=cmd_export_scenario)

    p_plan = sub.add_parser("plan", help="plan a trajectory")
    _add_scenario_args(p_plan)
    p_plan.add_argument(
        "--heuristic",
        default="optimistic",
        choices=["zero", "euclidean", "optimistic", "octile", "manhattan"],
    )
    p_plan.add_argument(
        "--weight", type=float, default=1.0, help="weighted A* factor (>= 1)"
    )
    p_plan.add_argument("--max-expansions", type=int)
    p_plan.add_argument("--time-limit", type=float)
    p_plan.add_argument("--with-baseline", action="store_true")
    p_plan.add_argument(
        "--with-analytic-baseline",
        action="store_true",
        help="also report the corner-free straight-line reference",
    )
    p_plan.add_argument("--render", action="store_true", help="ASCII map of the route")
    p_plan.add_argument("--json", action="store_true")
    p_plan.set_defaults(func=cmd_plan)

    p_base = sub.add_parser("baseline", help="direct-route baseline only")
    _add_scenario_args(p_base)
    p_base.set_defaults(func=cmd_baseline)

    p_exp = sub.add_parser("experiment", help="run an experiment matrix")
    p_exp.add_argument("--config", type=Path, help="experiment JSON document")
    p_exp.add_argument("--output-dir", type=Path, default=Path("results"))
    p_exp.set_defaults(func=cmd_experiment)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
