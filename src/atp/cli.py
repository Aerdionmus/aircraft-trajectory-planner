"""Command-line entry point: ``python -m atp ...`` or ``atp ...``.

The CLI is a thin adapter over the library.  It contains no planning logic, so
every capability it exposes is reachable (and testable) from Python directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .experiments.runner import (
    ExperimentSpec,
    run_baseline,
    run_matrix,
    run_plan,
    summarise,
    write_results,
)
from .scenarios.library import SCENARIO_LIBRARY, get_scenario, random_scenario
from .scenarios.spec import ScenarioSpec, build_scenario
from .visualization.ascii_map import render_ascii


def _load_scenario(args: argparse.Namespace) -> ScenarioSpec:
    if args.scenario_file:
        return ScenarioSpec.load(args.scenario_file)
    if args.seed is not None:
        return random_scenario(args.seed)
    return get_scenario(args.scenario)


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
