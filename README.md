# Aircraft Trajectory Planner

## Overview

Aircraft trajectory planning is a constrained path-planning problem in which a feasible route must be determined between an origin and a destination while accounting for environmental conditions, operational constraints, flight time, and estimated fuel consumption.

This project investigates the application of **A\* heuristic search** to aircraft flight trajectory planning in a simulated airspace environment. The system is designed to evaluate how different heuristic formulations affect trajectory quality, computational efficiency, and adherence to airspace constraints.

The project is developed as an academic Artificial Intelligence case study with a focus on classical heuristic search and its application to aviation systems.

> **Scope.** This is a simulation and teaching package. It models a synthetic
> airspace with synthetic wind, risk and restriction data and a heavily
> simplified aircraft performance model. It is **not** flight-planning software,
> is not validated against any operational system, and carries no airworthiness
> or certification claim. Every aerospace simplification is listed in
> [`docs/assumptions.md`](docs/assumptions.md).

## Objectives

The primary objectives of this project are to:

- Model aircraft trajectory planning as a heuristic search problem.
- Implement A\* search for constrained flight trajectory generation.
- Incorporate environmental and airspace constraints into the planning process.
- Investigate the effect of different heuristic formulations on planning performance.
- Compare A\* against appropriate baseline approaches.
- Evaluate trajectory quality using measurable aviation-oriented criteria.

## AI Approach

The core planning algorithm is **A\* search**:

\[
f(n) = g(n) + h(n)
\]

where:

- `g(n)` represents the accumulated cost from the origin to the current state.
- `h(n)` estimates the remaining cost to the destination.
- `f(n)` represents the total estimated cost of a solution through state `n`.

The project will investigate multiple heuristic formulations, including distance-based and environment-aware approaches.

## Problem Formulation

The simulated environment represents an airspace containing:

- Aircraft origin and destination states
- Navigable airspace
- Restricted or no-fly regions
- Wind conditions
- Aircraft movement constraints
- Cost factors associated with trajectory selection

The planner generates a feasible trajectory while minimizing a defined cost function that may incorporate:

- Estimated fuel expenditure
- Flight time
- Risk-related penalties
- Airspace constraint costs

## Experimental Evaluation

The project will evaluate different planning strategies using metrics such as:

- Total trajectory cost
- Estimated fuel consumption
- Flight time
- Path length
- Constraint violations
- Number of explored states
- Computational runtime

Baseline comparisons will include **Dijkstra's algorithm** and a direct-route baseline where applicable.

Experiments will investigate the effect of:

- Heuristic formulation
- Wind intensity
- Airspace constraint density
- Search-space size

## System Architecture

```text
┌─────────────────────────────────────┐
│        Scenario Generation          │
│  Grid, wind field, airspace limits  │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│     Airspace & Environment Model    │
│  Wind, restricted regions, geometry  │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│         Aircraft Cost Model         │
│   Fuel, time, distance, risk terms   │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│          A* Trajectory Planner      │
│    Search + heuristic evaluation    │
└──────────────────┬──────────────────┘
                   │
          ┌────────┴────────┐
          ▼                 ▼
┌────────────────┐  ┌──────────────────┐
│ Heuristic      │  │ Constraint       │
│ Module         │  │ Checker          │
└────────────────┘  └──────────────────┘
          │                 │
          └────────┬────────┘
                   ▼
┌─────────────────────────────────────┐
│       Trajectory Evaluation         │
│ Cost, fuel, time, path, constraints │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│      Visualization & Analysis       │
│ Trajectory plots and comparisons    │
└─────────────────────────────────────┘
```

The diagram above is the intended layering. Two deviations are deliberate:
visualization is a leaf that takes `(airspace, path)` and returns a string
rather than a pipeline stage, and evaluation re-scores a trajectory from
scratch rather than consuming the planner's own accumulators, so that A\*,
Dijkstra and the baseline are judged by identical code. See
[`docs/architecture.md`](docs/architecture.md).

## Installation

Python 3.10 or later. The core library has **no runtime dependencies**; it must
stay importable and testable without a scientific stack.

```bash
python -m pip install -e ".[dev]"
```

## Usage

```bash
# list the built-in scenarios and what each one isolates
atp scenarios

# plan a trajectory, compare against the direct-route baseline, draw an ASCII map
atp plan --scenario jetstream --heuristic optimistic --with-baseline --render

# Dijkstra (A* with a zero heuristic) as the optimality reference
atp plan --scenario two-no-fly --heuristic zero

# weighted A*: bounded suboptimality, far fewer expansions
atp plan --scenario dense-restrictions --heuristic optimistic --weight 2.5

# a reproducible random instance
atp plan --seed 101

# a scenario from a JSON file
atp plan --scenario-file configs/scenario_two_no_fly.json

# the full experiment matrix -> results/heuristic-comparison.{csv,json}
atp experiment --config configs/experiment_default.json --output-dir results
```

`python -m atp ...` works identically if the console script is not on `PATH`.

From Python:

```python
from atp.scenarios.library import get_scenario
from atp.scenarios.spec import build_scenario
from atp.experiments.runner import run_plan

built = build_scenario(get_scenario("convective-risk"))
report = run_plan(built, heuristic="optimistic")
print(report.evaluation.as_dict())
```

## Tests

```bash
python -m pytest -q
```

The suite covers geometry, the wind triangle, restriction enforcement, the cost
model, the search itself (on hand-built graphs with no aerospace model
involved), heuristic admissibility against a brute-force optimum, end-to-end
planning, scenario validation, determinism and the CLI.

`tests/test_audit_regressions.py` pins the specific defects found in the
Milestone 1 audit: the two heuristic-admissibility failures (vertical distance
charged at the speed-derived rate, and descent fuel credits), the corridor
intersection test that was sampled rather than exact, the partially-priced
totals of an infeasible trajectory, and scenario reachability.

Two properties are asserted rather than claimed:

- **Admissibility.** Dijkstra is run from every state of small grids across five
  environments; no heuristic declared admissible is allowed to overestimate.
- **Determinism.** A repeated experiment matrix must reproduce byte-for-byte
  apart from wall-clock columns.

Reported costs use `comparable_cost`, which is infinite for a trajectory that
violates a hard constraint, because such a trajectory's priced totals cover only
its feasible segments. The direct-route baseline is infeasible by design in the
restriction scenarios, so it is a compliance comparison there, not a cost one.

## Repository layout

```text
src/atp/
  core/           units and dependency-free 2D geometry
  environment/    airspace grid, wind, risk and restricted regions
  aircraft/       performance coefficients and the wind triangle
  planning/       search problem, cost model, heuristics, A*
  baselines/      direct-route reference planner
  evaluation/     trajectory scoring, independent of the planner
  scenarios/      declarative JSON specs and the built-in library
  experiments/    deterministic runner, CSV/JSON output
  visualization/  ASCII rendering (no plotting dependency)
configs/          example scenario and experiment documents
docs/             architecture and assumptions
tests/            unit and integration tests
```

## Status

Milestone 1 (MVP) is complete: airspace and problem representation, hard and
soft restrictions, the weighted cost model, the heuristic abstraction, A* and
Dijkstra, path reconstruction, the direct-route baseline, deterministic
experiments and the test suite. Deferred work, with the reason for each
deferral, is tabulated in [`docs/architecture.md`](docs/architecture.md).
