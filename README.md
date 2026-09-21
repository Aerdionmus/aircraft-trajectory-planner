# Aircraft Trajectory Planner

## Local JSON API

M5.1 provides a backend-only presentation interface for research and educational
use; it is not certified operational aviation software. Install the project
dependencies, then start the development server with:

```bash
python -m atp.api
```

The development API accepts `ATP_API_ORIGINS` as a comma-separated CORS origin
setting. Endpoints include:

- `GET /api/health`
- `GET /api/scenarios`
- `GET /api/scenarios/{scenario_name}`
- `POST /api/plan`
- `POST /api/experiments/run`
- `GET /api/experiments/config`

Example plan request:

```json
{"mode": "dynamic", "algorithm": "astar", "heuristic": "dynamic-optimistic",
 "cells": 3, "temporal_resolution_h": 0.5}
```

## CesiumJS frontend

The M5.2.1 frontend is a TypeScript/Vite/CesiumJS research visualization. Run
the backend and frontend in separate terminals:

```bash
python -m atp.api
cd frontend
npm install
npm run dev
```

The frontend uses `VITE_API_BASE_URL` for the API origin and defaults to
`http://127.0.0.1:8000`. Trajectories are displayed in a clearly labelled
synthetic local ENU-like coordinate system: planner grid cells are nautical
miles east/north of a synthetic reference point, and altitude comes from the
API trajectory. This is an educational visualization, not an operational or
certified flight-planning system. Cesium ion credentials are not required for
the ellipsoid-only globe used here.

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

# Milestone 2: bank-limited turns, with both direct-route references
atp plan --scenario turn-limited --with-baseline --with-analytic-baseline

# force turn dynamics onto any scenario, overriding the bank limit
atp plan --scenario two-no-fly --turn-model gate+cost --bank-deg 30

# Milestone 3: speed as a decision variable
atp plan --scenario speed-choice --with-baseline

# pin a fixed-speed arm (the Milestone 2 operating point)
atp plan --scenario speed-choice --speeds 450

# the grid where only the slower half of the envelope can turn at all
atp plan --scenario speed-turn-frontier
atp plan --scenario speed-turn-frontier --speeds 450   # reports unsolvable

# the Milestone 3 experiments
atp experiment --config configs/experiment_m3_speed_sweep.json    --output-dir results
atp experiment --config configs/experiment_m3_turn_frontier.json  --output-dir results

# the turn-model ablation (none / gate / gate+cost)
atp experiment --config configs/experiment_turn_ablation.json --output-dir results

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

`tests/test_milestone1_parity.py` pins Milestone 1 behaviour against a golden
results file captured from the frozen commit. `tests/test_turn_geometry.py`,
`tests/test_wind_heading.py`, `tests/test_heading_state.py`,
`tests/test_turn_cost.py`, `tests/test_admissibility_m2.py` and
`tests/test_m2_experiments.py` cover Milestone 2.

`tests/test_atmosphere.py`, `tests/test_speeds.py`,
`tests/test_speed_envelope.py`, `tests/test_speed_performance.py`,
`tests/test_speed_state.py`, `tests/test_admissibility_m3.py`,
`tests/test_m2_fixed_speed_parity.py` and `tests/test_m3_experiments.py` cover
Milestone 3.

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
  core/           units, dependency-free 2D geometry, ISA atmosphere, airspeeds
  environment/    airspace grid, wind, risk and restricted regions
  aircraft/       performance coefficients, wind triangle, turn geometry, speed envelope
  planning/       search problem, flight state, cost model, heuristics, A*
  baselines/      direct-route and analytic straight-line references
  evaluation/     trajectory scoring, independent of the planner
  scenarios/      declarative JSON specs and the built-in library
  experiments/    deterministic runner, CSV/JSON output
  visualization/  ASCII rendering (no plotting dependency)
configs/          example scenario and experiment documents
docs/             architecture, assumptions, turn model, milestone results,
                  M3 requirements and traceability
tests/            unit and integration tests
```

## Turn dynamics

Milestone 2 adds an optional heading dimension and a bank-limited turn model.
`turn_model: "none"` is the default and reproduces Milestone 1 exactly; `"gate"`
prunes illegal turns without charging for them; `"gate+cost"` also charges the
turn in time, fuel and risk. Heading is an index into the grid move set, not an
independent angular discretisation, so there is no rounding anywhere.

A bank limit binds harder than it looks: a 450 kt jet at 25 degrees of bank has a
6.33 NM turn radius, and on an 8-connected grid a 45 degree turn always joins an
axis leg to a diagonal one, so the 5 NM grids the Milestone 1 scenarios use admit
**no turn at all** at that bank. That is why they stay on `turn_model: "none"` and
why `turn-limited`, the one scenario with turn dynamics enabled, uses 12 NM cells.
The derivations, the feasibility frontier and the admissibility argument are in
[`docs/turn_model.md`](docs/turn_model.md).

Turn modelling may only *add* cost or *remove* edges, never make an edge cheaper.
That invariant is what lets the Milestone 1 heuristics keep their admissibility
and consistency declarations over the larger state space; it is proved in the docs
and checked edge by edge in the tests.

## Speed

Milestone 3 makes speed a planning decision rather than a constant. The aircraft
carries a synthetic **speed envelope** — a discrete set of selectable true
airspeeds plus CAS and Mach operating limits — and the limits become an
altitude-dependent TAS band through an ISA atmosphere model, so the set of speeds
the planner may choose genuinely changes with altitude.

Speed enters the state as `FlightState(ix, iy, il, ih, isp)`. It is there for a
correctness reason: a bank limit constrains the change in *air heading*, and air
heading depends on TAS through the wind triangle, so once speed can vary the
heading index alone is no longer a sufficient statistic for the next corner.

The decision is real, and it cuts two ways.

**Cost.** On a 12 NM grid where every speed can fly every turn, the fixed-speed
arms span 34% in time (13.9 to 18.5 min) and 34% in fuel (500 to 673 kg). Which
speed is cheapest depends on what you are buying: minimum time selects the fast
end of the envelope, minimum fuel selects near best specific range, and pricing
both selects something in between. That falls out of the power-required curve
rather than being imposed by a penalty term.

**Feasibility.** Turn radius grows with `V^2`, so on a 5 NM grid at 25 degrees of
bank the 45 degree turn stops fitting above about 439.5 kt — a threshold derived
analytically and then measured from the implementation's own gate. The Milestone
2 cruise speed of 450 kt is on the wrong side of it: `speed-turn-frontier`
reports **unsolvable** at a fixed 450 kt and solves with the envelope available,
by slowing for the corners and speeding up on the straights.

An aircraft that declares no envelope gets the singleton `{cruise_tas_kt}`, so
every Milestone 1 and 2 scenario is untouched. The Milestone 2 monotone-refinement
invariant (`c2 >= c1`) is **deliberately retired** — a faster selectable speed
makes an edge legitimately cheaper — and admissibility is re-derived over the
envelope rather than inherited. See
[`docs/milestone3_results.md`](docs/milestone3_results.md).

## Status

Milestone 1 (MVP) is complete and frozen: airspace and problem representation,
hard and soft restrictions, the weighted cost model, the heuristic abstraction,
A* and Dijkstra, path reconstruction, the direct-route baseline, deterministic
experiments and the test suite.

Milestone 2 is complete: heading-aware state, bank-limited turn geometry, the
air-heading solution of the wind triangle, turn-aware successor generation and
evaluation, schema v2 with version 1 migration, the analytic straight-line
reference, turn-model / bank-angle / connectivity experiment axes and
trajectory-shape metrics. `tests/test_milestone1_parity.py` replays an experiment
matrix captured from the frozen Milestone 1 commit and requires every
deterministic column to reproduce exactly.

Milestone 3 is complete: ISA atmosphere, TAS/CAS/Mach conversions, the speed
envelope, speed in the planning state, speed-dependent fuel flow and turn
geometry, a re-derived admissible heuristic, the speed-sweep and
feasibility-frontier experiments, and a lightweight requirements register with a
traceability matrix. Fixed-speed parity against Milestone 2 is exact — path,
cost, metrics **and** expansion counts — and is gated by
`tests/test_m2_fixed_speed_parity.py`.

A curvature-aware (Dubins) heuristic was designed but **not** shipped: its
admissibility could not be proved within the milestone, and an unproved lower
bound is an unsound claim. See [`docs/turn_model.md`](docs/turn_model.md) section
7. Other deferred work, with the reason for each deferral, is tabulated in
[`docs/architecture.md`](docs/architecture.md).
