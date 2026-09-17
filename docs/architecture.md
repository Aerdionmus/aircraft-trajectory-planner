# Architecture

## Purpose of this document

Why the code is shaped the way it is, what each layer is allowed to know about,
and what was deliberately left out of the MVP. Modelling simplifications are in
`assumptions.md`; this file is about structure.

## Layering

Dependencies point strictly downward. Nothing below imports anything above.

```
                      cli.py  /  experiments/
                              |
                   scenarios/ (declarative specs -> objects)
                              |
           evaluation/            baselines/
                              |
                          planning/
             (problem, cost, heuristics, astar)
                              |
        environment/                    aircraft/
 (airspace, wind, risk,            (performance,
  restrictions)                     kinematics)
                              |
                            core/
                     (units, geometry)
```

`visualization/` sits outside the stack entirely: it consumes `environment` and
a list of states and returns a string. Nothing depends on it.

The single most important consequence: **`planning/astar.py` imports no
aerospace module.** It is written against the `SearchProblem` protocol in
`planning/problem.py` and is tested on a five-node lettered graph. The airspace
enters only through `TrajectoryPlanningProblem`, which implements the same
protocol. A change to the wind model cannot break the search, and a search bug
cannot be mistaken for a modelling bug.

## Layer responsibilities

### `core/`

`units.py` is the only place conversion factors exist, so a unit error can be
introduced at most once. Every quantity crossing a module boundary carries its
unit in the name (`*_nm`, `*_ft`, `*_kt`, `*_h`, `*_kg`).

`geometry.py` is immutable, dependency-free 2D geometry: `Vec2`, exact
segment/segment and segment/circle intersection, ray-casting point-in-polygon,
and the meteorological wind conversion (reports are "from", the wind triangle
needs "toward").

### `environment/`

`airspace.py` holds the *only* discretisation in the system: a uniform grid of
square cells replicated over an explicitly enumerated set of flight levels. A
state is `GridState(ix, iy, il)`, located at its cell centre. Wind, risk and
restrictions are continuous functions that the grid merely samples, so raising
the resolution strictly improves fidelity rather than changing the model.

`wind.py`, `risk.py` and `restrictions.py` are abstract bases with several
implementations each. Two interface obligations exist because the heuristic
depends on them:

- `WindField.max_magnitude_kt()` must be a sound **upper** bound;
- `RiskField.min_density()` must be a sound **lower** bound.

`tests/test_wind.py` samples every shipped field to check the first empirically.

### `aircraft/`

`kinematics.py` solves the wind triangle and nothing else. `performance.py` is
a frozen dataclass of coefficients plus the functions that consume them. The
functional form is confined here: a BADA-style table could replace it behind the
same interface without touching the cost model.

### `planning/`

`problem.py` defines `SearchProblem` (domain-free) and
`TrajectoryPlanningProblem` (the aerospace instantiation). Successors are
generated in a fixed order, which is half of the determinism guarantee.

`cost.py` evaluates a transition physically (`SegmentMetrics`) and then prices it
(`CostBreakdown`). Splitting evaluation from pricing means the same flown
trajectory can be re-priced under different weights without re-simulating it,
which is what the experiment matrix needs.

`heuristics.py` exposes `admissible` and `consistent` flags, checked by tests
rather than asserted in prose. `astar.py` supplies A*, weighted A* (via
`WeightedHeuristic`) and Dijkstra (via `ZeroHeuristic`) from one implementation.

### `evaluation/`

`evaluate_trajectory` recomputes metrics from the state sequence rather than
accumulating them during search. A*, Dijkstra and the direct-route baseline are
therefore scored by identical code, and a search bug cannot flatter its own
results.

A segment that violates a hard constraint has no cost under this model, so it
contributes nothing to the totals. The totals of an **infeasible** trajectory
therefore cover only its feasible segments and are not comparable with a
feasible trajectory's. `unpriced_segments` records how many were dropped and
`comparable_cost` is infinite unless the whole trajectory is feasible; planner
comparisons must use `comparable_cost`. This matters most for the direct-route
baseline, which is infeasible by design in the restriction scenarios: reading
its truncated `cost.total` as a comparable figure would understate it.

### `scenarios/`, `experiments/`, `cli.py`

Non-blocked origin and destination cells do not imply a solvable instance --
several restrictions can jointly wall off the destination -- so
`scenarios/feasibility.py` floods the actual transition model to answer
reachability, and `random_scenario` uses it to make its solvability claim true
rather than assumed.

A scenario is a JSON document with no code in it, so experiments are
reproducible from a file under version control and a scenario can be reviewed
like any other artefact. Unknown keys are rejected rather than ignored. The
built-in library returns the same `ScenarioSpec` type a file produces, so it is
a convenience and never a second code path. `cli.py` contains no planning logic.

## Determinism

Reproducibility is a property of the whole stack, not of one component:

1. successor generation order is fixed by the connectivity tuple;
2. heap entries are `(f, h, insertion_counter, state)`, so comparison never
   reaches the payload and ties break by smaller `h` then FIFO;
3. scenario randomness uses a private `random.Random(seed)`, never the global
   RNG;
4. dictionaries are insertion-ordered (Python >= 3.7).

`tests/test_experiments.py` asserts that a repeated experiment matrix is
identical apart from wall-clock columns.

## Cost aggregation

Time, fuel, distance and risk are physically incommensurable, so they are not
combined with dimensionless weights. Each weight is an explicit unit price and
the sum is in abstract cost units:

```
cost = c_time [cu/h]   * time_h
     + c_fuel [cu/kg]  * fuel_kg
     + c_dist [cu/NM]  * distance_nm
     + c_risk [cu/exp] * risk_exposure
     + soft_restriction_penalty [cu]
```

This generalises an airline Cost Index. Setting all prices but one to zero
recovers the single-objective planners, which is how the experiment matrix
isolates effects.

Two invariants the search depends on:

1. every edge cost is finite and **non-negative** (descent fuel credits are
   clamped so this cannot be violated);
2. `speed_cost_lower_bound_per_nm()` never exceeds the time+fuel+risk cost per
   NM of horizontal progress, once `constant_cost_offset()` is given back.

Infeasible transitions are reported as infeasible, never as "very expensive".

## Admissibility argument

Over any feasible transition the ground speed is at most
`GS_max = TAS_max + |w|_max`, so one NM of **horizontal** travel takes at least
`1/GS_max` hours, burns at least `ff_min/GS_max` kg and accrues at least
`risk_min/GS_max` exposure. Soft penalties are non-negative. Writing
`A = speed_cost_lower_bound_per_nm()` and `c_dist = distance_cost_per_nm`, for
any trajectory

```
cost >= A * H_path + c_dist * D3_path - K
```

where `H_path` is horizontal path length, `D3_path` is 3D path length and `K` is
`constant_cost_offset()`. Since `H_path >= H_straight` and
`D3_path >= hypot(H_straight, V)`, the admissible heuristic is

```
h = max(0, A * H_straight + c_dist * hypot(H_straight, V) - K)
```

Two decompositions here are load-bearing, and both were got wrong in the first
implementation of this milestone:

- **The speed-derived term and the distance price are charged against different
  lengths.** `A` bounds cost per NM of *horizontal* progress; `c_dist` prices 3D
  length. Multiplying a combined per-NM figure by the 3D straight-line distance
  charges time and fuel against vertical distance too, and vertical travel is
  far slower than `GS_max`. That overestimates whenever the goal flight level is
  constrained.
- **`K` gives back the one cost term that is not proportional to distance:** the
  fuel a descent is credited. Segment fuel is `ff * t + vertical_delta`, and
  `vertical_delta < 0` on a descent, so a descending segment burns less than
  `ff_min * t`. Because a climb costs at least as much fuel as the matching
  descent refunds, no climb/descent cycle can generate fuel and the largest
  credit any trajectory can accumulate is a single monotone descent across the
  altitude span. `K` is zero unless fuel is priced *and* more than one flight
  level exists, which is most configurations. (If a configuration gives descents
  a larger credit than climbs cost, the fuel term is dropped from `A` entirely
  rather than bounded.)

Consistency follows because `h` is a non-negative combination of two metrics,
less a constant, clamped at zero.

`tests/test_heuristics.py` verifies this the hard way — Dijkstra from every state
on small grids, across five environments, with and without a constrained goal
level — and `tests/test_audit_regressions.py` pins the two specific
counterexamples above.

## MVP scope

**In.** Airspace representation; aircraft and planning-problem representation;
hard and soft restricted regions; weighted cost model; heuristic abstraction
with four heuristics plus a weighting wrapper; A* and Dijkstra behind one
implementation; path reconstruction; direct-route baseline; declarative
scenarios; deterministic experiment runner with CSV/JSON output; ASCII
rendering; a unit and integration test suite covering each of the above.

**Deferred**, with the reason:

| Deferred | Why |
| --- | --- |
| Heading in the state, turn radius, bank limits | Multiplies the state space by the number of headings for an MVP whose subject is heuristic search, not kinematics |
| Any-angle smoothing (Theta*, post-hoc string pulling) | Removes grid bias, but the bias is currently *measurable*, which is more useful while the cost model is being validated |
| BADA-style performance tables | Licensing, and the current interface already accommodates a drop-in replacement |
| Mass-varying fuel burn and step-climb optimisation | Requires a mass state and closes a feedback loop the MVP does not need |
| ARA*, JPS, bidirectional search, hierarchical abstraction | Worth doing once the baseline expansion counts are trusted |
| Matplotlib / plotting output | Presentation concern; keeping the sole renderer dependency-free stops plotting libraries leaking into model code |
| Real weather, terrain or traffic ingest | Would imply operational validity the project does not have |
| Multi-aircraft conflict resolution | A different problem class |
| Web UI, ML, deployment infrastructure | Explicitly out of scope |

## Known structural weaknesses

- **Cell-centre blocking** (`Airspace.is_blocked`) can miss a restriction
  smaller than one cell. The transition-level test in `CostModel.evaluate` is
  authoritative; the cell test is only a fast pre-filter. Relevant if you shrink
  a no-fly zone below the cell size.
- **8-connectivity overestimates path length** by up to about 8% versus a true
  straight line. (The ratio for a heading `t` off-axis is
  `cos t + (sqrt(2) - 1) sin t`, maximised at `t = 22.5` degrees at
  `sqrt(1 + (sqrt(2) - 1)^2)`, about 1.082.) This affects absolute distances, not the A*-versus-Dijkstra
  comparison, since both search the same graph.
- **One mid-segment wind sample per edge** is `O(h^2)` accurate in cell size for
  smooth fields and degrades across a sharp jet-stream edge.
- **Soft-restriction overlap is sampled**, not exact. Hard constraints are exact
  *horizontally* for all three region types (circle by point-to-segment distance,
  polygon by containment plus edge intersection, corridor by segment-to-segment
  distance) and conservative *vertically*, since a level-changing transition is
  tested against the whole altitude band its endpoints span. So a hard
  constraint can over-block a climbing transition but never under-block one.
- **`ProximityRisk`** estimates distance-to-region by bisection with eight
  angular probes. It is the weakest numerical component in the package and the
  first thing to rewrite if proximity risk becomes load-bearing.
- **Vertical rate limits couple cell size to level spacing.** At 450 kt a 5 NM
  cell takes 40 s, so a 2000 ft step demands 3000 fpm and is correctly rejected
  as unflyable. The `layered-jetstream` scenario needs 12 NM cells before a climb
  is feasible at all. This is a real constraint, not a bug, but it will surprise
  anyone who shrinks cells without checking.
