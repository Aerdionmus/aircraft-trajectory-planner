# Milestone 2 Results

The permanent engineering record for Milestone 2: what was measured, on what
configuration, and what may and may not be concluded from it.

Every number below was produced by executing the committed code against the
committed configurations. Claims are tagged:

- **[MEASURED]** — produced by a test or experiment run recorded here.
- **[CODE-DERIVED]** — a property established directly by the implementation.
- **[INTERPRETATION]** — an engineering conclusion drawn from the measurements.
- **[LIMITATION]** — a known simplification or a behaviour left uncovered.

Nothing in this document is an operational, certification, flight-safety or
air-traffic-management claim. The package is a synthetic research and
educational trajectory-planning prototype; every input is invented. See
[`assumptions.md`](assumptions.md).

---

## 1. Objective

Milestone 1 planned over `GridState(ix, iy, il)` — horizontal cell and flight
level. A returned trajectory could therefore contain instantaneous 45-degree
heading changes, which no aircraft can fly.

Milestone 2 adds, on top of that state:

- **heading state** — `FlightState(ix, iy, il, ih)`, where `ih` indexes the grid
  move set rather than discretising angle independently;
- **turn feasibility** — a fly-by tangent fit against a bank-limited turn radius,
  which removes corners the aircraft could not fly;
- **turn cost** — the corner charged in time, fuel and risk through the existing
  pricing path, with no new cost weight;
- **wind-aware air-heading handling** — the wind triangle solved for the air
  heading, so that the bank limit constrains the change in *air heading* rather
  than the change in ground track, and a corrected upper bound on the ground-path
  radius of curvature under locally uniform wind.

**[CODE-DERIVED]** The A* implementation itself is unchanged and remains
domain-free: `planning/astar.py` imports `heapq`, `time`, three `typing` names
and `.problem`, and no aerospace module of any kind. Heading enters entirely
inside `planning/`, behind the projection `pi(ix, iy, il, ih) = (ix, iy, il)`
that `environment/`, `evaluation/` and `visualization/` continue to speak.

**[CODE-DERIVED]** Results are described as *optimal within the discretised
transition graph* under the stated cost model. They are not claims about a
continuous optimal trajectory, and the turn model narrows but does not close the
gap between a graph-optimal trajectory and a flyable one.

---

## 2. Experimental Setup

**[MEASURED]** Execution environment:

```
Python         3.12.3 (CPython)
Platform       Linux-6.18.44-fc-v33-x86_64-with-glibc2.39
Repository     Aerdionmus/aircraft-trajectory-planner
Baseline commit 19919bd "Implement Milestone 2 turn-aware trajectory planning"
```

Commands executed for this document:

```bash
python -m pytest
atp experiment --config configs/experiment_parity.json        --output-dir results
atp experiment --config configs/experiment_turn_ablation.json --output-dir results
atp experiment --config configs/experiment_bank_frontier.json --output-dir results
atp experiment --config configs/experiment_wind_turns.json    --output-dir results
```

### Scenarios carrying turn dynamics

**[CODE-DERIVED]** Two library scenarios opt into a turn model; every other
scenario remains on `turn_model: "none"` and therefore on Milestone 1 semantics.

| scenario | grid | levels | wind | turn model | purpose |
| --- | --- | --- | --- | --- | --- |
| `turn-limited` | 40x40 at 12 NM | FL300 | none | `gate+cost` | isolates the cost of manoeuvring |
| `turn-limited-wind` | 40x40 at 12 NM | FL300 | uniform 90 kt from 270 | `gate+cost` | isolates the wind/heading coupling |

**[CODE-DERIVED]** The two differ *only* in the wind field. Grid, endpoints
(`[2,20,0]` to `[37,20,0]`), departure heading (090), restriction (`P-301`,
60 NM circle), aircraft (`medium-twin-jet`, 450 kt, 25-degree bank limit) and
cost weights are identical, which is asserted field by field in
`tests/test_m2_closure.py`. Any measured difference between them is therefore
attributable to the wind alone.

### Why 12 NM cells

**[CODE-DERIVED]** A fly-by arc needs `R tan(dpsi/2)` of leg on each side, and on
an 8-connected grid a 45-degree turn always joins an axis leg to a diagonal one,
so the binding length is the cell size. At 450 kt and 25 degrees of bank the
radius is 6.33 NM, so a 45-degree turn needs 2.62 NM against the 6.00 NM
half-leg a 12 NM cell provides. On the 5 NM grids the Milestone 1 scenarios use,
the same turn needs more leg than is available and **no turn is legal at all** —
which is why those scenarios stay on `turn_model: "none"`. Derivation in
[`turn_model.md`](turn_model.md) section 5.

### Two turn-count quantities

**[CODE-DERIVED]** The evaluator reports two distinct counts, and they must not
be conflated:

- `geometric_corners` — direction changes present in the returned ground track,
  computed from the cell sequence alone by
  `evaluation.metrics.count_geometric_corners`. It consults no cost model, turn
  model or aircraft, so it is identical under all three turn models.
- `num_turns` — corners the *active turn model* evaluated and charged. Zero by
  construction when `turn_model="none"`, because no turn was modelled.

**[LIMITATION]** `geometric_corners` counts the whole returned sequence,
including segments that violate a hard constraint, whereas the priced metrics
cover only feasible segments. It describes the shape of the trajectory that came
back, not the part of it that could be priced.

---

## 3. M1 Parity

**[MEASURED]** `tests/test_milestone1_parity.py` — 12 passed. The golden file
`tests/data/milestone1_parity.csv` was captured from commit `76e9c54`, before
Milestone 2 existed. Replaying `configs/experiment_parity.json` reproduces all
80 rows, and every deterministic column matches to `rel=1e-12, abs=1e-12`.

**[MEASURED]** Running the same configuration as an experiment:

```
rows                             80
distinct turn_model values       {"none"}
rows with num_turns > 0           0
rows with geometric_corners > 0  50
```

**[INTERPRETATION]** The last two lines are the reporting correction working at
Milestone 1 scale. Fifty of the eighty Milestone 1 trajectories contain real
direction changes; all eighty correctly report zero *charged* turns, because no
turn model is active. Before this milestone only the second quantity existed, so
the same runs reported `num_turns = 0` and offered no way to see that the routes
turned at all.

**[CODE-DERIVED]** Parity is structural rather than coincidental: with
`turn_model: "none"` the Milestone 1 successor generator runs untouched and
yields `GridState`, so the successor set and its order are identical.

**[MEASURED]** `test_heuristic_bounds_are_untouched_by_milestone_2` confirms for
every library scenario that `speed_cost_lower_bound_per_nm()` and
`constant_cost_offset()` are numerically identical with and without turn
modelling — as they must be, since both bound cost per NM of *horizontal*
progress and a turn adds no horizontal progress.

---

## 4. Turn Ablation

**[MEASURED]** `configs/experiment_turn_ablation.json`, scenario `turn-limited`.
Solved rows only; both direct-route baselines violate `P-301` by design and are
compliance comparisons, not cost ones.

| turn model | heuristic | cost | dist NM | time min | fuel kg | geom. corners | charged | total dpsi | expansions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | zero (Dijkstra) | 6326.2 | 479.6 | 64.0 | 2711.6 | 5 | 0 | 0.0 | 1427 |
| none | optimistic | 6326.2 | 479.6 | 64.0 | 2711.6 | 16 | 0 | 0.0 | 385 |
| none | octile | 6326.2 | 479.6 | 64.0 | 2711.6 | 9 | 0 | 0.0 | 173 |
| gate | zero (Dijkstra) | 6326.2 | 479.6 | 64.0 | 2711.6 | 7 | 7 | 315.0 | 10317 |
| gate | optimistic | 6326.2 | 479.6 | 64.0 | 2711.6 | 16 | 16 | 720.0 | 1407 |
| gate | octile | 6326.2 | 479.6 | 64.0 | 2711.6 | 13 | 13 | 630.0 | 671 |
| gate+cost | zero (Dijkstra) | 6493.1 | 479.6 | 65.9 | 2795.9 | 3 | 3 | 135.0 | 10184 |
| gate+cost | optimistic | 6493.1 | 479.6 | 65.9 | 2795.9 | 3 | 3 | 135.0 | 1305 |
| gate+cost | octile | 6493.1 | 479.6 | 65.9 | 2795.9 | 3 | 3 | 135.0 | 683 |

**[MEASURED]** Cost composition and turn contribution, A* with `optimistic`:

| turn model | total | time | fuel | distance | turn time min | turn fuel kg | corner cut NM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 6326.2 | 3197.6 | 2169.3 | 959.3 | 0.000 | 0.0 | 0.00 |
| gate | 6326.2 | 3197.6 | 2169.3 | 959.3 | 0.000 | 0.0 | 4.36 |
| gate+cost | 6493.1 | 3297.0 | 2236.7 | 959.3 | 1.988 | 84.3 | 0.82 |

### What this shows

**[MEASURED]** Charging turns raises cost from 6326.2 to 6493.1, **+2.64%**, of
which 1.988 min of turn time (3.0% of the 65.9 min flight) and 84.3 kg of turn
fuel (3.0% of 2795.9 kg). Path length is 479.6 NM in every solved row.

**[MEASURED]** Under `gate` the three planners return **identical cost** with 7,
16 and 13 corners respectively. Under `gate+cost` all three converge on 3 corners
and 135 degrees of heading change.

**[INTERPRETATION]** On a uniform grid, paths with the same cell count have the
same length, so when turns are not priced the *shape* of an optimal path is a
free variable within a large cost-tie class. Which member is returned depends
only on tie-breaking, and tie-breaking depends on the heuristic — hence 7 / 16 /
13. Pricing the turn collapses that tie class and makes the returned shape
well-determined. **This convergence is the substantive Milestone 2 result.**

**[INTERPRETATION]** The `gate` arm costs exactly what the `none` arm costs.
At 12 NM cells the gate removes some corners (a 90-degree axis turn is rejected;
diagonal-to-diagonal is not) but removes nothing that the optimum depended on, so
it is **non-binding at this resolution for this instance**. The whole measured
effect of the turn model here comes from pricing, not from pruning. That is a
statement about this scenario at this cell size, not a general property.

**[INTERPRETATION]** Gating costs roughly an order of magnitude in Dijkstra
expansions (1427 to 10317) because the heading dimension multiplies the reachable
state space by the move-set size. A* absorbs it far better (385 to 1407).

**[LIMITATION]** The `corner_cut_nm` figure of 4.36 NM in the `gate` row is a
diagnostic, never a credit. A fly-by arc is geometrically shorter than the
cornered path it replaces, but crediting that would push Milestone 2 edge costs
below their Milestone 1 values and invalidate the inherited-admissibility
argument. **Reported cost is therefore an upper bound on the cost of the
corresponding fly-by trajectory**, and `corner_cut_nm` measures the size of that
conservatism.

---

## 5. Bank-Angle Sensitivity

**[MEASURED]** `configs/experiment_bank_frontier.json`, scenario `turn-limited`,
A* with `optimistic`, `gate+cost`.

| bank | cost | delta vs 15 deg | time min | fuel kg | turn time min | turn fuel kg | dist NM | corners | total dpsi | expansions |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15 | 6616.6 | — | 67.41 | 2858.3 | 3.460 | 146.7 | 479.6 | 3 | 135.0 | 1157 |
| 20 | 6540.0 | −1.16% | 66.50 | 2819.6 | 2.547 | 108.0 | 479.6 | 3 | 135.0 | 1281 |
| 25 | 6493.1 | −1.87% | 65.94 | 2795.9 | 1.988 | 84.3 | 479.6 | 3 | 135.0 | 1305 |
| 30 | 6461.0 | −2.35% | 65.56 | 2779.7 | 1.606 | 68.1 | 479.6 | 3 | 135.0 | 1542 |
| 35 | 6437.3 | −2.71% | 65.28 | 2767.7 | 1.324 | 56.1 | 479.6 | 3 | 135.0 | 1550 |

### This is a cost-sensitivity experiment, not a feasibility frontier

**[MEASURED]** Across a 20-degree range of bank limit, distance (479.6 NM),
geometric corner count (3), charged turn count (3) and total heading change
(135.0 degrees) are **identical in every row**. What changes is turn time
(3.460 to 1.324 min), turn fuel (146.7 to 56.1 kg), and total cost (−2.71%).

**[INTERPRETATION]** The route does not change. The bank limit is moving only the
`dpsi / omega` time term and the fuel and risk that time attracts, through
`omega = g tan(phi) / V`. At 12 NM cells the gate is far from binding across this
whole range, so no feasibility transition occurs and none is claimed. This
experiment must be described as **bank-angle cost sensitivity over a fixed
route**. Calling it a feasibility frontier would assert a transition the data
does not contain.

**[INTERPRETATION]** Expansions rise monotonically with bank angle (1157 to
1550). A larger bank admits more legal corners, so the branching factor grows;
the optimum is unaffected but more of the graph is explored to prove it.

**[LIMITATION]** A genuine feasibility frontier would require varying bank
against cell size until the tangent fit fails. The measured frontier table in
[`turn_model.md`](turn_model.md) section 5 does exactly that at the unit level
and is the correct reference for feasibility; this experiment is not.

---

## 6. Wind-Bearing Turn Experiment

**[CODE-DERIVED]** Milestone 2's wind-aware work — air-heading recovery from the
wind triangle, signed drift, and the corrected ground-curvature bound — was
covered by unit tests but by **no experiment**, because `turn-limited` has zero
wind and the 5 NM wind scenarios admit no legal turn at 450 kt. Every previously
shipped experiment therefore reported `wind_heading_excess_deg = 0`.
`turn-limited-wind` and `configs/experiment_wind_turns.json` close that gap.

**[MEASURED]** Controlled comparison, A* with `optimistic`, `gate+cost`. All
three heuristics (`zero`, `optimistic`, `octile`) return identical values.

| quantity | `turn-limited` | `turn-limited-wind` |
| --- | ---: | ---: |
| wind | none | uniform 90 kt from 270 |
| waypoints | 36 | 36 |
| distance NM | 479.65 | 479.65 |
| geometric corners | 3 | 3 |
| charged turns | 3 | 3 |
| **total ground-track change** | **135.00 deg** | **135.00 deg** |
| **total air-heading change** | **135.00 deg** | **159.39 deg** |
| **`wind_heading_excess_deg`** | **0.00** | **+24.39** |
| largest single corner (heading) | 45.00 deg | 53.13 deg |
| turn time min | 1.988 | 2.347 |
| turn fuel kg | 84.3 | 99.5 |
| corner cut NM | 0.82 | 1.99 |
| mean ground speed kt | 450.0 | 526.9 |
| time min | 65.94 | 57.01 |
| fuel kg | 2795.9 | 2417.4 |
| comparable cost | 6493.1 | 5743.9 |
| expansions | 1305 | 1238 |

### What this shows

**[MEASURED]** The two runs return the **same ground track** — same 36
waypoints, same 479.65 NM, same three corners, same 135.00 degrees of total track
change — but the aircraft turns through **159.39 degrees of air heading** in wind
against 135.00 in still air. A single corner that changes ground track by 45
degrees requires **53.13 degrees** of heading change.

**[CODE-DERIVED]** `wind_heading_excess_deg` is defined as total heading change
minus total track change, which `tests/test_m2_closure.py` asserts directly. It
is exactly zero in still air, so it measures the wind coupling and nothing else.

**[INTERPRETATION]** This is the quantity a bank limit actually constrains. The
aircraft rolls about its own axis, so a coordinated turn changes *air heading*;
the ground track is what the wind triangle then produces. Treating the two as the
same quantity — which Milestone 1 necessarily did, since it reported drift
unsigned and had no heading state — would under-count the manoeuvre by 18% here
and would under-charge the turn in time, fuel and risk. The measured turn time
rises from 1.988 to 2.347 min and turn fuel from 84.3 to 99.5 kg accordingly.

**[INTERPRETATION]** The corner cut also grows, 0.82 to 1.99 NM, because the
ground-path radius under a wind component along the heading is larger than the
still-air radius. This is the corrected bound
`R_max = (V + |w|)^2 / (V omega)` behaving as derived: the gate becomes *harder*
in wind, never easier, which is the conservative direction for a gate documented
to over-block rather than under-block.

**[MEASURED]** The wind (90 kt) is well inside the `|w| <= TAS` precondition
(450 kt) that the bound requires, asserted in
`test_the_wind_field_is_nonzero_and_inside_the_curvature_precondition`. Outside
that regime the bound returns `inf` and the gate rejects the corner outright.

**[INTERPRETATION]** Cost falls from 6493.1 to 5743.9 and time from 65.94 to
57.01 min because a westerly on an eastbound route is a tailwind; mean ground
speed rises from 450.0 to 526.9 kt. That is the Milestone 1 wind model working,
not a Milestone 2 result, and it is reported here only so the cost difference is
not misread as an effect of the turn model.

**[LIMITATION]** The wind is spatially uniform on purpose, so the node-centre
wind sample the turn model uses is exact and the measured difference is a
property of the wind triangle rather than of the sampling approximation. Across a
sharp shear — a jet-stream edge — the node-centre sample degrades, and that case
is **not** covered by any experiment here.

---

## 7. Heuristic and Search Behaviour

**[MEASURED]** Every heuristic exercised in the ablation reports
`admissible = True` and `consistent = True` under all three turn models, and
`reopened = 0` on all nine solved rows — consistent with the consistency
declaration, since a consistent heuristic never needs a closed node revisited.

**[MEASURED]** Expansions to the same optimal cost, `turn-limited`:

| heuristic | none | gate | gate+cost |
| --- | ---: | ---: | ---: |
| zero (Dijkstra) | 1427 | 10317 | 10184 |
| optimistic | 385 | 1407 | 1305 |
| octile | 173 | 671 | 683 |

**[CODE-DERIVED]** Admissibility and consistency are *inherited* from Milestone 1
rather than re-derived, resting on two facts: every Milestone 2 edge projects
onto a Milestone 1 edge (the gate only removes successors), and
`c2(s, s') >= c1(pi(s), pi(s'))` (turn terms are non-negative and no distance is
ever subtracted). `tests/test_admissibility_m2.py` — 153 tests — runs Dijkstra
from every `FlightState` of small grids across five environments and three turn
models and checks the consistency inequality edge by edge.

**[LIMITATION]** Every heuristic is a function of the projection `pi` alone, so
none can see that an aircraft pointing away from the goal must first spend time
turning round. In a still-air 7x7 test grid, starting pointed away costs nearly
twice as much as starting pointed towards and the heuristic assigns both the same
estimate. The bound is sound but loose.

**[LIMITATION]** A curvature-aware (Dubins) heuristic was derived and
**deliberately not shipped**: the turn-then-tangent length was not proved to be a
*lower* bound on the optimal curvature-bounded path with free terminal heading,
and an unproved lower bound is an unsound admissibility claim.
`tests/test_admissibility_m2.py` asserts no `dubins` entry exists so the omission
cannot be lost.

---

## 8. Interpretation

**[INTERPRETATION]** Taken together, the measurements support these conclusions
and no stronger ones.

1. **Milestone 2 is a kinematic refinement, not a re-routing mechanism.** In the
   one scenario where turns are active and priced, path length is unchanged at
   479.6 NM and cost rises 2.64%. What changes is the *shape* of the route —
   corners fall from 16 to 3 under `optimistic` — and the fact that the returned
   shape is now determined rather than arbitrary.

2. **Pricing, not gating, produced the effect measured here.** The `gate` arm
   costs exactly what `none` costs. At 12 NM cells the gate is non-binding for
   this instance. A reader should not generalise this: at 5 NM the same gate
   forbids every turn.

3. **The wind/heading distinction is real and material.** 24.39 degrees of excess
   heading change on a route whose ground track is unchanged, and an 18% increase
   in turn time and fuel. It is now demonstrated experimentally rather than only
   asserted in unit tests.

4. **Bank angle is currently a price, not a constraint.** Over 15 to 35 degrees
   nothing geometric moves.

5. **Reported cost is an upper bound.** Corner-cutting is measured and never
   credited, so the true fly-by trajectory is cheaper than the figure reported,
   by an amount `corner_cut_nm` quantifies (0.82 NM still air, 1.99 NM in wind).

**[INTERPRETATION]** The corrected reporting is what makes the ablation
interpretable at all. Reading the previous `num_turns` column, the `none` arm
appeared to fly a straight route and gating appeared to *introduce* corners. The
geometric count shows the opposite: the Milestone 1 route already contained 16
corners, gating left them in place, and only pricing reduced them to 3.

---

## 9. Limitations

**[LIMITATION]** Beyond those noted inline:

- **Coverage is narrow.** Turn dynamics are exercised by two scenarios, both at
  12 NM cells, FL300 only, one aircraft, one restriction geometry. No turn-active
  scenario has more than one flight level, so turn/climb interaction is untested
  — and is not modelled in any case.
- **The turn model is a constant-bank coordinated *level* turn.** No roll-in or
  roll-out time, no bank scheduling, no load-factor limit beyond the bank limit,
  and no turn/climb coupling. A real aircraft loses climb performance in a turn.
- **Both legs of a corner use the wind sampled at the node centre**, not at their
  own midpoints. That is what makes the incoming move index a sufficient
  statistic for the state, and it is an approximation of the same order as the
  existing mid-segment sampling.
- **A pure level change has no ground track and therefore no corner.** The
  heading is carried through and nothing is charged. This is a modelling choice,
  not a derivation.
- **Aircraft parameters are synthetic**, chosen to be the right order of
  magnitude. They are not manufacturer data, not from a certified performance
  manual, and not from BADA.
- **Point mass at constant mass, single commanded TAS.** Fuel burn does not
  reduce weight; there is no speed schedule, no atmosphere model, no Mach or CAS
  scheduling, and no thrust-limited ceiling.
- **The heuristic is heading-blind**, as noted in section 7.
- **`turn-limited-wind` now participates in every test parametrised over
  `SCENARIO_LIBRARY`**, which is the intended additional coverage but was a
  consequence of registering it rather than an independent design decision.

---

## 10. M2 Conclusions

**[MEASURED]** Validation state at the time of writing:

```
python -m pytest        501 passed in 263.64s
  of which tests/test_milestone1_parity.py    12 passed
            tests/test_admissibility_m2.py   153 passed
            tests/test_turn_radius_soundness.py 38 passed
            tests/test_m2_experiments.py      38 passed
            tests/test_turn_cost.py           35 passed
            tests/test_heading_state.py       26 passed
            tests/test_turn_geometry.py       23 passed
            tests/test_m2_closure.py          14 passed
```

**[MEASURED]** Milestone 1 parity holds exactly: 80 of 80 golden rows reproduce
to `rel=1e-12`.

**[INTERPRETATION]** Milestone 2 delivered what it set out to deliver. Heading is
in the state with zero representation error, corners are gated against a
bank-limited radius, turns are charged through the existing pricing path without
introducing a dimensionless fudge weight, the air-heading/ground-track
distinction is implemented and now experimentally demonstrated, and the
Milestone 1 admissibility argument survives by projection rather than by
reassertion. The unproved Dubins bound was left out rather than guessed at.

**[INTERPRETATION]** The honest summary of scale: on the one turn-active
scenario, the turn model changes cost by 2.64% and route length not at all. Its
value is in making the returned trajectory kinematically plausible and its shape
determinate, not in finding materially different routes.

**[CODE-DERIVED]** The system remains a synthetic research and educational
prototype. Optimality claims are optimality *within the discretised transition
graph* under a synthetic objective over a synthetic environment. No airworthiness,
certification, operational-approval, flight-safety or air-traffic-management
claim is made or implied.
