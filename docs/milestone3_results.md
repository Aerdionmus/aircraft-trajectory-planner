# Milestone 3 Results

The permanent engineering record for Milestone 3: what was measured, on what
configuration, and what may and may not be concluded from it.

Every number below was produced by executing the committed code against the
committed configurations. Claims are tagged:

- **[MEASURED]** — produced by a test or experiment run recorded here.
- **[CODE-DERIVED]** — a property established directly by the implementation.
- **[INTERPRETATION]** — an engineering conclusion drawn from the measurements.
- **[LIMITATION]** — a known simplification or a behaviour left uncovered.

Nothing in this document is an operational, certification, flight-safety or
air-traffic-management claim. The package is a synthetic research and
educational trajectory-planning prototype; every aircraft input is invented, and
the atmosphere is the *standard* atmosphere rather than any weather product. See
[`assumptions.md`](assumptions.md).

---

## 1. Objective

Milestone 2 planned over `FlightState(ix, iy, il, ih)` at a single commanded
true airspeed. The aircraft's speed was a constant of the model, so it could not
be traded against anything.

Milestone 3 is **speed as a decision variable on an ISA atmosphere**. The
engineering question is what happens to trajectory cost and turn feasibility
once the planner may choose how fast to fly, with atmospheric conditions and
speed-dependent aircraft performance accounted for.

What is added:

- **an ISA atmosphere** — temperature, pressure, density and speed of sound as a
  deterministic function of altitude;
- **airspeed conversions** — TAS↔Mach and CAS↔TAS on that atmosphere, each with
  a stated domain and an exact inverse;
- **a speed envelope** — a discrete set of selectable planning speeds plus
  CAS and Mach operating limits that narrow the usable TAS band with altitude;
- **speed in the planning state** — `FlightState(ix, iy, il, ih, isp)`, so the
  speed decision participates in the transition graph;
- **speed-dependent performance** — fuel flow and turn geometry evaluated at the
  selected speed;
- **a re-derived admissible heuristic** — because the Milestone 2 inheritance
  argument does not survive the speed decision.

**[CODE-DERIVED]** The A\* implementation is again unchanged and remains
domain-free: `planning/astar.py` imports `heapq`, `time`, `dataclasses`, `enum`,
three `typing` names and `.problem`, and no aerospace module of any kind —
including the new atmosphere and envelope modules. This is asserted by parsing
the module's import statements in
`tests/test_m3_experiments.py::test_the_generic_search_still_imports_no_aerospace_module`
rather than claimed in prose.

**[CODE-DERIVED]** Results remain *optimal within the discretised transition
graph* under the stated cost model. Adding a speed dimension enlarges the graph;
it does not turn a discrete search into a continuous optimal-control solution.

---

## 2. Experimental setup

**[MEASURED]** Execution environment:

```
Python          3.12.3 (CPython)
Platform        Linux-6.18.44-fc-v33-x86_64-with-glibc2.39
Repository      Aerdionmus/aircraft-trajectory-planner
Baseline commit 211664d "Close Milestone 2 with wind-aware turn validation"
```

Commands executed for this document:

```bash
python -m pytest
atp experiment --config configs/experiment_m3_parity.json         --output-dir results
atp experiment --config configs/experiment_m3_speed_sweep.json    --output-dir results
atp experiment --config configs/experiment_m3_objective.json      --output-dir results
atp experiment --config configs/experiment_m3_turn_frontier.json  --output-dir results
```

### Scenarios carrying a speed envelope

**[CODE-DERIVED]** Five library scenarios declare one; every other scenario
keeps the aircraft's implicit single-speed envelope and therefore Milestone 2
semantics exactly.

| scenario | grid | turn model | speeds | weights | isolates |
| --- | --- | --- | --- | --- | --- |
| `turn-limited-fixed-speed` | 40×40 at 12 NM | `gate+cost` | {450} | base | fixed-speed parity |
| `speed-choice` | 10×10 at 12 NM | `gate+cost` | 6 options | time+fuel | cost trade-off |
| `speed-choice-min-time` | 10×10 at 12 NM | `gate+cost` | 6 options | time only | objective sensitivity |
| `speed-choice-min-fuel` | 10×10 at 12 NM | `gate+cost` | 6 options | fuel only | objective sensitivity |
| `speed-turn-frontier` | 10×10 at 5 NM | `gate+cost` | 6 options | base | feasibility frontier |

**[CODE-DERIVED]** The three `speed-choice` variants differ **only** in their
cost weights — grid, endpoints, departure heading, restriction, aircraft, bank
limit, turn model, speed envelope and start speed are asserted equal field by
field in `tests/test_m3_experiments.py`. Any difference between them is
therefore attributable to the objective alone.

**[CODE-DERIVED]** `turn-limited-fixed-speed` differs from the Milestone 2
`turn-limited` scenario **only** by declaring a one-element envelope, asserted
the same way. It is deliberately kept at the original 40×40 extent, because it
exists to demonstrate parity against `turn-limited` rather than to demonstrate
speed, and parity is more informative on a route with the same length and
expansion count as the Milestone 2 scenario it is compared against.

**[CODE-DERIVED]** The four scenarios that *do* demonstrate speed --
`speed-choice` and its two single-objective variants, and
`speed-turn-frontier` -- are deliberately kept small (10×10, a 7-cell route).
A library-wide reachability sweep runs on every registered scenario and its
state space is already multiplied by the number of speed options; at the
original 40×40 extent that sweep alone took 51-82 seconds *per scenario*,
which is what made the full test suite impractical to run. Section 15 measures
the state-space effect directly at the original size instead, on a one-off
instance built for that purpose and not registered as a scenario.

### Why the two speed scenarios use different cell sizes

**[CODE-DERIVED]** This is the point of the pair, and it is the difference
between the two experiment types the brief warns against conflating.

At 25° of bank a 45° fly-by needs `R tan 22.5°` of leg with `R = V²/(g tan φ)`.
On a 12 NM grid the half-leg available is 6.00 NM and the fastest selectable
speed, 480 kt, needs 2.98 NM — so on `speed-choice` **every** speed can fly
**every** turn the grid offers, and turn feasibility is constant across the
decision. On a 5 NM grid the available half-leg is 2.50 NM and the same
arithmetic makes the faster options illegal, which is what `speed-turn-frontier`
exists to exercise.

So `speed-choice` is a **cost-sensitivity** experiment and `speed-turn-frontier`
is a **feasibility** experiment. Neither is presented as the other.

### The cost weights used for the speed scenarios

**[CODE-DERIVED]** `SPEED_WEIGHTS` prices fuel at 3.0 cu/kg rather than the
0.8 cu/kg the other scenarios use. The reason is stated rather than buried: at
0.8 cu/kg the time price dominates across the whole shipped envelope and the
cost-optimal speed sits on the fast boundary, so the trade-off exists but is not
*visible* — every arm would report "pick the maximum". At 3.0 the optimum is
interior. **[LIMITATION]** This is a choice of operating point for a synthetic
study. It is not a claim about any airline's cost index, and a reader should not
read the selected speeds below as representative of real operations.

---

## 3. Architecture

**[CODE-DERIVED]** Placement follows the existing dependency direction,
`core → environment/aircraft → planning → evaluation`:

```
core/atmosphere.py    ISA. Depends on core/units only.
core/speeds.py        TAS/CAS/Mach. Depends on core/atmosphere only.
aircraft/envelope.py  SpeedEnvelope. Depends on core only.
aircraft/performance.py   gains `speed_envelope`, speed-dependent fuel and turns.
planning/state.py     FlightState gains `isp`.
planning/cost.py      evaluate/turn_metrics take a speed index; bound re-derived.
planning/problem.py   successors branch over speed.
evaluation/metrics.py reads the speed schedule off the trajectory.
```

**[CODE-DERIVED]** Three layering properties are asserted by tests that parse
the sources: A\* imports no aerospace module; `core/atmosphere.py` and
`core/speeds.py` import nothing outside `core`; `aircraft/envelope.py` imports
only from `core`.

**[CODE-DERIVED]** Heading and speed both live entirely inside `planning/`,
behind the unchanged projection `π(ix, iy, il, ih, isp) = (ix, iy, il)`, so
`environment/`, `visualization/` and the restriction, risk and blocking caches
continue to speak `GridState` and learned nothing this milestone.

---

## 4. ISA atmosphere and airspeed conversions

### Model

**[CODE-DERIVED]** Two layers: a constant-lapse troposphere to 11 000 m and an
isothermal layer to 20 000 m. Outside that range `isa()` raises
`AtmosphereDomainError` rather than extrapolating a law that does not hold
there. Geopotential and geometric altitude are not distinguished; the
distinction is intentionally omitted rather than quantified, since it sits
below the fidelity already implied by the rest of this synthetic model
(constant mass, a fixed-lapse-rate atmosphere with no deviation, and
representative rather than measured aircraft coefficients).

**[MEASURED]** Reference values, reproduced by `tests/test_atmosphere.py`:

| altitude | T [K] | p [Pa] | ρ [kg/m³] | a [kt] |
| --- | --- | --- | --- | --- |
| sea level | 288.150 | 101325.0 | 1.22500 | 661.48 |
| 30 000 ft | 228.714 | 30089.6 | 0.45831 | 589.32 |
| 41 000 ft | 216.650 | 17873.8 | 0.28741 | 573.57 |

**[MEASURED]** Density and pressure decrease strictly and monotonically over 201
samples spanning the whole modelled range; temperature falls linearly then holds
at exactly 216.65 K; the two layers join continuously to within the genuine
hydrostatic gradient across the join.

**[CODE-DERIVED]** Sea-level density is *derived* from pressure and temperature
through the gas law rather than asserted, so its agreement with the standard's
quoted 1.225 kg/m³ is a consistency check on the gas constant rather than a
tautology.

### Conversions

**[CODE-DERIVED]** The subsonic isentropic stagnation relation and its exact
inverse. Both are strictly increasing over the subsonic range, which is what
makes each uniquely invertible.

**[MEASURED]** Round trips TAS↔Mach, CAS↔TAS and CAS↔Mach reproduce their input
to 1e-12 (Mach) and 1e-9 (CAS) relative, across seven altitudes from sea level
to FL410. CAS equals TAS at sea level to 1e-12, and Mach equals TAS over the
local speed of sound to 1e-12.

**[MEASURED]** A given CAS maps to a strictly increasing TAS with altitude, and
a constant Mach to a strictly decreasing CAS — the behaviour that makes the
envelope's limits altitude-dependent.

**[LIMITATION]** The supersonic (Rayleigh) pitot branch is **not** implemented.
Every conversion raises `SpeedDomainError` at or above Mach 1 rather than
returning a number from the wrong formula. This is reachable in practice: 340 kt
CAS at FL410 is about Mach 1.09.

**[LIMITATION]** Perfect gas, dry air, no instrument or position error, no
CAS/EAS distinction, no ISA temperature deviation.

---

## 5. Speed envelope

**[CODE-DERIVED]** The envelope separates two things:

- **operating limits** — `min_cas_kt`, `max_cas_kt` (a `Vmo` analogue) and
  `max_mach` (an `Mmo` analogue), converted to a TAS band at each altitude
  through the ISA model. The band narrows with height, which is the qualitative
  behaviour real envelopes have.
- **planning speeds** — the finite set of TAS values the search selects between.
  They are given in TAS because TAS is the only airspeed the wind triangle and
  the coordinated-turn equations accept, and because a set expressed in Mach
  could not reproduce a fixed-TAS operating point exactly.

**[MEASURED]** The shipped synthetic jet envelope at FL300 (`min_cas` 200 kt,
`max_cas` 340 kt, `max_mach` 0.82):

| planning TAS | CAS at FL300 | Mach at FL300 | available |
| --- | --- | --- | --- |
| 330 kt | 207.3 | 0.5600 | yes |
| 360 kt | 227.2 | 0.6109 | yes |
| 390 kt | 247.5 | 0.6618 | yes |
| 420 kt | 268.0 | 0.7127 | yes |
| 450 kt | 288.8 | 0.7636 | yes |
| 480 kt | 310.0 | 0.8145 | yes |
| *510 kt* | *331.4* | *0.8654* | **no** — exceeds `max_mach` |

**[CODE-DERIVED]** The limits are not decoration. 510 kt is refused at FL300 by
the Mach limit, 300 kt is refused by the low-speed CAS limit, and 450 kt is
refused at sea level by the high-speed CAS limit — each asserted in
`tests/test_speed_envelope.py`.

**[CODE-DERIVED]** A speed whose limits cannot be evaluated at all — outside the
ISA range, or supersonic — is reported **unavailable**, never clamped. This
matches the package's existing rule that an unproven number blocks rather than
permits.

**[CODE-DERIVED]** Every number in an envelope is synthetic. `describe()`
carries `synthetic: True` so a reporting consumer cannot lose the provenance.

---

## 6. Speed as planning state

### Why the state dimension is necessary

**[CODE-DERIVED]** This is the load-bearing design argument of the milestone,
and it is a correctness argument rather than a symmetry one.

The bank limit constrains the change in **air heading**. Milestone 2 established
that the air heading of a leg is recovered from the wind triangle,
`ψ̂ = (GS·t̂ − w)/TAS`, as a function of the commanded ground track, the wind
**and the true airspeed**. Two legs flown along the same ground track at
different TAS therefore have different air headings.

The consequence: once speed can vary, `ih` alone is no longer a sufficient
statistic for the corner at the next node — the pair `(ih, isp)` is. Dropping
`isp` would make the turn gate and the turn charge depend on information outside
the state, which the `SearchProblem` contract forbids and which would invalidate
both the search and the admissibility argument.

**[MEASURED]** Checked directly rather than argued.
`tests/test_speed_state.py::test_the_corner_charged_at_a_node_depends_on_the_incoming_speed`
holds both ground tracks fixed in an 80 kt crosswind and shows the heading
change moves with the incoming speed while the *track* change does not.
`::test_in_still_air_the_speed_coupling_vanishes` is the control: in zero wind
the same comparison is identical to 1e-12, confirming the effect is the wind
coupling and not an artefact of something else.

**[INTERPRETATION]** In still air the speed dimension of the state is, for turn
purposes, redundant. The shipped `speed-choice` and `speed-turn-frontier`
scenarios are zero-wind, so they exercise the *cost* and *feasibility*
consequences of speed but not this coupling directly; the coupling itself is
exercised by unit test (`test_the_corner_charged_at_a_node_depends_on_the_incoming_speed`)
and, end to end through a real A* search rather than isolated corner math, by
`test_wind_shifts_the_optimal_speed_through_a_full_search`, which shows a
headwind alone moving the cost-optimal speed on a small grid. Neither is a
registered library scenario or a documented experiment, so closing that
remaining gap with a windy multi-speed scenario in the experiment configs is
still follow-on work.

### Representation

**[CODE-DERIVED]** `FlightState(ix, iy, il, ih, isp)`, where `isp` indexes the
envelope's planning speeds and denotes the speed flown on the leg that *arrived*
at the cell. `isp` defaults to `0`, so `FlightState(ix, iy, il, ih)` is the same
tuple it was in Milestone 2 and compares and hashes equal — which is why the
Milestone 2 tests that construct states directly needed no change.

**[CODE-DERIVED]** There is deliberately no `NO_SPEED` sentinel to match
`NO_HEADING`. A speed index is only ever consulted to recover the air heading of
an *incoming* leg, and a state with no incoming leg has no corner to evaluate,
so the value is never read.

**[LIMITATION]** With `turn_model="none"` no corner is ever evaluated, so `isp`
is not load-bearing and merely multiplies the state space by the number of speed
options. The representation is kept uniform anyway — one state type, one
successor generator — and §15 measures what that costs rather than arguing
about it.

---

## 7. Aircraft performance: fuel flow

**[CODE-DERIVED]** Fuel flow is taken proportional to power required, split into
parasite and induced terms, and normalised at the reference speed. With
`x = V/V_ref` and `b` the induced share of power at the reference speed:

```
ff(V, alt) = ff_M2(alt) · [ (1 − b) x³ + b/x ]
```

`ff_M2(alt)` is the **unchanged** Milestone 1/2 linear-in-altitude expression, so
the altitude trend is inherited rather than re-modelled and the two effects are
separable. `b = 0.25` for both shipped aircraft. Synthetic.

**[CODE-DERIVED]** `factor(1) = 1` by construction, and the equality is
short-circuited rather than left to floating point, which is what makes
fixed-speed parity bit-exact rather than merely close.

**[MEASURED]** At FL300 for the medium twin jet:

| TAS | fuel flow [kg/h] | specific range [NM/kg] |
| --- | --- | --- |
| 330 | 1619.7 | 0.2037 |
| 360 | 1771.9 | 0.2032 |
| 390 | 1975.9 | 0.1974 |
| 420 | 2232.7 | 0.1881 |
| 450 | 2544.0 | 0.1769 |
| 480 | 2911.9 | 0.1648 |

**[MEASURED]** The curve has an interior minimum at
`x = (b/(3(1−b)))^{1/4} ≈ 0.577`, i.e. about 260 kt — below the envelope, but
its existence is what makes the curve a trade-off rather than a one-way knob, and
it is verified by perturbation in `tests/test_speed_performance.py`. Maximum
specific range sits at a **different** speed: maximising `x / factor(x)` rather
than minimising `factor(x)` gives `x = (b/(1−b))^{1/4}`, which is a different
expression from the fuel-flow minimum above (no factor of 3 in the
denominator), not merely a different number. For the shipped `b = 0.25` this
happens to reduce to `x = 3^{−1/4} ≈ 0.760`, about 342 kt; that reduction is a
property of this aircraft's coefficient, not the general formula, and
`tests/test_speed_performance.py::test_the_specific_range_formula_is_not_just_the_b_equals_quarter_coincidence`
checks the general form at a different `b` where the two expressions
numerically diverge, so the correct formula is pinned rather than merely
consistent with the one value that happens to coincide with it.

**[INTERPRETATION]** That those two speeds differ is exactly why a minimum-fuel
plan and a minimum-time plan select different speeds, and it falls out of the
power-required split rather than being imposed. No dimensionless "speed penalty"
was introduced anywhere in the cost model.

**[LIMITATION]** This is a shape, not a propulsion model. No thrust/drag
integration, no mass dependence, no altitude/speed cross-coupling, and — the
omission that matters most here — **no compressibility drag rise** near the Mach
limit, so the fastest options are modelled as cheaper than they would really be.
The envelope's Mach limit still removes them where they are illegal, but within
the legal band the model understates the cost of speed.

---

## 8. Turn performance and speed

**[CODE-DERIVED]** The Milestone 2 turn model is used unchanged, evaluated at
the selected TAS:

```
R = V² / (g tan φ)        ω = g tan(φ) / V
```

so the radius grows with the **square** of speed while the turn rate falls as
`1/V`. Nothing in the gate was weakened to accommodate speed: the wind-aware
ground-curvature bound, the fly-by tangent fit and the hard cap are all as
Milestone 2 left them.

**[CODE-DERIVED]** A corner may join two legs at different speeds. The two enter
differently, and both deliberately:

- the **air heading** of each leg is solved at that leg's own TAS, because that
  is what the physics says;
- the **turn itself** is charged at `max(V_in, V_out)`. The model does not
  resolve where in the speed transition the corner is flown, so it is charged at
  the faster of the two, which is conservative *with respect to the modelled
  turn radius and turn rate at that discrete speed pair* — the radius is larger
  (the gate over-blocks rather than under-blocks, matching the package's
  documented bias) and the turn rate is lower (the charge is an upper bound).
  This is not, and is not claimed to be, a physically validated bound on the
  unmodelled acceleration or deceleration manoeuvre a real aircraft would fly
  between the two speeds — that manoeuvre has no model here at all (§9).

**[MEASURED]** With one selectable speed the two coincide and the expression
reduces exactly to Milestone 2's; asserted in
`tests/test_m2_fixed_speed_parity.py::test_turn_metrics_are_identical`.

**[MEASURED]** Turn time through a 45° corner at FL300, medium twin jet, 25° of
bank: 29.2 s at 330 kt rising to 42.4 s at 480 kt.

---

## 9. Speed transition model

**[CODE-DERIVED]** The choice, stated explicitly as the brief requires:

- speed is selected **per planning segment**;
- speed changes occur **only at nodes**;
- the change is **instantaneous and unpriced** — no acceleration or deceleration
  dynamics, no time or fuel charged, no bound on how far the speed may jump
  between adjacent segments.

**[INTERPRETATION]** This is the simplest defensible abstraction, and it was
chosen over a half-modelled alternative deliberately. A bounded index step would
*look* more physical while still not being a longitudinal-dynamics model, and
would have needed its own justification for the bound — replacing an honest
abstraction with a fudge factor.

**[LIMITATION]** The abstraction is **optimistic**. A real aircraft needs time
and distance to change speed, so a trajectory containing speed changes would be
flown slightly later and slightly less efficiently than reported. Nothing here
is a flight-control law or an autothrottle model.

**[CODE-DERIVED]** The exposure is made measurable rather than left unknown:
`TrajectoryEvaluation.speed_changes` counts every segment boundary at which the
selected speed changed -- horizontal-to-horizontal or through a pure level
transition alike, not only between two horizontal legs -- and it is emitted as
an experiment column. Like `geometric_corners` it describes the shape of the
returned trajectory, so it counts the whole sequence rather than only the
priced part.

**[MEASURED]** Speed changes in the shipped free-choice runs: 2 on
`speed-choice`, 2 on `speed-turn-frontier`, 0 on each single-objective variant
(each selects one speed for the whole route, so there is nothing to change
between).

**[INTERPRETATION]** Two changes over an 8-waypoint trajectory is not
negligible. A reader comparing a free-choice result against a fixed-speed one
should treat the difference as an upper bound on the achievable gain, since the
fixed-speed arm pays no unmodelled acceleration cost and the free-choice arm
would.

---

## 10. Heuristics

### The Milestone 2 argument does not survive, and is not reused

**[CODE-DERIVED]** Milestone 2 inherited admissibility from monotone
refinement — no Milestone 2 edge was cheaper than its Milestone 1 projection, so
a bound valid on the smaller graph stayed valid on the larger one.

**That property is false in Milestone 3 and has been explicitly retired.**
Allowing the planner to choose a faster speed is precisely the act of making
some edges cheaper than their fixed-speed values. Invariant 4 in
`planning/cost.py` has been rescoped to the turn model alone, with the reasoning
recorded in place so that a future reader cannot re-derive an unsound
inheritance from it.

**[MEASURED]** The retirement is not theoretical: on `speed-choice` the
free-choice optimum costs 2573.3, below the best fixed-speed arm at 2578.3 (360
kt) and well below the 450 kt arm at 2759.7. Under `c₃ ≥ c₂` that would be
impossible.

### The re-derived bound

**[CODE-DERIVED]** For any feasible transition flown at a selectable speed `V`
with horizontal length `H`, the ground speed satisfies `GS ≤ V + |w|_max`, so
`H/GS ≥ H/(V + |w|_max)` and each priced term is bounded below independently.
Minimising over the speeds the envelope offers:

```
A = min over selectable V of
        ( c_time + c_fuel · ff_min(V) + c_risk · ρ_min ) / ( V + |w|_max )
```

where `ff_min(V)` is the least fuel flow at speed `V` over the enumerated
levels. Climb fuel, turn time, turn fuel, turn risk and soft penalties are all
dropped; every one is non-negative, so the bound only loosens. The heuristic is
then unchanged in form from Milestone 1:

```
h = max(0, A · H_straight + c_dist · hypot(H_straight, V_alt) − K)
```

with `K = constant_cost_offset()` unchanged — the descent fuel credit is a
function of the altitude span and the aircraft, not of the selected speed.

**[CODE-DERIVED]** Two soundness points worth stating because both are places a
plausible-looking shortcut would break:

- **Minimising the quotient, not the parts.** Taking `ff_min` over the whole
  envelope and dividing by the envelope's largest `GS_max` is also sound but
  needlessly loose: it pairs the slowest option's fuel flow with the fastest
  option's ground speed, a combination no transition can realise. Minimising the
  quotient keeps each speed's fuel flow with its own ground-speed bound.
- **Restricting to selectable speeds is sound** because a transition can only be
  flown at a speed inside the limits at both endpoint levels, so every
  realisable speed is in the minimisation set. If that set is empty the
  minimisation falls back to the whole envelope, a superset, which is still a
  lower bound.

**[MEASURED]** With a one-element envelope the minimisation collapses to exactly
the Milestone 2 quotient — bit-identical, not merely within tolerance, asserted
in `tests/test_m2_fixed_speed_parity.py::test_the_lower_bound_and_offset_are_bit_identical`.

### Evidence

**[MEASURED]** `tests/test_admissibility_m3.py`, 60 tests:

- no heuristic declared admissible overestimates, across five environments
  (still air, uniform wind, vortex, hazard, restriction) against the true
  cost-to-go from every state;
- the optimistic bound holds across three turn models × three envelope shapes,
  including one reaching **above** the old cruise speed (the case that breaks
  inheritance) and one entirely below it;
- consistency `h(s) ≤ c(s,s') + h(s') ` holds edge by edge over every generated
  edge in every environment × envelope combination, and `h` vanishes on goal
  states;
- A\* with the optimistic heuristic reproduces the Dijkstra optimum to 1e-9.

**[MEASURED]** The specific failure the re-derivation prevents is constructed
rather than hoped for: with an envelope of {450, 600} kt the naive
cruise-speed bound would be 3000/450 = 6.667 cu/NM while the shipped bound is
3000/600 = 5.000 cu/NM, and the naive value would overestimate.

**[CODE-DERIVED]** The cost-to-go sweep was rewritten for this milestone. A
per-state forward Dijkstra — what Milestones 1 and 2 used — grows with the
fourth power of a quantity the speed dimension just multiplied, and the first
version of this file did not finish in ten minutes. It now enumerates the
forward edges once and runs a single multi-source Dijkstra backwards from the
goal states, computing the same quantity in one pass; the full file runs in
seconds. `::test_the_reverse_sweep_agrees_with_forward_dijkstra` validates the
fast method against the slow one on a case small enough to run both ways, so the
optimisation cannot quietly weaken the property being asserted.

**[CODE-DERIVED]** No Dubins heuristic was added. Milestone 2's section 7
recorded that neither the CS-optimality proof nor the curvature lower bound was
established; nothing in Milestone 3 changes that, and a speed envelope makes the
curvature bound *harder* rather than easier, since the minimum radius must now
hold across the whole envelope. The omission stands.

---

## 11. Parity

### Milestone 1

**[MEASURED]** The original Milestone 1 golden matrix in
`tests/test_milestone1_parity.py` and its numerical parity assertions are
**unchanged**: the 80-row matrix captured from commit `76e9c54` reproduces
exactly, every deterministic column to `rel=1e-12, abs=1e-12`. No Milestone 1
golden data was touched. The file itself was edited in one place unrelated to
that matrix: `test_library_scenarios_default_to_milestone_1_semantics` asserts
an exact set of scenario names that opt into turn dynamics, and that set had to
grow to include the five Milestone 3 scenarios that also opt in (a speed
decision is only meaningful with turn dynamics enabled). That assertion carries
no numerical tolerance to weaken -- it is a name-equality check -- so widening it
to name the new scenarios is a registration update, not a relaxation of the
Milestone 1 parity claim.

**[CODE-DERIVED]** Parity is structural. An aircraft that declares no envelope
gets the implicit singleton `{cruise_tas_kt}`; `models_speed` is then false, the
successor generator's speed loop has one iteration, and the state stays a
`GridState` with `turn_model="none"`. A Milestone 1 configuration cannot
accidentally acquire a speed decision. Schema versions 1 and 2 refuse a
`speed_envelope` key outright.

### Milestone 2 fixed-speed

**[MEASURED]** `tests/test_m2_fixed_speed_parity.py`, 59 tests. The tolerance
claimed is **exact equality**, not a numerical tolerance, because the reduction
is structural:

| quantity | result |
| --- | --- |
| successor states and edge costs | identical, every state, 4 environments × 3 turn models |
| segment metrics, field by field | identical |
| turn metrics | identical |
| heuristic lower bound and offset | bit-identical |
| heuristic value at every state | identical |
| path, cost, expansions, generated, reopened | identical |
| Dijkstra cost and expansions | identical |
| evaluation, every shared column | identical |

**[MEASURED]** At scenario level, `turn-limited` versus
`turn-limited-fixed-speed` under three heuristics:

```
cost        6493.0544  ==  6493.0544
time_h      1.0990151  ==  1.0990151
fuel_kg     2795.8944  ==  2795.8944
expansions  1305       ==  1305
generated   2392       ==  2392
```

**[INTERPRETATION]** Equal cost with *different* expansion counts would mean the
two configurations were searching different graphs and arriving at the same
answer by luck. Equal expansion counts are what make this a structural claim.

**[CODE-DERIVED]** The parity property is stated narrowly on purpose. It is a
statement about the **Milestone 2 speed**, not about any fixed speed: pinning a
different singleton legitimately changes the answer, and
`::test_a_singleton_envelope_at_a_different_speed_is_not_claimed_to_be_parity`
asserts that it does, so the property cannot be read more widely than it holds.

---

## 12. Speed non-degeneracy

**[CODE-DERIVED]** This is a **cost-sensitivity** experiment. On `speed-choice`
every speed is legal and flies every turn, so nothing about feasibility moves.

**[MEASURED]** `configs/experiment_m3_speed_sweep.json`, `speed-choice`,
optimistic heuristic, one variable changed per arm (the speed set):

| arm | cost | time [min] | fuel [kg] | turn time [min] | mean TAS | speed changes | expansions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 330 kt | 2616.1 | 18.54 | 500.4 | 1.46 | 330.0 | 0 | 15 |
| 360 kt | 2578.3 | 17.25 | 509.3 | 1.59 | 360.0 | 0 | 17 |
| 390 kt | 2594.7 | 16.18 | 532.7 | 1.72 | 390.0 | 0 | 18 |
| 420 kt | 2657.0 | 15.28 | 568.4 | 1.86 | 420.0 | 0 | 18 |
| 450 kt | 2759.7 | 14.51 | 615.4 | 1.99 | 450.0 | 0 | 14 |
| 480 kt | 2899.4 | 13.86 | 672.8 | 2.12 | 480.0 | 0 | 14 |
| **free choice** | **2573.3** | 16.94 | 512.8 | 1.59 | 367.7 | 2 | 67 |

All arms fly the same 93.9 NM route with 3 geometric corners.

**[MEASURED]** Every arm differs in time, fuel, total cost **and** turn time —
the four quantities the brief asks for. Time falls strictly with speed; turn
time rises strictly with speed; cost is U-shaped with an interior minimum at
360 kt.

**[MEASURED]** Objective sensitivity, `configs/experiment_m3_objective.json` —
identical instance, only the cost weights change:

| objective | selected TAS | time [min] | fuel [kg] |
| --- | --- | --- | --- |
| minimum time | 480 (only) | 13.86 | 672.8 |
| time + fuel | 360 | 16.94 | 512.8 |
| minimum fuel | 330 (only) | 18.54 | 500.4 |

**[INTERPRETATION]** The selected speed moves with the objective, and the
balanced objective lands strictly between the two single-objective extremes.
This is the multi-objective physical trade-off the milestone asks for rather
than a cosmetic speed field: minimum time goes to the envelope's fast boundary,
minimum fuel goes to the slow end near best specific range, and pricing both
selects something in between. The 4.7-minute spread and the 172 kg fuel spread
are the size of the decision on this (deliberately short) instance; §2 records
why the demonstration scenarios were kept small, and the fixed-speed
*percentage* spread below is the figure that is independent of route length.

**[INTERPRETATION]** The free-choice arm beats **every** fixed-speed arm (2573.3
against a best fixed of 2578.3). The margin is small — 0.2 % — because on this
scenario the route is identical and only the speed schedule differs. It is
nonetheless strictly better, which is the statement that the decision is real.

**[LIMITATION]** The free-choice advantage here is within the range that the
unmodelled acceleration cost (§9, 2 speed changes) could plausibly erase. The
non-degeneracy result rests on the *fixed-speed spread* — 12 % in cost, 34 % in
time, 34 % in fuel — not on the free-choice margin.

---

## 13. Feasibility frontier

**[CODE-DERIVED]** This is a **feasibility** experiment, not a cost-sensitivity
one. It is reported separately for that reason.

### Derivation

From the fly-by tangent fit `R tan(Δψ/2) ≤ ½ min(L_in, L_out)` with
`R = V²/(g tan φ)`, and `min(L_in, L_out) = cell` because on an 8-connected grid
a 45° turn always joins an axis leg to a diagonal one:

```
V* = sqrt( ½ · cell · g · tan(φ) / tan(Δψ/2) )
```

**[CODE-DERIVED]** At 5 NM and 25° of bank this gives **439.5 kt** for a 45°
turn and **282.8 kt** for a 90° turn.

**[MEASURED]** The threshold is measured from the implementation by bisecting
its own `turn_fits` gate to 200 iterations and compared with the derivation;
they agree to 1e-9 relative. Neither number is asserted from the other, and the
derivation is recomputed in the test from the equations rather than imported
from the code.

**[MEASURED]** As an independent cross-check, this formula reproduces the
Milestone 2 frontier table in `turn_model.md` §5: 5 NM / 25° / 450 kt rejected,
5 NM / 30° / 450 kt accepted.

### Measured frontier

**[MEASURED]** `configs/experiment_m3_turn_frontier.json`, `speed-turn-frontier`:

| arm | status | cost | time [min] | fuel [kg] | turns | expansions |
| --- | --- | --- | --- | --- | --- | --- |
| 330 kt | solved | 692.2 | 8.57 | 231.5 | 3 | 14 |
| 360 kt | solved | 675.7 | 8.11 | 239.6 | 3 | 17 |
| 390 kt | solved | 669.6 | 7.74 | 255.0 | 3 | 19 |
| 420 kt | solved | 672.3 | 7.45 | 277.1 | 3 | 22 |
| 450 kt | **unsolvable** | ∞ | — | — | 0 | 2 |
| 480 kt | **unsolvable** | ∞ | — | — | 0 | 2 |
| **free choice** | solved | **662.5** | 7.59 | 256.1 | 3 | 50 |

**[MEASURED]** The frontier falls between 420 and 450 kt, bracketing the
analytic 439.5 kt. The Milestone 2 cruise speed is on the **infeasible** side:
at 450 kt no turn is legal on this grid, the start is pointed east with no legal
successor that turns, and the instance is correctly reported unsolvable after 2
expansions.

**[INTERPRETATION]** This is the sharpest result of the milestone. A fixed-speed
planner at the Milestone 2 operating point reports this instance has no
solution. A speed-aware planner solves it — and it does so not by flying slowly
throughout but by **slowing down for the corners and speeding up on the
straights**: the free-choice trajectory ranges over 360–480 kt with 2 speed
changes and a mean of 399.9 kt, and is cheaper than every fixed-speed arm that
solves at all.

**[INTERPRETATION]** That behaviour is the clearest possible demonstration that
speed participates in the transition graph rather than being applied afterwards.
No post-hoc speed assignment could produce it, because the choice of speed on
one segment is what makes a *turn* on the next segment legal.

**[LIMITATION]** The advantage is overstated by exactly the omission in §9: the
two speed changes are free in this model and would not be free in reality. The
*feasibility* conclusion is unaffected — the 450 kt arm has no trajectory at any
price — but the free-choice cost figure is optimistic.

---

## 14. Determinism

**[MEASURED]** Repeated identical runs of `speed-choice`, `speed-choice-min-fuel`
and `speed-turn-frontier` reproduce the state sequence, the total cost, the
**speed sequence** and every evaluation metric exactly.

**[MEASURED]** A repeated experiment matrix over the seven-arm speed sweep is
identical in every column apart from wall clock.

**[CODE-DERIVED]** The mechanism is the existing one extended by one loop:
successors are generated moves-outer, then **speed**, then level offsets, with
pure level changes last; heap entries remain `(f, h, counter, state)` so
comparison never reaches the payload. The ordering is asserted directly in
`tests/test_speed_state.py::test_successor_order_is_moves_then_speeds_then_levels`
rather than assumed, and with a single-speed envelope it collapses to Milestone
2's moves-then-levels — which is why fixed-speed parity holds expansion for
expansion.

---

## 15. Performance impact

The demonstration scenarios (§2) are deliberately small, so the state-space
effect of the speed dimension is measured here on a separate, larger instance
built for this section only: the `speed-choice` geometry reproduced at the
original 40×40 / 12 NM extent, not registered in the scenario library.

**[MEASURED]** Same machine, optimistic heuristic, 400 000-expansion cap:

| configuration | expansions | generated | runtime | cost |
| --- | --- | --- | --- | --- |
| `turn-limited` (M2, 1 speed, 40×40) | 1305 | 2392 | 0.35 s | 6493.05 |
| `turn-limited-fixed-speed` (M3, 1 speed, 40×40) | 1305 | 2392 | 0.35 s | 6493.05 |
| `speed-choice` geometry, 450 kt fixed, 40×40 | 1299 | 2408 | 0.36 s | 12644.0 |
| `speed-choice` geometry, free choice, 6 speeds, 40×40 | 8267 | 21573 | 12.97 s | 12177.6 |
| `speed-choice` (shipped, 10×10), 450 kt fixed | 14 | 41 | 4 ms | 2759.7 |
| `speed-choice` (shipped, 10×10), free choice | 67 | 371 | 101 ms | 2573.3 |
| `speed-turn-frontier` (shipped, 10×10), 420 kt fixed | 22 | 54 | 6 ms | 672.3 |
| `speed-turn-frontier` (shipped, 10×10), free choice | 50 | 197 | 52 ms | 662.5 |

**[MEASURED]** At the 40×40 size, a six-speed envelope costs about **6.4× the
expansions, 9.0× the generated states and 37× the runtime** against the
fixed-speed arm of the same scenario. At the shipped 10×10 size the ratios are
smaller (4.8× expansions, 9.0× generated on `speed-choice`; 2.3× expansions,
3.6× generated on `speed-turn-frontier`) -- expected, since a short route
reaches the goal before the frontier has had room to grow to its 40×40
proportions, and the smallest counts (2 and 14 expansions) are close enough to
the search's fixed per-call overhead that the ratio is noisy at this scale.
Wall-clock time at the shipped size is single-digit milliseconds and is not a
meaningful comparison; the 40×40 figures are the ones to read for the
state-space effect, and the generated-state ratio (9.0×, matching the speed
count exactly) is the more stable one to trust in both regimes.

**[INTERPRETATION]** The branching factor is multiplied by the number of speed
options and the reachable state space with it, so the generated-state ratio
tracking the speed count is the expected result. The runtime ratio at 40×40
exceeds it because a larger frontier costs more per operation. **[CODE-DERIVED]**
One avoidable part was removed during development: the envelope's CAS and Mach
limits are evaluated through the ISA atmosphere, and doing that per transition
rather than caching it per flight level dominated the loop when first written;
it is now cached by level.

**[INTERPRETATION]** 13 s is acceptable for a research instance of this size and
no further optimisation was attempted, in keeping with the project's preference
against premature optimisation. A reader planning larger instances should expect
the speed dimension to be the dominant cost and should reach for the existing
weighted-A\* knob first.

**[LIMITATION]** Single-run timings on one machine, not a benchmark. Expansion
and generation counts are deterministic and are the figures to compare; runtime
is indicative only, and at the shipped scenario size it is too small to compare
meaningfully at all.

---

## 16. Adversarial review

Performed before the milestone was declared complete. Each item was an attempt
to break the implementation, and the outcome is recorded whether or not it found
anything.

| # | Attack | Outcome |
| --- | --- | --- |
| 1 | Speed-state transitions | No defect. Speeds branch correctly; order asserted. |
| 2 | Milestone 2 parity | No defect. Bit-exact including expansion counts. |
| 3 | Heuristic admissibility | **Defect found and fixed during design**: the Milestone 2 inherited argument is invalid here. Bound re-derived over the envelope. |
| 4 | Heuristic consistency | No defect. Holds edge by edge. |
| 5 | Speed envelope bounds | No defect. Out-of-range indices raise; limits bind where the conversions predict. |
| 6 | ISA calculations | No defect. Matches the standard's own values and laws. |
| 7 | Wind + speed + turn interaction | No defect. A wind stronger than the slowest selectable speed leaves downwind legs flyable and makes the curvature bound infinite, so the gate rejects every turn at that speed — conservative, no crash. |
| 8 | Deterministic ordering | No defect. |
| 9 | Zero-wind behaviour | No defect. The speed/heading coupling correctly vanishes. |
| 10 | Impossible-speed configuration (no speed legal at any level) | No defect. `available_indices` returns empty, no successors are generated, the instance is reported unsolvable, and the heuristic bound falls back to the whole envelope and stays sound. |
| 11 | Impossible-turn configuration (1 NM cells) | No defect. Only straight flight is generated. |
| 12 | Minimum and maximum speed edges | No defect. |
| 13 | State explosion | No defect; quantified in §15. |
| 14 | Existing M1/M2 tests | No regressions. |
| 15 | Start speed outside the operating limits at the start level | **Defect found and fixed.** The start speed defines the air heading of the notional incoming leg, so it is genuinely flown once a departure heading is declared. It was possible to construct a problem whose default start speed was illegal at the start level, and the departure corner was then computed from a speed the aircraft may not use. Now refused explicitly at construction rather than clamped; regression test added. |
| 16 | Fuel-law reference moving with the envelope | **Defect found and fixed.** Attaching a substituted speed envelope also moved the aircraft's cruise TAS, which is the speed the fuel law is normalised at. Every arm of a fixed-speed sweep would then have been a differently-scaled aircraft and the sweep would have compared nothing. `reference_tas_kt` is now separated from the envelope's cruise speed and pinned by a test. |
| 17 | Admissibility sweep passing vacuously | **Defect found and fixed.** The first version of the Milestone 3 sweep used a grid on which the goal cell fell inside the no-fly circle, so no state reached the goal and the test passed having checked nothing. The sweep now asserts a non-zero state count. |
| 18 | A pure level speed change invisible to the next turn evaluation | **Defect found and fixed.** `evaluate_trajectory` updated its tracked incoming speed only on horizontal segments, so a speed selected during a pure level change was never carried forward: the next corner was priced at the speed flown *before* the level change, disagreeing with the planner's own successor generator, which does carry the new speed through `FlightState.isp`. At the sizes involved this does not merely mis-price a trajectory, it can flip a turn from feasible to infeasible. Fixed by updating the tracked speed on every segment, not only horizontal ones; a regression test constructs a grid where the bug's stale speed makes a real turn wrongly infeasible, with a negative control confirming the test actually discriminates the two behaviours. |
| 19 | An unnamed default speed skipping envelope validation | **Defect found and fixed.** `CostModel.evaluate(..., speed_index=None)` resolved to the aircraft's default TAS with no check against the envelope's operating limits, even when a real speed decision existed. An out-of-envelope *default* speed was therefore flown silently, while the identical explicit choice was correctly rejected -- an inconsistency between two paths that should agree. Fixed so an unnamed speed resolves to the envelope's cruise index and is then validated exactly like an explicit one; the Milestone 1/2 behaviour (no envelope, no check at all) is preserved bit-for-bit when there is no real decision to make. |

---

## 17. Limitations

Collected, including those stated above.

**[LIMITATION]** Everything inherited from Milestones 1 and 2 still applies:
flat earth, constant mass, no vertical wind, steady wind sampled once per edge,
8-connectivity length bias, sampled soft-restriction overlap, cell-centre
blocking, no time dimension. See [`assumptions.md`](assumptions.md).

New to this milestone:

**[LIMITATION]** **Speed changes are free.** No acceleration dynamics, no time or
fuel charged, no bound on the jump. Reported free-choice results are optimistic
by that amount; `speed_changes` quantifies the exposure per trajectory.

**[LIMITATION]** **No compressibility drag rise.** The fuel model understates the
cost of the fastest legal speeds.

**[LIMITATION]** **Speed is constant within a segment.** Changes happen only at
nodes.

**[LIMITATION]** **The atmosphere is standard, and only where it is used.** It
drives the airspeed conversions and hence the envelope's operating limits. It
does **not** drive fuel flow, which keeps Milestone 1's synthetic linear altitude
trend. No ISA deviation, no temperature offset, no real weather.

**[LIMITATION]** **No mass depletion**, so no fuel-burn feedback on optimum speed
along the route and no cost-index-style speed drift. Explicitly out of scope.

**[LIMITATION]** **The shipped multi-speed scenarios are zero-wind**, so the
wind/speed/heading coupling that *justifies* `isp` being in the state is
exercised by unit test and by one cheap end-to-end search-level test
(`test_wind_shifts_the_optimal_speed_through_a_full_search`), not by any
shipped scenario or experiment config. A windy multi-speed scenario in the
experiment configs is the obvious next addition.

**[LIMITATION]** **`isp` is redundant under `turn_model="none"`**, multiplying the
state space for no modelling benefit in that configuration.

**[LIMITATION]** **The cost weights for the speed scenarios were chosen** so that
the optimum is interior to the envelope. The selected speeds are a property of
that choice, not of any real operation.

**[LIMITATION]** **Every aircraft number remains synthetic**, including all
envelope limits. The ISA constants are defining values of the standard, not
measurements.

**[LIMITATION]** **Optimality is optimality with respect to this cost model**,
over a discretised graph, in a synthetic environment. Nothing here is a claim
about any real flight, and no certification, airworthiness or operational
approval is claimed or implied.
