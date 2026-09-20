# The turn model

Milestone 2 adds heading to the planner state and a bank-limited turn model to
the cost model. This document is the derivation; `assumptions.md` lists what the
model does not do, and `architecture.md` says where the code lives.

## Scope, stated once

This is a simulation and teaching package. The turn model makes a returned
trajectory *kinematically plausible for a point-mass aircraft with a stated bank
limit*. It is not a flight-dynamics simulation, the aircraft parameters remain
synthetic and representative rather than manufacturer-sourced, and nothing here
carries any airworthiness, certification or operational claim.

## 1. State

```
FlightState(ix, iy, il, ih)        ih in {0, ..., K-1} or NO_HEADING
```

`ih` indexes `GridSpec.moves`: it is the ground track just flown to arrive at the
cell, not an independent angular discretisation. On a grid the representable
tracks *are* the move set, so indexing it gives zero heading-representation error
by construction, makes wraparound modular arithmetic on indices, and keeps the
state a hashable `NamedTuple` that `planning/astar.py` consumes unchanged.

`NO_HEADING` is the start sentinel: a state with no incoming leg, hence no
corner, no gate and no charge.

The projection `pi(ix, iy, il, ih) = GridState(ix, iy, il)` is used by every
environment, restriction, risk, blocking-cache and evaluation call, so nothing
below `planning/` learns about heading.

**`MOVES_16` is not uniformly spaced.** Its knight moves sit at
`atan2(1, 2) = 26.565` degrees, so consecutive headings alternate 26.565 and
18.435 degree steps rather than `360 / 16 = 22.5`. `TurnTable` therefore computes
every angle from the actual move vectors. Anything that assumes
`2 pi |i - j| / K` is wrong for that move set; `tests/test_heading_state.py`
pins it.

## 2. Air heading from the wind triangle

The planner commands a **ground track**; the heading is a consequence.
Rearranging `GS * t_hat = TAS * psi_hat + w`:

```
psi_hat = (GS * t_hat - w) / TAS
```

and this is exactly a unit vector whenever the solution is feasible. With
`a = w . t_hat` and `c = |w - a t_hat|`:

```
|GS t_hat - w|^2 = GS^2 - 2 GS a + |w|^2
                 = (GS - a)^2 + c^2            since |w|^2 = a^2 + c^2
                 = (TAS^2 - c^2) + c^2 = TAS^2
```

so `|psi_hat| = 1`. No trigonometry, no quadrant handling, and exactly consistent
with the ground speed the cost model already uses. Checked to 1e-12 over a random
sweep including `|w| > TAS` in `tests/test_wind_heading.py`.

`drift_angle_deg` is now **signed** — the angle from air heading to ground track,
so `track = heading + drift` — with the same magnitude `asin(c / TAS)` Milestone 1
reported unsigned. Milestone 1 could not tell a left crab from a right one.

**This matters because a bank limit constrains the change in air heading, not the
change in ground track, and in wind the two differ.** That difference is reported
as `wind_heading_excess_deg` and is exactly zero in still air.

## 3. Turn geometry

Coordinated level turn at bank `phi`, true airspeed `V`:

```
tan(phi) = V^2 / (g R)    =>    R = V^2 / (g tan phi)
omega    = V / R          =>    omega = g tan(phi) / V
```

With `V` in knots and `g = G_NM_PER_H2 = 9.80665 / 1852 * 3600^2 = 68625.369`
NM/h^2, `R` comes out in NM and `omega` in rad/h:

```
kt^2 / (NM/h^2)   = (NM/h)^2 h^2 / NM = NM
(NM/h^2) / (NM/h) = 1/h
```

*Wind translates, it does not rotate.* In a spatially uniform wind the ground
velocity is `v(t) = V psi_hat(t) + w` with `w` constant, so the ground track of a
constant-bank turn is a trochoid, not a circle -- its radius of curvature varies
continuously through the turn.

**Correction.** An earlier version of this document, and of `aircraft/turn.py`,
claimed the ground-frame turn rate equals the air-frame rate and that the
ground-path radius is simply `GS / omega`, using the larger of the two legs'
endpoint speeds. That is false in general and was not conservative: with
`a = w . psi_hat` the wind component along the *instantaneous* heading, the
radius of curvature at that instant is

```
R(t) = GS(t)^3 / (V omega (V + a))
```

which equals `GS(t) / omega` only when `a = 0`. In a tailwind component
(`a > 0`) the true radius is *larger* than `GS(t) / omega`, so the old formula
under-estimated the space a turn needs -- the wrong direction for a gate
documented elsewhere as over-blocking rather than under-blocking. Numerically,
at `V = 450` kt with a 100 kt wind aligned with the heading at the point of
tightest curvature, the true radius is about 22% larger than `GS / omega`
computed from either leg's endpoint speed, and the error grows with wind
strength.

**Fix.** Bounding over every heading the turn could pass through, not just the
two leg endpoints: writing `x = cos(theta)` for the cosine of the angle between
`w` and the instantaneous heading, `R(x)` above is non-decreasing in `x` on
`[-1, 1]` whenever `|w| <= V` (the sign of its derivative reduces to
`2 V^2 + V |w| x - |w|^2 >= 0`, which holds throughout that range under that
precondition), so its maximum is attained at `x = 1`, heading aligned with the
wind:

```
R_max = (V + |w|)^2 / (V omega)
```

This is `ground_curvature_radius_bound_nm` in `aircraft/turn.py`. It reduces to
the exact still-air radius `V / omega` when `w = 0` -- so it changes nothing for
the only turn-active library scenario, `turn-limited`, which uses zero wind --
and it is checked against a finite-difference computation of the true
instantaneous curvature in `tests/test_turn_radius_soundness.py`. Outside the
`|w| <= V` precondition (a wind stronger than the aircraft, not reachable by any
wind field or aircraft shipped in this package) the function returns `math.inf`,
which makes the gate reject the corner outright rather than accept it on an
unproven number.

Values from these equations (analytic, reproduced by `tests/test_turn_geometry.py`):

| TAS | bank | R | omega | 45 deg turn | tangent `R tan 22.5` |
| --- | --- | --- | --- | --- | --- |
| 450 kt | 15 deg | 11.01 NM | 0.65 deg/s | 69.2 s | 4.56 NM |
| 450 kt | 20 deg | 8.11 NM | 0.88 deg/s | 50.9 s | 3.36 NM |
| 450 kt | 25 deg | 6.33 NM | 1.13 deg/s | 39.8 s | 2.62 NM |
| 450 kt | 30 deg | 5.11 NM | 1.40 deg/s | 32.1 s | 2.12 NM |
| 450 kt | 35 deg | 4.21 NM | 1.70 deg/s | 26.5 s | 1.75 NM |
| 280 kt | 25 deg | 2.45 NM | 1.82 deg/s | 24.7 s | 1.01 NM |

A 450 kt jet cannot achieve ICAO rate-one (3 deg/s) within a 25 degree bank. That
is correct: rate-one is a low-speed construct.

## 4. Legality

Two gates at each corner, both configurable per scenario.

**G1, fly-by tangent fit.** An arc of radius `R` tangent to both legs needs a
tangent length `T = R tan(dpsi / 2)` back from the corner along each leg, and only
half of each leg may be claimed because the neighbouring corners need their own:

```
R_fit tan(dpsi / 2) <= 0.5 min(L_in, L_out)      R_fit = (V + |w|)^2 / (V omega)
```

where `|w|` is the wind magnitude sampled at the node (see section 3 for the
derivation and correction of `R_fit`; the tangent fit does not otherwise change).

**G2, hard cap.** `dpsi <= max_turn_deg`, default 180 (inactive).

Degenerate cases, all explicit:

- `dpsi < 0.5 deg` — straight flight: legal, free, still reported.
- `NO_HEADING` — no incoming leg: no gate, no charge.
- **pure level change** — no horizontal displacement, so no ground track and no
  corner: the heading is carried through unchanged and nothing is charged. This
  is a modelling choice, not a derivation.
- `dpsi = 180 deg` — `tan(90 deg)` is infinite and is handled as such, so a course
  reversal is always rejected rather than overflowing.

**Ordering with the vertical-rate check.** The required climb rate is computed on
the **straight-leg time only**; turn time is added afterwards. Folding turn time
in first would make the aircraft appear to have longer to climb and would admit an
unflyable level change. `tests/test_turn_cost.py` constructs the exact case: an
8 NM leg at 450 kt takes 64 s, so a 2000 ft step needs 1875 fpm against an 1800 fpm
limit and must be rejected — a 45 degree turn would add about 40 s and drop the
apparent requirement to roughly 1160 fpm.

## 5. The feasibility frontier

**Correction to the Milestone 2 design review.** The review's frontier table
claimed a 45 degree turn is "accepted at a diagonal corner" on the 5 NM grids.
That is wrong. On an 8-connected grid a 45 degree turn *always* joins an
axis-aligned leg to a diagonal one, so the binding quantity is
`min(L_in, L_out) = cell_size`, never the diagonal. Measured (and pinned in
`tests/test_heading_state.py`):

| cell | bank | 45 deg | 90 deg |
| --- | --- | --- | --- |
| 5 NM | 25 deg | rejected | rejected |
| 5 NM | 30 deg | **accepted** | rejected |
| 12 NM | 25 deg | accepted | mixed (diagonal-to-diagonal only) |
| 3 NM | 25 deg | rejected | rejected |
| 3 NM, turboprop | 25 deg | accepted | rejected |

So at 5 NM and 25 degrees of bank **nothing but straight flight is legal** — a
stronger result than the review predicted, and the reason every Milestone 1
scenario stays on `turn_model: "none"`. The one built-in scenario with turn
dynamics, `turn-limited`, uses 12 NM cells, where a 45 degree turn needs 2.62 NM
against 6.00 NM available.

A related consequence, pinned in `tests/test_admissibility_m2.py`: a start in a
corner cell pointed out of the airspace has *no* legal successor, and the instance
is correctly reported unsolvable.

## 6. Cost

Additive, and strictly non-negative:

```
dt_turn    = dpsi / omega(alt)                              [h]
dfuel_turn = ff(alt) * dt_turn * turn_fuel_factor           [kg]
drisk_turn = rho(node) * dt_turn                            [exposure]
ddist_turn = 0                                              [NM]
```

priced through the existing `CostModel.price`, so no new weight is introduced —
a `turn_cost_per_turn` knob would be the dimensionless fudge `cost.py` argues
against. Risk is a *rate* in this model, so time spent turning genuinely accrues
exposure; omitting it would let the planner buy free time inside a hazard.

The turn is charged on the **outgoing** transition, which keeps the cost a pure
function of `(s, s')` as `SearchProblem` requires, and makes the evaluator's
triple-based recomputation line up with the search's accumulation exactly
(agreement measured at ~3e-12).

### Why corner-cutting is measured but never credited

A fly-by arc replaces two tangents of total length `2R tan(dpsi/2)` with an arc of
length `R dpsi`, and since `dpsi/2 < tan(dpsi/2)` on `(0, pi)` the arc is
**shorter**:

```
dL = R dpsi - 2 R tan(dpsi/2) < 0
```

Crediting that would (a) produce Milestone 2 edge costs below their Milestone 1
values, breaking invariant 4 and with it the inherited-admissibility argument;
(b) potentially produce negative edge costs, breaking Dijkstra outright; and
(c) make the cost of `(s, s')` depend on the *next* leg's length, which is not
available when the edge is generated.

So legality uses the fly-by geometry and cost uses the cornered geometry plus a
non-negative charge. The consequence, stated plainly: **the planner's reported
cost is an upper bound on the cost of the corresponding fly-by trajectory.**
`corner_cut_nm` reports the size of that conservatism so it is measurable rather
than unknown.

## 7. Heuristics

Write `G1` for the Milestone 1 graph, `G2` for the Milestone 2 one, and
`pi: S2 -> S1` for the projection. Two facts hold by construction:

- **P1** every edge of `G2` projects onto an edge of `G1` — the gate only removes
  successors, never invents moves outside `GridSpec.moves`;
- **P2** `c2(s, s') >= c1(pi(s), pi(s'))` — invariant 4.

**Theorem 1 (admissibility is inherited).** Let `P` be an optimal `G2` path from
`s` to the goal. By P1 its projection is a walk to the goal in `G1`; by P2 that
walk costs at most `h*_2(s)`. All costs are non-negative so no walk beats the
optimal path, giving `h(pi(s)) <= h*_1(pi(s)) <= c1(pi(P)) <= h*_2(s)`.

**Theorem 1' (consistency is inherited).** For any `G2` edge,
`h(pi(s)) <= c1(pi(s), pi(s')) + h(pi(s')) <= c2(s, s') + h(pi(s'))`, and `h` still
vanishes on goal states.

Consequently `speed_cost_lower_bound_per_nm()` and `constant_cost_offset()` are
**unchanged** — they bound cost per NM of *horizontal* progress, and a turn adds
no horizontal progress. `zero`, `euclidean`, `optimistic` and `octile` keep their
declarations; `manhattan` stays the inadmissible control.

Both theorems are asserted, not claimed: `tests/test_admissibility_m2.py` runs
Dijkstra from every `FlightState` of small grids across five environments and
three turn models, and checks the consistency inequality edge by edge. **The
argument is fragile**: it rests entirely on invariant 4, so any future change that
credits corner-cutting, smooths a path after the fact, or otherwise lowers an edge
cost invalidates every admissibility flag in the package.

### What inheritance costs

`h` is a function of `pi` alone, so it cannot see that an aircraft pointing away
from the goal must first spend time turning round. In a still-air 7x7 test grid,
starting pointed away costs nearly twice as much as starting pointed towards, and
the heuristic assigns both the same estimate. That is sound but loose, and it is
the motivation for a heading-aware bound.

### The Dubins heuristic was not implemented

The design review proposed a `DubinsLowerBoundHeuristic`. It is **deliberately
absent**.

What was derived and is correct: for a curvature-bounded path from `(p, psi)` with
minimum radius `R`, the turning circle centre is `c = p + s R perp(psi_hat)` for
`s = +1` (left) or `-1` (right); the tangent point to an external target `q` lies
at `T - c = R * rotate(u, -s alpha)` with `u = (q - c)/d`, `d = |q - c|` and
`alpha = arccos(R/d)`; the tangent length is `sqrt(d^2 - R^2)`; and the arc angle
is `((angle(T - c) - angle(p - c)) * s) mod 2 pi`. A candidate lower bound on the
ground-path radius was proposed as `R_lb = V(V - |w|_max) / (g tan phi_max)`, on
the claim that the ground turn rate equals the air-frame rate and `GS >=
V - |w|_max`.

**That claim is false and this bound is unsound.** Section 3 derives the correct
instantaneous radius, `R(t) = GS(t)^3 / (V omega (V + a))` with `a` the
along-heading wind component; its true minimum over a full sweep is
`(V - |w|_max)^2 / (g tan phi_max)`, which is *smaller* than the `R_lb` proposed
above by a factor of `(V - |w|_max) / V`. So the ground-path radius can in fact
dip below the claimed `R_lb`, which was never a sound lower bound. This does not
change the conclusion below -- the Dubins heuristic was correctly left
unimplemented -- but it means even the curvature bound that section would have
needed was not established either, reinforcing rather than weakening the
decision to omit it.

What was **not** established: that this turn-then-tangent (CS) length is the
*optimal* curvature-bounded path to a point with free terminal heading. Admissibility
requires a **lower** bound on the flown length, but minimising over a subset of
path families yields an **upper** bound on the true optimum — the wrong direction.
The case where the target lies strictly inside one of the two turning circles
(reachable only by swinging round on the other circle, and not obviously optimal
against CC-type alternatives) was not settled within this milestone.

Shipping `admissible = True` on an unproved claim is precisely the failure mode the
Milestone 1 audit caught. The heuristic is therefore omitted rather than guessed
at, `optimistic` remains the unconditionally-safe default, and
`tests/test_admissibility_m2.py` asserts that no `dubins` entry exists so the
omission cannot be lost. Closing the CS-optimality proof, or replacing it with a
different provable curvature bound, is Milestone 3 work.

## 8. Speed (Milestone 3)

Everything above is stated at a single true airspeed because Milestone 2 had
one. Milestone 3 evaluates the same relations at the *selected* TAS, which
changes nothing in the derivations and a great deal in the consequences: `R`
scales with `V^2` and `omega` with `1/V`, so the speed decision moves turn
feasibility faster than it moves anything else in the model.

Two points specific to a corner joining legs at different speeds:

- each leg's **air heading** is solved at that leg's own TAS, which is what makes
  the pair `(ih, isp)` rather than `ih` alone the sufficient statistic for a
  corner (see `planning/state.py`);
- the **turn itself** is charged at `max(V_in, V_out)`. The model does not
  resolve where in the speed transition the corner is flown, so charging the
  faster of the two is conservative *with respect to the modelled turn radius
  and turn rate at that discrete speed pair*: a larger radius (the gate
  over-blocks, matching the bias documented in section 4) and a lower turn
  rate (the charge is an upper bound). This is not a claim about the unmodelled
  acceleration/deceleration manoeuvre a real aircraft would fly between the two
  speeds -- that manoeuvre has no physical model here at all (see the
  "Speed transitions" limitation in `planning/problem.py` and
  `docs/assumptions.md`) and `max(V_in, V_out)` is not offered as a bound on it.

The frontier table in section 5 is a slice of a larger surface at 450 kt. The
general threshold for a `dpsi` turn on a grid of a given cell size is

```
V* = sqrt( 0.5 * cell * g * tan(phi) / tan(dpsi / 2) )
```

which reproduces that table (5 NM / 25 deg / 450 kt rejected, 5 NM / 30 deg
accepted) and is measured against the implementation's own gate in
`tests/test_m3_experiments.py`. The Dubins omission in section 7 is unchanged and
if anything firmer: a curvature lower bound would now have to hold across the
whole speed envelope.

## 9. Backward compatibility

With `turn_model: "none"` and a single-speed envelope -- the default for any
aircraft that does not declare one -- the Milestone 1 code path runs untouched:
states are `GridState`, the successor set and its order are identical, and no
turn is evaluated. Parity is therefore structural rather than coincidental, and it is
gated by `tests/test_milestone1_parity.py`, which replays an 80-row experiment
matrix captured from commit `76e9c54` and requires every deterministic column to
reproduce exactly.

Schema version 1 documents, and documents with no `schema_version` at all, are
migrated to `turn_model: "none"`.
