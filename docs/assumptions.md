# Modelling assumptions

Everything below is a simplification. The list is kept in one place so that no
result produced by this package can be read as more than it is.

**This is a teaching and research simulator. It is not flight-planning
software, is not validated against any operational system, and carries no
airworthiness, certification or operational-approval claim. Do not use it to
plan a flight.**

## Data provenance

Every number in this repository is synthetic.

- Aircraft parameters (`MEDIUM_TWIN_JET`, `REGIONAL_TURBOPROP`) are
  *representative* figures chosen to be in the right order of magnitude. They
  are not manufacturer data, not from a certified performance manual, and not
  from BADA.
- Wind fields are analytic constructions (uniform, layered, Rankine vortex).
  They are not a forecast product and are not derived from GRIB or any NWP
  output.
- Risk fields are Gaussian bumps and proximity decays with no calibration
  against any convective-weather, turbulence, icing or conflict-probability
  product. "Risk units" are dimensionless and have no external meaning.
- Restricted regions are invented geometry, not AIP or NOTAM data.
- Cost weights are unit prices in abstract cost units, not currency.

## Geometry and earth model

- **Flat-earth local projection.** `x` is east, `y` is north, both in NM from
  the south-west corner. No spherical or geodetic model, no great-circle
  distance, no convergence of meridians. Errors grow with airspace size; the
  shipped scenarios span a few hundred NM, where the approximation is tolerable
  for relative comparisons and wrong for absolute navigation.
- **No magnetic variation.** All bearings are true.
- **2.5D, not 3D.** Continuous-ish laterally, discrete vertically over an
  explicitly enumerated list of flight levels.
- **Grid discretisation.** With 8-connectivity a path can only take headings in
  45-degree increments, which overestimates length by up to about 8% versus a
  straight line.

## Aircraft

- **Point mass at constant mass.** Fuel burn does not reduce weight, so there is
  no weight/fuel-flow feedback and no mass-driven step-climb optimisation.
- **Single commanded TAS per level**, independent of mass and temperature. No
  Mach or CAS schedule, no ISA deviation, no buffet boundary, no
  thrust-limited ceiling (only a hard `service_ceiling_ft`).
- **Fuel flow varies linearly with altitude** about a reference level. Real
  specific fuel consumption is not linear in altitude; the linear term encodes
  only the qualitative "higher is more efficient" trend.
- **Climb and descent** are charged a fixed fuel delta per 1000 ft on top of the
  cruise burn for the time spent, and limited by a constant maximum vertical
  rate. There is no thrust/drag integration and no climb-speed schedule.
- **No turn dynamics.** Heading is not part of the state, so turn radius, bank
  angle limits and the path lengthening they cause are not modelled. A planned
  trajectory may contain instantaneous 45-degree heading changes.
- **Descent fuel credits are clamped at zero** per segment, so a descent can
  never produce a negative edge cost. Without the clamp A* would be invalid.

## Environment

- **No vertical wind.** Updraughts and downdraughts are not modelled.
- **Wind is sampled once at the mid-point of each edge**, giving `O(h^2)`
  accuracy in cell size for smooth fields. Accuracy degrades across a sharp
  gradient such as a jet-stream edge.
- **Wind is steady.** No time dimension anywhere in the model: the aircraft
  experiences the field as it is, not as it will be on arrival. This also means
  there is no time-of-day, no forecast validity window and no en-route update.
- **Risk is an exposure rate** in risk-units per hour, integrated over segment
  time by the midpoint rule. Using a rate rather than a per-NM density means
  slowing down inside a hazard is correctly penalised.
- **Restricted regions have a simple `[lower_ft, upper_ft]` band.** A transition
  that changes level is tested against the band spanned by its endpoints, which
  is conservative: it can flag a climb that only clips a corner of the band.
- **Cell-centre blocking can miss a region smaller than one cell.** The
  transition-level intersection test is the authoritative one. That test is
  exact horizontally for circles, polygons and corridors, and conservative
  vertically: a level-changing transition is tested against the whole altitude
  band its endpoints span, so it can be blocked by a region it would only clip.
- **Soft-restriction overlap is sampled**, not exact, so a soft penalty carries
  an O(1/samples) error. Hard constraints never use sampling.

## Search

- **Static, single-aircraft, offline.** No replanning, no other traffic, no
  conflict detection or resolution, no ATC interaction, no clearances.
- **No fuel reserve, alternate, diversion or ETOPS logic**, no payload/range
  constraint, no takeoff or landing phase. The problem starts and ends in
  cruise.
- **Pure vertical transitions** (when enabled) ignore the forward distance the
  aircraft covers during the manoeuvre, making them slightly optimistic.
- **Optimality is optimality with respect to this cost model.** A path that is
  optimal here is optimal for a synthetic objective over a synthetic
  environment, and nothing more.

## Costs of infeasible trajectories

A segment violating a hard constraint has no defined cost, so an infeasible
trajectory's distance, time, fuel and cost totals cover only its feasible
segments. Those totals are **not** comparable with a feasible trajectory's;
`comparable_cost` is infinite in that case and is what planner comparisons use.

## What "admissible" means here

The heuristic is admissible **under the assumptions above** — specifically that
`WindField.max_magnitude_kt()` is a sound upper bound over the airspace and
`RiskField.min_density()` a sound lower bound. If a custom field implements
either incorrectly, A* silently loses its optimality guarantee. Any new field
should be added to the bound-sampling test in `tests/test_wind.py`.

Admissibility also depends on the aircraft model: the fuel lower bound assumes a
climb costs at least as much fuel as the matching descent refunds. A performance
set that violates this has its fuel term dropped from the bound rather than
silently trusted. The derivation is in `docs/architecture.md`.

Note also what "optimal" does *not* mean here. A* returns a least-cost path in
the discretised state space under this cost model. It is not the optimal
continuous trajectory, not optimal under any other objective, and not a claim
about any real flight.
