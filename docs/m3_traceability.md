# Milestone 3 traceability

Requirement → implementation → verification → evidence, for the register in
[`requirements_m3.md`](requirements_m3.md).

The point of this table is that a reviewer should never have to take a claim on
trust: every row names the code that implements it and the test that fails if it
stops being true. Where a row's verification column says *analysis*, the
derivation is in the linked document **and** the numeric consequence of that
derivation is pinned by the test named alongside it.

This is an engineering aid, not a certification artefact. See the scope note at
the top of [`requirements_m3.md`](requirements_m3.md).

---

## Matrix

| Requirement | Implementation | Verification | Evidence |
| --- | --- | --- | --- |
| `M3-REQ-001` ISA atmosphere | `src/atp/core/atmosphere.py` | `tests/test_atmosphere.py` | [`milestone3_results.md`](milestone3_results.md) §4 |
| `M3-REQ-002` Airspeed conversions | `src/atp/core/speeds.py` | `tests/test_speeds.py` | §4 |
| `M3-REQ-003` Speed envelope | `src/atp/aircraft/envelope.py`; `AircraftPerformance.envelope` | `tests/test_speed_envelope.py` | §5 |
| `M3-REQ-004` Speed as planning state | `FlightState.isp` in `src/atp/planning/state.py`; `TrajectoryPlanningProblem._successors_with_flight_state` | `tests/test_speed_state.py` | §6 |
| `M3-REQ-005` Speed-dependent fuel flow | `AircraftPerformance.speed_fuel_factor`, `.fuel_flow_kg_per_h` | `tests/test_speed_performance.py` | §7 |
| `M3-REQ-006` Speed-dependent turn performance | `AircraftPerformance.turn_radius_nm` / `.turn_rate_rad_per_h`; `CostModel.turn_metrics` | `tests/test_speed_performance.py`, `tests/test_speed_state.py` | §8 |
| `M3-REQ-007` Speed transition model | `TrajectoryPlanningProblem` module docstring; `TrajectoryEvaluation.speed_changes` | `tests/test_speed_state.py`, `tests/test_m3_experiments.py` | §9 |
| `M3-REQ-008` Cost model | `CostModel.evaluate`, `.turn_metrics`, `.price` | `tests/test_m3_experiments.py` | §11, §12 |
| `M3-REQ-009` Heuristic admissibility | `CostModel.speed_cost_lower_bound_per_nm`, `.constant_cost_offset` | `tests/test_admissibility_m3.py` | §10 |
| `M3-REQ-010` M2 fixed-speed parity | `SpeedEnvelope.fixed`; `AircraftPerformance.envelope` default; `turn-limited-fixed-speed` scenario | `tests/test_m2_fixed_speed_parity.py` | §11 |
| `M3-REQ-011` Speed non-degeneracy | `speed-choice{,-min-time,-min-fuel}` scenarios; `configs/experiment_m3_speed_sweep.json`, `configs/experiment_m3_objective.json` | `tests/test_m3_experiments.py` | §12 |
| `M3-REQ-012` Feasibility frontier | `speed-turn-frontier` scenario; `configs/experiment_m3_turn_frontier.json` | `tests/test_m3_experiments.py::test_the_measured_turn_threshold_matches_the_analytic_one` and `::test_speed_changes_turn_feasibility_on_the_frontier_scenario` | §13 |
| `M3-REQ-013` Domain validation | `AtmosphereDomainError`, `SpeedDomainError`, `SpeedEnvelope.tas_is_within_limits` | `tests/test_atmosphere.py`, `tests/test_speeds.py`, `tests/test_speed_envelope.py` | §4, §5 |
| `M3-REQ-014` Determinism | fixed successor order in `_successors_with_flight_state`; heap ordering in `astar.py` | `tests/test_m3_experiments.py::test_repeated_runs_reproduce_states_cost_speeds_and_metrics`, `::test_a_repeated_experiment_matrix_is_identical_apart_from_wall_clock` | §14 |
| `M3-REQ-015` M1 parity | unchanged Milestone 1 path; default singleton envelope | `tests/test_milestone1_parity.py` (golden matrix and its assertions unchanged; only the scenario-registration set was widened for the five M3 opt-in scenarios), `tests/test_m2_fixed_speed_parity.py::test_milestone_1_scenarios_still_plan_over_grid_states` | §11 |
| `M3-REQ-016` Architecture | module placement: `core/atmosphere.py`, `core/speeds.py`, `aircraft/envelope.py` | `tests/test_m3_experiments.py::test_the_generic_search_still_imports_no_aerospace_module` and the two layer tests beside it | §3 |
| `M3-REQ-017` Performance impact | — (measurement, not implementation) | — | §15 |
| `M3-REQ-018` Documentation and provenance | `docs/milestone3_results.md`, this file, `docs/requirements_m3.md`, provenance notes in `aircraft/envelope.py` and `core/atmosphere.py` | `tests/test_m3_experiments.py::test_the_results_document_exists_and_declares_its_tags`, `::test_the_requirements_and_traceability_documents_cover_every_requirement` | this document |

---

## Derivations verified against the implementation

Each of these is an **analysis** that could have been left as prose. Each is
instead pinned by a test that recomputes the quantity from the equations and
compares it with what the code actually does, so a drift between the derivation
and the implementation fails the build rather than quietly invalidating the
documentation.

| Derivation | Where derived | Where checked against the code |
| --- | --- | --- |
| ISA pressure and density laws | `core/atmosphere.py` docstring | `test_atmosphere.py::test_known_altitudes_reproduce_standard_table_values` |
| Subsonic stagnation relation and its inverse | `core/speeds.py` docstring | `test_speeds.py` round-trip tests |
| Fuel-flow speed factor `(1-b)x³ + b/x` | `AircraftPerformance.speed_fuel_factor` | `test_speed_performance.py::test_fuel_flow_follows_the_documented_power_law` |
| Interior minimum of fuel flow at `x = (b/(3(1-b)))^{1/4}` | same | `::test_the_fuel_flow_curve_has_an_interior_minimum_where_the_derivation_says` |
| Best specific range at `x = (b/(1-b))^{1/4}` (reduces to `3^{-1/4}` only at the shipped `b=0.25`) | same | `::test_best_specific_range_sits_at_a_different_speed_than_least_fuel_flow`, `::test_the_specific_range_formula_is_not_just_the_b_equals_quarter_coincidence` |
| `R = V²/(g tan φ)`, `ω R = V` | `aircraft/turn.py` | `test_speed_performance.py::test_turn_radius_grows_with_the_square_of_speed`, `::test_the_product_of_rate_and_radius_is_the_speed_at_every_speed` |
| Turn-fit threshold `V* = √(½ · cell · g · tan φ / tan(Δψ/2))` | `speed_turn_frontier` docstring | `test_m3_experiments.py::test_the_measured_turn_threshold_matches_the_analytic_one` (bisection of the implementation's own gate) |
| Envelope-wide lower bound `A = min_V (c_t + c_f ff_min(V) + c_r ρ_min)/(V + \|w\|_max)` | `CostModel.speed_cost_lower_bound_per_nm` | `test_admissibility_m3.py::test_the_bound_is_the_minimum_over_the_envelope_not_a_fixed_speed_quotient`, `::test_the_bound_never_exceeds_any_realised_cost_per_horizontal_nm` |
| Air heading depends on TAS, hence `isp` is necessary in the state | `planning/state.py` docstring | `test_speed_state.py::test_the_corner_charged_at_a_node_depends_on_the_incoming_speed` (and its still-air control) |

---

## Deliberate gaps

Recorded here rather than left for a reader to discover.

| Gap | Consequence | Why it is acceptable for this milestone |
| --- | --- | --- |
| No acceleration model between speeds | Reported trajectories are optimistic by the time and fuel a real speed change would cost | The abstraction is stated in `planning/problem.py`, and `speed_changes` counts how often it is exercised so the exposure is measurable per trajectory rather than unknown |
| `isp` is not load-bearing with `turn_model="none"` | The state space is multiplied by the speed count for no modelling benefit in that configuration | Measured in §15 rather than argued; the representation is kept uniform, and the configuration of interest is `gate+cost` |
| No compressibility drag rise near the Mach limit | The fastest options are modelled as cheaper than they would really be | The envelope's Mach limit still removes them where they are illegal; the omission is stated in `speed_fuel_factor` |
| Speed does not vary *within* a segment | Segment-level granularity only | Consistent with the existing one-wind-sample-per-edge granularity |
| No mass depletion, so no speed/mass coupling | No cost-index-style optimum speed drift along the route | Explicitly out of scope for Milestone 3 |
