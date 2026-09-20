# Milestone 3 requirements

Scope of this document
----------------------

This is a *lightweight* requirements register for a research engineering
project. It exists so that every capability Milestone 3 claims can be traced to
the code that implements it and the test that verifies it, and so that a
reviewer can tell at a glance which claims are asserted by machine and which are
only argued in prose.

It is **not** a certification artefact. It is not written to DO-178C,
ARP4754A, DO-254 or any other standard, it carries no airworthiness or
operational-approval claim, and no process claim should be read into its
existence. The package remains a synthetic research and teaching simulator; see
[`assumptions.md`](assumptions.md).

Verification method is one of:

- **T** — automated test.
- **E** — experiment recorded in [`milestone3_results.md`](milestone3_results.md).
- **A** — analysis, i.e. a derivation in the documentation. Every **A** here is
  paired with a **T** that checks the derived quantity against the
  implementation; no requirement is verified by analysis alone.

The requirement-to-evidence mapping is in
[`m3_traceability.md`](m3_traceability.md).

---

## Atmosphere and airspeeds

### `M3-REQ-001` — ISA atmosphere

The package shall provide a deterministic ISA model returning temperature,
pressure, density and speed of sound over the altitude range the scenarios use,
with no dependency on external or real-weather data.

*Method:* T, A.

### `M3-REQ-002` — Airspeed conversions

The package shall provide TAS↔Mach and CAS↔TAS conversions on that atmosphere,
each with a documented domain and an exact inverse. A conversion outside its
valid domain shall raise rather than return a value from an inapplicable
relation.

*Method:* T, A.

### `M3-REQ-013` — Atmosphere and speed domain validation

Sea-level reference values, monotonic density and pressure behaviour with
altitude, speed-of-sound behaviour, conversion round trips, and the rejection of
invalid altitude and speed domains shall all be verified automatically.

*Method:* T.

---

## Aircraft model

### `M3-REQ-003` — Speed envelope

The aircraft model shall no longer assume cruise TAS is the only possible
speed. It shall carry a speed envelope providing a discrete set of selectable
planning speeds and operating limits that vary correctly with altitude. All
values shall be explicitly identified as synthetic.

*Method:* T, A.

### `M3-REQ-005` — Speed-dependent fuel flow

Fuel flow shall depend on both altitude and selected speed, shall reduce exactly
to the Milestone 2 value at the reference speed, and its functional form shall
be documented and physically motivated rather than fitted to produce a desired
route.

*Method:* T, A.

### `M3-REQ-006` — Speed-dependent turn performance

Turn radius and turn rate shall be evaluated at the selected TAS through the
existing coordinated-turn relations. The Milestone 2 turn feasibility model
shall not be weakened or bypassed to accommodate speed.

*Method:* T, A.

---

## Planning

### `M3-REQ-004` — Speed as planning state

Speed shall participate in the transition graph as a state dimension, not be
computed after planning. The necessity of the state dimension shall be
justified, not assumed.

*Method:* T, A.

### `M3-REQ-007` — Speed transition model

The model of how speed changes shall be chosen explicitly, documented, and its
optimism quantified or made measurable. No physically impossible behaviour shall
be permitted silently.

*Method:* T, A.

### `M3-REQ-008` — Cost model

The objective shall expose a real trade-off between distance, time, fuel, risk
and turn cost, with the speed decision affecting at least segment time, fuel
consumption and turn performance. No dimensionless speed penalty shall be
introduced.

*Method:* T, E.

### `M3-REQ-009` — Heuristic admissibility

Admissibility shall be **re-derived** over the speed-expanded graph rather than
inherited from Milestone 2, whose monotone-refinement property does not survive
the speed decision. Any heuristic declaring itself admissible shall be shown not
to overestimate; if consistency is declared it shall be checked edge by edge. No
heuristic shall be labelled admissible without a derivation.

*Method:* T, A.

### `M3-REQ-016` — Architecture

The dependency direction `core → environment/aircraft → planning → evaluation`
shall be preserved. The generic A\* implementation shall import no
aerospace-specific module, including the new atmosphere and envelope modules.

*Method:* T.

---

## Regression and evidence

### `M3-REQ-010` — Milestone 2 fixed-speed parity

A speed envelope containing exactly the Milestone 2 cruise speed shall reproduce
Milestone 2 behaviour — path, cost, metrics and expansion counts — within a
stated tolerance. The tolerance claimed is **exact equality**.

*Method:* T, E.

### `M3-REQ-015` — Milestone 1 parity

The Milestone 1 golden results shall continue to reproduce exactly. No Milestone
1 golden data shall be modified.

*Method:* T.

### `M3-REQ-011` — Speed non-degeneracy

At least one controlled experiment shall demonstrate that speed selection
changes measurable quantities: time, fuel, turn characteristics and total cost.

*Method:* T, E.

### `M3-REQ-012` — Speed/turn feasibility frontier

At least one controlled experiment shall demonstrate that changing speed changes
turn feasibility or turn geometry. The threshold shall be derived analytically
**and** measured from the implementation, with neither asserted from the other.

*Method:* T, A, E.

### `M3-REQ-014` — Determinism

Repeated identical runs shall reproduce the state sequence, the cost, the speed
sequence and the evaluation metrics.

*Method:* T.

### `M3-REQ-017` — Performance impact

The computational cost of the added state dimension shall be measured and
reported against the fixed-speed configuration.

*Method:* E.

### `M3-REQ-018` — Documentation and provenance

A permanent results document shall record objective, design, assumptions,
derivations, evidence, measured results, interpretation and limitations, with
measured facts tagged separately from interpretation. Every aircraft and
atmosphere number shall be identified as synthetic or as a defining constant of
the standard, and no operational or certification claim shall be made.

*Method:* T, A.
