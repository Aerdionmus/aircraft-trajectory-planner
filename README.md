# Aircraft Trajectory Planner

## Overview

Aircraft trajectory planning is a constrained path-planning problem in which a feasible route must be determined between an origin and a destination while accounting for environmental conditions, operational constraints, flight time, and estimated fuel consumption.

This project investigates the application of **A\* heuristic search** to aircraft flight trajectory planning in a simulated airspace environment. The system is designed to evaluate how different heuristic formulations affect trajectory quality, computational efficiency, and adherence to airspace constraints.

The project is developed as an academic Artificial Intelligence case study with a focus on classical heuristic search and its application to aviation systems.

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
