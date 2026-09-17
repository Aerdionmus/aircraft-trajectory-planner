"""A* tested with no aerospace model in sight.

These tests are the reason :class:`atp.planning.problem.SearchProblem` exists:
the search can be exercised, including its optimality and re-opening behaviour,
on hand-built graphs where the correct answer is known by inspection.
"""

from __future__ import annotations

import pytest

from atp.planning.astar import SearchStatus, astar, dijkstra, reconstruct_path
from atp.planning.problem import Transition


class GraphProblem:
    def __init__(self, edges: dict[str, list[tuple[str, float]]], start: str, goal: str):
        self.edges = edges
        self.start = start
        self.goal = goal
        self.expansions: list[str] = []

    def initial_state(self) -> str:
        return self.start

    def is_goal(self, state: str) -> bool:
        return state == self.goal

    def successors(self, state: str):
        self.expansions.append(state)
        for target, cost in self.edges.get(state, []):
            yield Transition(target, cost)


#      B --3-- D
#    /  \       \
#   A    2       1
#    \    \     /
#      C --5-- E
# Cheapest A -> E is A-B-D-E = 1 + 3 + 1 = 5.
EDGES = {
    "A": [("B", 1.0), ("C", 2.0)],
    "B": [("D", 3.0), ("C", 2.0)],
    "C": [("E", 5.0)],
    "D": [("E", 1.0)],
    "E": [],
}


def test_astar_finds_optimal_path_with_zero_heuristic():
    problem = GraphProblem(EDGES, "A", "E")
    result = astar(problem, lambda s: 0.0)
    assert result.status is SearchStatus.SOLVED
    assert result.path == ["A", "B", "D", "E"]
    assert result.cost == pytest.approx(5.0)
    assert len(result.transitions) == len(result.path) - 1


def test_dijkstra_matches_astar():
    a = astar(GraphProblem(EDGES, "A", "E"), lambda s: 0.0)
    d = dijkstra(GraphProblem(EDGES, "A", "E"))
    assert a.cost == pytest.approx(d.cost)
    assert a.path == d.path


def test_admissible_heuristic_preserves_optimality_and_cuts_expansions():
    perfect = {"A": 5.0, "B": 4.0, "C": 5.0, "D": 1.0, "E": 0.0}
    blind = GraphProblem(EDGES, "A", "E")
    guided = GraphProblem(EDGES, "A", "E")
    blind_result = astar(blind, lambda s: 0.0)
    guided_result = astar(guided, lambda s: perfect[s])
    assert guided_result.cost == pytest.approx(blind_result.cost)
    assert guided_result.statistics.expansions <= blind_result.statistics.expansions


def test_inadmissible_heuristic_can_return_a_worse_path():
    """Documents the failure mode rather than hiding it."""
    misleading = {"A": 0.0, "B": 100.0, "C": 0.0, "D": 0.0, "E": 0.0}
    result = astar(GraphProblem(EDGES, "A", "E"), lambda s: misleading[s])
    assert result.status is SearchStatus.SOLVED
    assert result.cost >= 5.0 - 1e-9


def test_unsolvable_problem_is_reported_not_crashed():
    edges = {"A": [("B", 1.0)], "B": [], "Z": []}
    result = astar(GraphProblem(edges, "A", "Z"), lambda s: 0.0)
    assert result.status is SearchStatus.UNSOLVABLE
    assert result.path == []
    assert result.cost == float("inf")


def test_expansion_limit_is_enforced():
    chain = {str(i): [(str(i + 1), 1.0)] for i in range(50)}
    chain["50"] = []
    result = astar(GraphProblem(chain, "0", "50"), lambda s: 0.0, max_expansions=5)
    assert result.status is SearchStatus.LIMIT_REACHED
    assert result.statistics.expansions == 5


def test_goal_is_tested_on_expansion_not_on_generation():
    """A cheap-but-late goal edge must win over an expensive early one."""
    edges = {
        "A": [("G", 10.0), ("B", 1.0)],
        "B": [("G", 1.0)],
        "G": [],
    }
    result = astar(GraphProblem(edges, "A", "G"), lambda s: 0.0)
    assert result.cost == pytest.approx(2.0)
    assert result.path == ["A", "B", "G"]


def test_reconstruct_path_pairs_states_with_transitions():
    parents = {
        "B": ("A", Transition("B", 1.0)),
        "C": ("B", Transition("C", 2.0)),
    }
    states, transitions = reconstruct_path(parents, "C")
    assert states == ["A", "B", "C"]
    assert [t.state for t in transitions] == ["B", "C"]


def test_infinite_cost_transitions_are_skipped():
    edges = {"A": [("B", float("inf")), ("C", 1.0)], "C": [("B", 1.0)], "B": []}
    result = astar(GraphProblem(edges, "A", "B"), lambda s: 0.0)
    assert result.path == ["A", "C", "B"]
    assert result.cost == pytest.approx(2.0)
