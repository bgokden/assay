"""Decision graphs: a tree of typed questions walked by the model's own answers.

A graph names a start node; each question node carries a typed question and edges keyed by the
option the model chooses; outcome nodes end the walk. Because question branches are isolated
over a shared state, every question in the graph is answered in ONE forward pass and the walk
is then pure logic over those answers - a whole decision tree costs what a single question
costs, plus a few microseconds.

Routing can use more than the argmax: a node may require a minimum probability or evidence, or
require the conformal "act" threshold, and send the case elsewhere (typically to a human) when
the model does not clear it. The path probability reported at the end is the product of the
chosen options' probabilities, which is meaningful precisely because the model is calibrated.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from assay.conformal import should_act
from assay.schema import Answer, Question

MAX_NODES = 256


@dataclasses.dataclass
class Node:
    name: str
    question: Question | None = None
    edges: dict[str, str] = dataclasses.field(default_factory=dict)
    default: str | None = None  # taken when the chosen option has no edge
    fallback: str | None = None  # taken when a guard below is not met
    min_probability: float | None = None
    min_evidence: float | None = None
    require_act: bool = False  # use the conformal act threshold for this question's type
    outcome: Any = None  # set on terminal nodes

    @property
    def terminal(self) -> bool:
        return self.question is None


@dataclasses.dataclass
class Graph:
    start: str
    nodes: dict[str, Node]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Graph:
        if "start" not in data or "nodes" not in data:
            raise ValueError("graph needs 'start' and 'nodes'")
        nodes: dict[str, Node] = {}
        for name, raw in data["nodes"].items():
            question = Question.from_dict(raw["question"]) if raw.get("question") else None
            node = Node(
                name=name,
                question=question,
                edges=dict(raw.get("edges") or {}),
                default=raw.get("default"),
                fallback=raw.get("fallback"),
                min_probability=raw.get("min_probability"),
                min_evidence=raw.get("min_evidence"),
                require_act=bool(raw.get("require_act", False)),
                outcome=raw.get("outcome"),
            )
            if node.terminal and node.outcome is None:
                raise ValueError(f"node {name!r} has neither a question nor an outcome")
            nodes[name] = node
        graph = cls(start=data["start"], nodes=nodes)
        graph.validate()
        return graph

    def validate(self) -> None:
        if len(self.nodes) > MAX_NODES:
            raise ValueError(f"graph has {len(self.nodes)} nodes, limit is {MAX_NODES}")
        if self.start not in self.nodes:
            raise ValueError(f"start node {self.start!r} is not defined")
        for node in self.nodes.values():
            targets = [*node.edges.values(), node.default, node.fallback]
            for target in targets:
                if target is not None and target not in self.nodes:
                    raise ValueError(f"node {node.name!r} points at undefined node {target!r}")
            if node.question is not None:
                keys = set(node.question.keys)
                unknown = set(node.edges) - keys
                if unknown:
                    raise ValueError(
                        f"node {node.name!r} has edges for unknown options {sorted(unknown)}"
                    )
        self._check_acyclic()

    def _check_acyclic(self) -> None:
        colour: dict[str, int] = {}

        def visit(name: str, path: list[str]) -> None:
            state = colour.get(name, 0)
            if state == 1:
                raise ValueError("graph has a cycle: " + " -> ".join([*path, name]))
            if state == 2:
                return
            colour[name] = 1
            node = self.nodes[name]
            for target in [*node.edges.values(), node.default, node.fallback]:
                if target is not None:
                    visit(target, [*path, name])
            colour[name] = 2

        visit(self.start, [])

    def questions(self) -> dict[str, Question]:
        """Every question in the graph, keyed by node name: one forward pass answers all."""
        return {name: n.question for name, n in self.nodes.items() if n.question is not None}


def next_node(
    node: Node, answer: Answer, conformal: dict[str, Any] | None
) -> tuple[str | None, str | None]:
    """The next node name and, when a guard stopped the normal route, the reason."""
    probs = list(answer.probabilities.values())  # insertion order is the question's key order
    reason = None
    if node.min_probability is not None and max(probs) < node.min_probability:
        reason = f"probability {max(probs):.2f} below {node.min_probability:.2f}"
    elif node.min_evidence is not None and answer.evidence < node.min_evidence:
        reason = f"evidence {answer.evidence:.2f} below {node.min_evidence:.2f}"
    elif node.require_act:
        if conformal is None:
            reason = "no conformal thresholds are fitted for this model"
        elif not should_act(probs, answer.type, conformal):
            reason = "below the fitted act threshold for this question type"
    if reason is not None:
        return node.fallback, reason
    return node.edges.get(answer.argmax, node.default), None


def walk(
    graph: Graph, answers: dict[str, Answer], conformal: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Follow the graph using already-computed answers. Returns the path, the outcome and the
    product of the chosen options' probabilities."""
    path: list[dict[str, Any]] = []
    name = graph.start
    probability = 1.0
    seen: set[str] = set()
    while True:
        if name in seen:
            raise ValueError(f"graph revisited node {name!r}")
        seen.add(name)
        node = graph.nodes[name]
        if node.terminal:
            return {
                "outcome": node.outcome,
                "node": name,
                "path": path,
                "path_probability": round(probability, 6),
                "complete": True,
            }
        answer = answers[name]
        target, reason = next_node(node, answer, conformal)
        chosen = answer.argmax
        step = {
            "node": name,
            "question": node.question.instructions,
            "chosen": chosen,
            "probability": round(max(answer.probabilities.values()), 4),
            "confidence": round(answer.confidence, 4),
            "evidence": round(answer.evidence, 4),
        }
        if reason is not None:
            step["guard"] = reason
        else:
            probability *= max(answer.probabilities.values())
        path.append(step)
        if target is None:
            return {
                "outcome": None,
                "node": name,
                "path": path,
                "path_probability": round(probability, 6),
                "complete": False,
                "stopped": reason or f"no edge for option {chosen!r}",
            }
        name = target
