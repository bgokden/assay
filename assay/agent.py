"""Agents: a decision graph, the actions its outcomes stand for, and a bounded loop.

An agent is a named decision graph plus a mapping from its outcomes to actions. Deciding is
one forward pass -- every question in the graph is answered at once -- and the rest is
routing, so an agent costs what a single question costs no matter how many branches it has.

The library decides; the caller acts. `Agent.route` is pure: given the answers it returns the
outcome and the action that outcome stands for, with its arguments. `Agent.run` is the
convenience for a script: it answers, routes, calls your handler for the action, and if that
handler returns a new state it decides again, up to `max_steps`. A server has no business
running your side effects, so it returns the planned action instead and the client performs it.

    {
      "name": "support-triage",
      "description": "Route a support ticket and say what to do with it",
      "max_steps": 2,
      "graph": {"start": "triage", "nodes": {...}},
      "actions": {
        "issue_refund": {"action": "refund", "arguments": {"queue": "billing"}},
        "human_review": {"action": "escalate"}
      }
    }

Outcomes without an entry in `actions` are returned as themselves, which is what a read-only
classifier wants. A node that guards on probability, evidence or the conformal act threshold
sends the case to its fallback, so "the model is not sure" is a route like any other.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Callable
from typing import Any

from assay.graph import Graph, walk
from assay.schema import Answer, Question

MAX_STEPS = 8
MAX_AGENTS = 256  # a server holds a registry, not a database


@dataclasses.dataclass(frozen=True)
class Action:
    """What an outcome stands for: a name the caller knows how to perform, and its arguments."""

    name: str
    arguments: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.name, "arguments": self.arguments}


@dataclasses.dataclass
class Decision:
    """One pass of the graph: where it ended and what that means."""

    outcome: Any
    action: Action | None
    path: list[dict[str, Any]]
    path_probability: float
    complete: bool
    stopped: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "outcome": self.outcome,
            "path": self.path,
            "path_probability": self.path_probability,
            "complete": self.complete,
        }
        if self.action is not None:
            out.update(self.action.to_dict())
        if self.stopped is not None:
            out["stopped"] = self.stopped
        return out


@dataclasses.dataclass
class AgentRun:
    """What a full `run` did: every pass it made and the result of the last handler."""

    agent: str
    decisions: list[Decision]
    answers: dict[str, dict[str, Any]]
    result: Any = None
    steps: int = 0

    @property
    def last(self) -> Decision:
        return self.decisions[-1]

    @property
    def outcome(self) -> Any:
        return self.last.outcome

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "steps": self.steps,
            "result": self.result,
            "answers": self.answers,
            **self.last.to_dict(),
        }


@dataclasses.dataclass
class Agent:
    name: str
    graph: Graph
    actions: dict[str, Action] = dataclasses.field(default_factory=dict)
    description: str = ""
    max_steps: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Agent:
        if not data.get("name"):
            raise ValueError("an agent needs a name")
        if "graph" not in data:
            raise ValueError(f"agent {data['name']!r} has no graph")
        graph = Graph.from_dict(data["graph"])
        actions = {}
        for outcome, raw in (data.get("actions") or {}).items():
            if isinstance(raw, str):
                raw = {"action": raw}
            if not raw.get("action"):
                raise ValueError(f"outcome {outcome!r} has no action name")
            actions[outcome] = Action(raw["action"], dict(raw.get("arguments") or {}))
        steps = int(data.get("max_steps", 1))
        if not 1 <= steps <= MAX_STEPS:
            raise ValueError(f"max_steps must be between 1 and {MAX_STEPS}, not {steps}")
        agent = cls(
            name=data["name"],
            graph=graph,
            actions=actions,
            description=data.get("description", ""),
            max_steps=steps,
        )
        agent.validate()
        return agent

    @classmethod
    def from_file(cls, path: str) -> Agent:
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def validate(self) -> None:
        """Every action must belong to an outcome the graph can actually reach, or the spec is
        describing an agent that is not the one the graph implements."""
        outcomes = {n.outcome for n in self.graph.nodes.values() if n.terminal}
        unknown = set(self.actions) - outcomes
        if unknown:
            raise ValueError(
                f"agent {self.name!r} has actions for outcomes the graph never reaches: "
                + ", ".join(sorted(str(u) for u in unknown))
            )

    def questions(self) -> dict[str, Question]:
        return self.graph.questions()

    def route(self, answers: dict[str, Answer], conformal: dict[str, Any] | None = None):
        """Pure: the answers in, the outcome and its action out."""
        result = walk(self.graph, answers, conformal)
        outcome = result["outcome"]
        return Decision(
            outcome=outcome,
            action=self.actions.get(outcome) if outcome is not None else None,
            path=result["path"],
            path_probability=result["path_probability"],
            complete=result["complete"],
            stopped=result.get("stopped"),
        )

    def decide(self, model, state: Any, conformal: dict[str, Any] | None = None):
        """One pass: answer every question of the graph and route on those answers."""
        questions = self.questions()
        answers = model.answer(state, questions)
        serialised = {n: a.to_dict(questions[n]) for n, a in answers.items()}
        return self.route(answers, conformal), serialised

    def run(
        self,
        model,
        state: Any,
        handlers: dict[str, Callable[..., Any]] | None = None,
        conformal: dict[str, Any] | None = None,
    ) -> AgentRun:
        """Decide, act, and decide again while a handler hands back a new state.

        A handler is called as `handler(state, **arguments)`. Returning a mapping with a
        `state` key continues the loop with that state; anything else ends the run and becomes
        the result.
        """
        handlers = handlers or {}
        run = AgentRun(agent=self.name, decisions=[], answers={})
        for _ in range(self.max_steps):
            decision, answers = self.decide(model, state, conformal)
            run.decisions.append(decision)
            run.answers = answers
            run.steps += 1
            if decision.action is None or decision.action.name not in handlers:
                return run
            result = handlers[decision.action.name](state, **decision.action.arguments)
            run.result = result
            if isinstance(result, dict) and "state" in result:
                state = result["state"]
                continue
            return run
        return run

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "max_steps": self.max_steps,
            "questions": len(self.graph.questions()),
            "nodes": len(self.graph.nodes),
            "actions": {outcome: a.to_dict() for outcome, a in self.actions.items()},
        }


def load_agents(path: str) -> dict[str, Agent]:
    """Every `*.json` agent in a directory, keyed by name. A single file also works."""
    if os.path.isfile(path):
        agent = Agent.from_file(path)
        return {agent.name: agent}
    agents: dict[str, Agent] = {}
    for filename in sorted(os.listdir(path)):
        if not filename.endswith(".json"):
            continue
        agent = Agent.from_file(os.path.join(path, filename))
        if agent.name in agents:
            raise ValueError(f"two agents are called {agent.name!r} in {path}")
        agents[agent.name] = agent
    if not agents:
        raise ValueError(f"no agent specifications in {path}")
    return agents
