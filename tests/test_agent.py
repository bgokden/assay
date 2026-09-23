import json

import pytest

from assay.agent import Agent, load_agents
from assay.schema import Answer, Question, confidence_from_probabilities

TRIAGE = {
    "name": "support-triage",
    "description": "Route a ticket and say what to do with it",
    "graph": {
        "start": "triage",
        "nodes": {
            "triage": {
                "question": {
                    "type": "choice",
                    "instructions": "Which team should handle this?",
                    "options": {"billing": "Charges", "technical": "Faults"},
                },
                "edges": {"billing": "refund", "technical": "outage"},
                "min_probability": 0.5,
                "fallback": "human",
            },
            "refund": {
                "question": {"type": "bool", "instructions": "Are they asking for money back?"},
                "edges": {"yes": "pay", "no": "reply"},
            },
            "outage": {
                "question": {"type": "bool", "instructions": "Is a service outage described?"},
                "edges": {"yes": "page"},
                "default": "reply",
            },
            "pay": {"outcome": "issue_refund"},
            "page": {"outcome": "page_oncall"},
            "reply": {"outcome": "reply"},
            "human": {"outcome": "human_review"},
        },
    },
    "actions": {
        "issue_refund": {"action": "refund", "arguments": {"queue": "billing"}},
        "page_oncall": "page",
        "human_review": {"action": "escalate"},
    },
}


class StubModel:
    """Answers each question with a distribution the test dictates, by node name."""

    def __init__(self, probabilities: dict[str, dict[str, float]], evidence: float = 1.0) -> None:
        self.probabilities = probabilities
        self.evidence = evidence
        self.calls = 0

    def answer(self, state, questions: dict[str, Question]) -> dict[str, Answer]:
        self.calls += 1
        out = {}
        for name, q in questions.items():
            probs = self.probabilities[name]
            ordered = {k: probs.get(k, 0.0) for k in q.keys}
            out[name] = Answer(
                type=q.type,
                probabilities=ordered,
                confidence=confidence_from_probabilities(list(ordered.values())),
                evidence=self.evidence,
            )
        return out


def test_the_spec_describes_the_agent():
    agent = Agent.from_dict(TRIAGE)
    assert agent.name == "support-triage"
    assert len(agent.questions()) == 3  # one pass answers all of them
    assert agent.actions["issue_refund"].arguments == {"queue": "billing"}
    assert agent.actions["page_oncall"].name == "page"  # the short form
    assert agent.to_dict()["nodes"] == 7


def test_an_action_for_an_unreachable_outcome_is_rejected():
    bad = {**TRIAGE, "actions": {"send_flowers": {"action": "flowers"}}}
    with pytest.raises(ValueError, match="outcomes the graph never reaches"):
        Agent.from_dict(bad)


def test_a_spec_without_a_graph_is_rejected():
    with pytest.raises(ValueError, match="no graph"):
        Agent.from_dict({"name": "empty"})
    with pytest.raises(ValueError, match="needs a name"):
        Agent.from_dict({"graph": TRIAGE["graph"]})
    with pytest.raises(ValueError, match="max_steps"):
        Agent.from_dict({**TRIAGE, "max_steps": 99})


def test_routing_reaches_the_action_for_the_outcome():
    agent = Agent.from_dict(TRIAGE)
    model = StubModel(
        {
            "triage": {"billing": 0.9, "technical": 0.1},
            "refund": {"yes": 0.8, "no": 0.2},
            "outage": {"yes": 0.1, "no": 0.9},
        }
    )
    run = agent.run(model, "a ticket")
    assert run.outcome == "issue_refund"
    assert run.last.action.to_dict() == {"action": "refund", "arguments": {"queue": "billing"}}
    assert [step["node"] for step in run.last.path] == ["triage", "refund"]
    assert run.last.path_probability == pytest.approx(0.72)
    assert model.calls == 1  # the whole tree in one pass
    assert set(run.answers) == {"triage", "refund", "outage"}


def test_an_unsure_model_routes_to_the_fallback():
    """The guard is the point: 'not sure' is a route, not an exception."""
    spec = json.loads(json.dumps(TRIAGE))
    spec["graph"]["nodes"]["triage"]["min_probability"] = 0.8
    agent = Agent.from_dict(spec)
    model = StubModel(
        {
            "triage": {"billing": 0.45, "technical": 0.55},
            "refund": {"yes": 0.5, "no": 0.5},
            "outage": {"yes": 0.5, "no": 0.5},
        }
    )
    run = agent.run(model, "an ambiguous ticket")
    assert run.outcome == "human_review"
    assert run.last.action.name == "escalate"
    assert "probability" in run.last.path[0]["guard"]


def test_a_handler_runs_and_its_result_is_returned():
    agent = Agent.from_dict(TRIAGE)
    model = StubModel(
        {
            "triage": {"billing": 0.9, "technical": 0.1},
            "refund": {"yes": 0.9, "no": 0.1},
            "outage": {"yes": 0.1, "no": 0.9},
        }
    )
    seen = {}

    def refund(state, queue):
        seen["state"] = state
        seen["queue"] = queue
        return {"refunded": True}

    run = agent.run(model, "refund me", handlers={"refund": refund})
    assert seen == {"state": "refund me", "queue": "billing"}
    assert run.result == {"refunded": True}
    assert run.steps == 1


def test_a_handler_that_returns_a_state_decides_again():
    agent = Agent.from_dict({**TRIAGE, "max_steps": 3})
    model = StubModel(
        {
            "triage": {"billing": 0.9, "technical": 0.1},
            "refund": {"yes": 0.9, "no": 0.1},
            "outage": {"yes": 0.1, "no": 0.9},
        }
    )
    states = []

    def refund(state, queue):
        states.append(state)
        if len(states) < 3:
            return {"state": f"{state} +note{len(states)}"}
        return "done"

    run = agent.run(model, "ticket", handlers={"refund": refund})
    assert states == ["ticket", "ticket +note1", "ticket +note1 +note2"]
    assert run.steps == 3 and run.result == "done"
    assert model.calls == 3


def test_the_loop_is_bounded():
    agent = Agent.from_dict({**TRIAGE, "max_steps": 2})
    model = StubModel(
        {
            "triage": {"billing": 0.9, "technical": 0.1},
            "refund": {"yes": 0.9, "no": 0.1},
            "outage": {"yes": 0.1, "no": 0.9},
        }
    )
    run = agent.run(model, "ticket", handlers={"refund": lambda state, queue: {"state": state}})
    assert run.steps == 2 and model.calls == 2


def test_without_a_handler_the_planned_action_is_returned():
    """What a server answers: the decision, for the client to carry out."""
    agent = Agent.from_dict(TRIAGE)
    model = StubModel(
        {
            "triage": {"billing": 0.1, "technical": 0.9},
            "refund": {"yes": 0.5, "no": 0.5},
            "outage": {"yes": 0.95, "no": 0.05},
        }
    )
    run = agent.run(model, "everything is down")
    assert run.to_dict()["action"] == "page"
    assert run.to_dict()["outcome"] == "page_oncall"
    assert run.result is None


def test_agents_load_from_a_directory(tmp_path):
    (tmp_path / "triage.json").write_text(json.dumps(TRIAGE))
    (tmp_path / "notes.txt").write_text("ignored")
    agents = load_agents(str(tmp_path))
    assert list(agents) == ["support-triage"]
    assert load_agents(str(tmp_path / "triage.json"))["support-triage"].max_steps == 1
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no agent specifications"):
        load_agents(str(empty))


def test_every_shipped_example_agent_is_valid():
    """The templates are documentation, so they have to keep loading."""
    agents = load_agents("examples/agents")
    assert set(agents) == {"support-triage", "content-review", "document-intake"}
    for agent in agents.values():
        assert agent.description
        assert agent.questions()
        for outcome, action in agent.actions.items():
            assert action.name, outcome
