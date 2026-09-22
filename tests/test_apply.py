import json

import pytest

from assay.apply import load_questions, read_states


def test_reads_bare_states_and_records(tmp_path):
    path = tmp_path / "states.jsonl"
    path.write_text(
        json.dumps("a plain string state")
        + "\n"
        + json.dumps({"channel": "email", "message": "an object state"})
        + "\n"
        + json.dumps({"state": "from a record", "meta": {"id": "set/0"}, "questions": {}})
        + "\n\n"  # blank lines are skipped
    )
    got = list(read_states(str(path)))
    assert [identifier for _, identifier in got] == [0, 1, "set/0"]
    assert got[1][0]["message"] == "an object state"
    assert got[2][0] == "from a record"


def test_limit_stops_early(tmp_path):
    path = tmp_path / "states.jsonl"
    path.write_text("\n".join(json.dumps(f"state {i}") for i in range(5)))
    assert len(list(read_states(str(path), limit=2))) == 2


def test_questions_are_validated(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps({"q": {"type": "noul", "instructions": "Short?"}}))
    questions = load_questions(str(path))
    assert questions["q"].type == "bool"  # the other tools' name for it

    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    with pytest.raises(ValueError, match="non-empty"):
        load_questions(str(empty))


class RoutingStub:
    """A model for the row runner: every question gets the distribution the test names, by
    the first word of its instructions."""

    temperature = 1.0

    class Tokenizer:
        def __call__(self, text, truncation=False, max_length=None):
            return {"input_ids": list(range(len(str(text).split())))}

    def __init__(self, by_question: dict[str, list[float]]) -> None:
        self.tokenizer = self.Tokenizer()
        self.by_question = by_question
        self.calls = 0

    def __call__(self, states, questions):
        import torch

        self.calls += 1
        width = max(len(q.keys) for q in questions)
        logits = torch.full((len(questions), width), -20.0)
        for i, q in enumerate(questions):
            probs = self.by_question[q.instructions.split()[0]]
            for k, p in enumerate(probs[: len(q.keys)]):
                logits[i, k] = torch.tensor(p).clamp(min=1e-6).log()
        return logits, torch.zeros(len(questions))


AGENT = {
    "name": "router",
    "graph": {
        "start": "team",
        "nodes": {
            "team": {
                "question": {
                    "type": "choice",
                    "instructions": "Which team should handle this?",
                    "options": {"billing": "Money", "technical": "Faults"},
                },
                "edges": {"billing": "pay", "technical": "page"},
                "min_probability": 0.8,
                "fallback": "human",
            },
            "pay": {"outcome": "issue_refund"},
            "page": {"outcome": "page_oncall"},
            "human": {"outcome": "human_review"},
        },
    },
    "actions": {"issue_refund": {"action": "refund"}},
}


def write_states(tmp_path, rows):
    path = tmp_path / "states.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return str(path)


def test_routing_a_file_scores_against_expected_outcomes(tmp_path):
    from assay.agent import Agent
    from assay.apply import read_records_for_apply, route_states, summarise_routing

    agent = Agent.from_dict(AGENT)
    states = write_states(
        tmp_path,
        [
            {"state": "charged twice", "expected": "issue_refund", "meta": {"id": "a"}},
            {"state": "charged twice", "expected": "page_oncall", "meta": {"id": "b"}},
        ],
    )
    model = RoutingStub({"Which": [0.95, 0.05]})
    routed = list(route_states(model, agent, read_records_for_apply(states), batch_size=2))
    assert [r["id"] for r in routed] == ["a", "b"]
    assert routed[0]["outcome"] == "issue_refund" and routed[0]["correct"] is True
    assert routed[1]["correct"] is False
    assert routed[0]["action"] == "refund"
    assert model.calls == 1  # both states in one pass

    summary = summarise_routing(routed)
    assert summary == {
        "cases": 2,
        "accuracy": 0.5,
        "routed": 2,
        "routed_accuracy": 0.5,
        "handed_over": 0,
        "handed_over_accuracy": None,
        "hand_over_rate": 0.0,
    }


def test_guarded_cases_are_counted_separately(tmp_path):
    """The number that matters when tuning a threshold: how often the agent handed over, and
    whether it was right to."""
    from assay.agent import Agent
    from assay.apply import read_records_for_apply, route_states, summarise_routing

    agent = Agent.from_dict(AGENT)
    states = write_states(tmp_path, [{"state": "unclear", "expected": "human_review"}])
    model = RoutingStub({"Which": [0.55, 0.45]})  # below the node's 0.8 guard
    routed = list(route_states(model, agent, read_records_for_apply(states)))
    assert routed[0]["outcome"] == "human_review" and routed[0]["correct"] is True
    summary = summarise_routing(routed)
    assert summary["handed_over"] == 1 and summary["routed"] == 0
    assert summary["hand_over_rate"] == 1.0 and summary["handed_over_accuracy"] == 1.0


def test_unlabelled_routing_reports_only_the_count(tmp_path):
    from assay.apply import summarise_routing

    assert summarise_routing([{"outcome": "x", "path": []}]) == {"routed": 1}
