import os

import pytest
import torch
from fastapi.testclient import TestClient

from assay.model import AssayModel
from assay.server import create_app

BASE = os.environ.get("ASSAY_TEST_BASE", "Qwen/Qwen3-0.6B-Base")
DEVICE = os.environ.get("ASSAY_TEST_DEVICE", "cuda")

pytestmark = pytest.mark.skipif(
    DEVICE == "cuda" and not torch.cuda.is_available(), reason="needs a GPU"
)


@pytest.fixture(scope="module")
def client():
    dtype = torch.float32 if DEVICE == "cpu" else torch.bfloat16
    model = AssayModel.from_base(BASE, lora_r=None, dtype=dtype, device=DEVICE)
    model.eval()
    return TestClient(create_app(model, "test-model"))


def test_decide_returns_all_answer_types(client):
    body = {
        "state": {"message": "My card was charged twice for order A-104. Please refund me."},
        "questions": {
            "refund": {"type": "bool", "instructions": "Does the customer ask for money back?"},
            "team": {
                "type": "choice",
                "instructions": "Which team should handle `message`?",
                "options": {"billing": "Charges and refunds", "technical": "Bugs and outages"},
            },
            "anger": {
                "type": "score",
                "instructions": "How angry is the customer?",
                "levels": ["calm", "annoyed", "furious"],
            },
        },
    }
    r = client.post("/v1/decide", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["model"] == "test-model"
    assert out["usage"]["input_tokens"] > 0
    a = out["answers"]
    assert set(a) == {"refund", "team", "anger"}
    assert 0.0 <= a["refund"]["p_true"] <= 1.0
    assert a["team"]["choice"] in ("billing", "technical")
    assert abs(sum(a["team"]["probabilities"].values()) - 1.0) < 1e-3
    assert 0.0 <= a["anger"]["score"] <= 2.0
    assert a["anger"]["legend"]["2"] == "furious"
    for ans in a.values():
        assert 0.0 <= ans["confidence"] <= 1.0
        assert 0.0 <= ans["evidence"] <= 1.0


def test_decide_rejects_bad_question(client):
    r = client.post(
        "/v1/decide",
        json={
            "state": "x",
            "questions": {"q": {"type": "choice", "instructions": "?", "options": {"only": None}}},
        },
    )
    assert r.status_code == 422


def test_models_endpoint(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["models"][0]["id"] == "test-model"


GRAPH = {
    "start": "triage",
    "nodes": {
        "triage": {
            "question": {
                "type": "choice",
                "instructions": "Which team should handle `message`?",
                "options": {"billing": "Charges and refunds", "technical": "Bugs and outages"},
            },
            "edges": {"billing": "refund", "technical": "outage"},
            "min_probability": 0.4,
            "fallback": "human",
        },
        "refund": {
            "question": {"type": "bool", "instructions": "Does the customer ask for money back?"},
            "edges": {"yes": "pay", "no": "explain"},
        },
        "outage": {
            "question": {"type": "bool", "instructions": "Is a service outage described?"},
            "edges": {"yes": "page"},
            "default": "explain",
        },
        "pay": {"outcome": "refund"},
        "explain": {"outcome": "reply"},
        "page": {"outcome": "page_oncall"},
        "human": {"outcome": "human_review"},
    },
}


def test_decide_graph_walks_to_an_outcome(client):
    body = {
        "state": {"message": "My card was charged twice for order A-104. Please refund me."},
        "graph": GRAPH,
    }
    r = client.post("/v1/decide_graph", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["outcome"] in {"refund", "reply", "page_oncall", "human_review"}
    assert out["usage"]["forward_passes"] == 1
    assert out["usage"]["questions"] == 3  # every question in the graph, one pass
    assert 0.0 <= out["path_probability"] <= 1.0
    assert out["path"] and out["path"][0]["node"] == "triage"
    assert set(out["answers"]) == {"triage", "refund", "outage"}
    names = [step["node"] for step in out["path"]]
    assert len(names) == len(set(names))  # no node visited twice


def test_decide_graph_rejects_a_cycle(client):
    bad = {
        "start": "a",
        "nodes": {
            "a": {
                "question": {"type": "bool", "instructions": "Loop?"},
                "edges": {"yes": "b", "no": "b"},
            },
            "b": {
                "question": {"type": "bool", "instructions": "Back?"},
                "edges": {"yes": "a", "no": "a"},
            },
        },
    }
    r = client.post("/v1/decide_graph", json={"state": "x", "graph": bad})
    assert r.status_code == 422
    assert "cycle" in r.json()["detail"]


def test_decide_graph_rejects_unknown_targets(client):
    bad = {
        "start": "a",
        "nodes": {
            "a": {"question": {"type": "bool", "instructions": "?"}, "edges": {"yes": "nowhere"}}
        },
    }
    r = client.post("/v1/decide_graph", json={"state": "x", "graph": bad})
    assert r.status_code == 422
    assert "nowhere" in r.json()["detail"]


def test_health_and_metrics(client):
    assert client.get("/health").json()["status"] == "ok"
    body = {
        "state": "x",
        "questions": {"q": {"type": "bool", "instructions": "Is this text short?"}},
    }
    client.post("/v1/decide", json=body)
    text = client.get("/metrics").text
    assert "assay_requests_total" in text and "assay_batches_total" in text
    stats = client.get("/v1/stats").json()
    assert stats["requests"] >= 1 and stats["batches"] >= 1
    assert stats["latency_ms"]["p50"] > 0


def test_concurrent_requests_share_forward_passes(client):
    """Requests that arrive together are answered in one batch, so batches < requests."""
    import concurrent.futures

    body = {
        "state": "Please refund my duplicate charge.",
        "questions": {
            "refund": {"type": "bool", "instructions": "Does the customer ask for money back?"}
        },
    }
    before = client.get("/v1/stats").json()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: client.post("/v1/decide", json=body), range(8)))
    assert all(r.status_code == 200 for r in responses)
    after = client.get("/v1/stats").json()
    assert after["requests"] - before["requests"] >= 8
    assert after["batches"] - before["batches"] <= 8


SYSTEM_ONE_BODY = {
    "model": "assay",
    "state": "Customer reports a duplicate charge and asks for a refund.",
    "questions": {
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {"billing": "Payments and refunds", "technical": "Product faults"},
        },
        "refund_requested": {"type": "noul", "instructions": "Did the customer request a refund?"},
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["Not urgent", "Soon", "Immediate"],
        },
    },
}


def test_systemone_interface(client):
    """The payload other open decision models accept: `criteria` options and the noul type."""
    r = client.post("/v1/systemone", json=SYSTEM_ONE_BODY)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["choices"]["route"]["choice"] in ("billing", "technical")
    assert 0.0 <= out["nouls"]["refund_requested"]["noul"] <= 1.0
    assert 0.0 <= out["scores"]["urgency"]["score"] <= 2.0
    assert out["scores"]["urgency"]["legend"]["2"] == "Immediate"
    for group in ("choices", "nouls", "scores"):
        for answer in out[group].values():
            assert abs(sum(answer["probabilities"].values()) - 1.0) < 1e-3
            assert 0.0 <= answer["confidence"] <= 1.0
            assert 0.0 <= answer["evidence"] <= 1.0


def test_systemone_and_decide_agree(client):
    """Both interfaces are the same model and parser, so they must give the same numbers."""
    native = {
        "state": SYSTEM_ONE_BODY["state"],
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which team should handle this request?",
                "options": {"billing": "Payments and refunds", "technical": "Product faults"},
            },
            "refund_requested": {
                "type": "bool",
                "instructions": "Did the customer request a refund?",
            },
        },
    }
    a = client.post("/v1/decide", json=native).json()["answers"]
    b = client.post("/v1/systemone", json=SYSTEM_ONE_BODY).json()
    for key, value in a["route"]["probabilities"].items():
        assert b["choices"]["route"]["probabilities"][key] == pytest.approx(value, abs=1e-4)
    assert b["nouls"]["refund_requested"]["noul"] == pytest.approx(
        a["refund_requested"]["p_true"], abs=1e-4
    )
