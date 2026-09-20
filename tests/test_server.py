import os

import pytest
import torch
from fastapi.testclient import TestClient

from sezgi.model import SezgiModel
from sezgi.server import create_app

BASE = os.environ.get("SEZGI_TEST_BASE", "Qwen/Qwen3-0.6B-Base")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(scope="module")
def client():
    model = SezgiModel.from_base(BASE, lora_r=None, dtype=torch.bfloat16)
    model.eval()
    return TestClient(create_app(model, "test-model"))


def test_decide_returns_all_answer_types(client):
    body = {
        "state": {"message": "My card was charged twice for order A-104. Please refund me."},
        "questions": {
            "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"},
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
    assert 0.0 <= a["refund"]["noul"] <= 1.0
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
        json={"state": "x", "questions": {"q": {"type": "choice", "instructions": "?", "options": {"only": None}}}},
    )
    assert r.status_code == 422


def test_models_endpoint(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["models"][0]["id"] == "test-model"
