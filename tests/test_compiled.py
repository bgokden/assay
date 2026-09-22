import os

import pytest
import torch

from assay.compiled import CompiledModel, CrossEncoderModel, load_any
from assay.records import Record
from assay.schema import Question
from assay.train_compiled import make_batches

ENCODER = os.environ.get("ASSAY_TEST_ENCODER", "Alibaba-NLP/gte-modernbert-base")
DEVICE = os.environ.get("ASSAY_TEST_DEVICE", "cuda")

pytestmark = pytest.mark.skipif(
    DEVICE == "cuda" and not torch.cuda.is_available(), reason="needs a GPU"
)

STATE = {"message": "My card was charged twice for order A-104. Please refund me today."}
TEAM = Question(
    type="choice",
    instructions="Which team should handle this?",
    options={"billing": "Charges and refunds", "technical": "Bugs and outages", "sales": None},
)
REFUND = Question(type="bool", instructions="Does the customer ask for money back?")
ANGER = Question(
    type="score", instructions="How angry is the customer?", levels=["calm", "annoyed", "furious"]
)


@pytest.fixture(scope="module")
def model():
    m = CompiledModel.from_encoder(ENCODER, device=DEVICE)
    m.eval()
    return m


def test_forward_shapes_and_masking(model):
    logits, evidence = model([STATE, STATE, STATE], [TEAM, REFUND, ANGER])
    assert logits.shape == (3, 3)
    assert evidence.shape == (3,)
    assert torch.isinf(logits[1, 2])  # the bool has two options
    assert torch.isfinite(logits[0]).all() and torch.isfinite(logits[2]).all()


def test_answers_are_distributions(model):
    answers = model.answer(STATE, {"team": TEAM, "refund": REFUND, "anger": ANGER})
    assert list(answers["team"].probabilities) == ["billing", "technical", "sales"]
    for a in answers.values():
        assert abs(sum(a.probabilities.values()) - 1.0) < 1e-4
        assert 0.0 <= a.evidence <= 1.0


def test_options_are_scored_by_content(model):
    """Reordering the options permutes the logits: nothing depends on position."""
    torch.manual_seed(0)
    with torch.no_grad():
        model.bilinear.weight.normal_(std=0.05)
        model.option_bias.weight.normal_(std=0.05)
    reordered = Question(
        type="choice",
        instructions=TEAM.instructions,
        options={k: TEAM.options[k] for k in ["sales", "billing", "technical"]},
    )
    a = model.answer(STATE, {"q": TEAM})["q"].probabilities
    b = model.answer(STATE, {"q": reordered})["q"].probabilities
    for key in TEAM.keys:
        assert a[key] == pytest.approx(b[key], abs=1e-4)


def test_zero_shot_prefers_matching_option(model):
    logits = model.zero_shot_logits([STATE], [TEAM])
    assert logits.argmax().item() == 0  # billing


def test_save_and_load_roundtrip(model, tmp_path):
    model.temperature = 1.7
    model.save_pretrained(str(tmp_path))
    loaded = load_any(str(tmp_path), device=DEVICE)
    assert isinstance(loaded, CompiledModel) and not isinstance(loaded, CrossEncoderModel)
    assert loaded.temperature == pytest.approx(1.7)
    before = model.answer(STATE, {"team": TEAM})["team"].probabilities
    after = loaded.answer(STATE, {"team": TEAM})["team"].probabilities
    for key in TEAM.keys:
        assert after[key] == pytest.approx(before[key], abs=1e-4)


def test_cross_encoder_forward_and_roundtrip(tmp_path):
    m = CrossEncoderModel.from_encoder(ENCODER, device=DEVICE)
    m.eval()
    logits, evidence = m([STATE, STATE], [TEAM, REFUND])
    assert logits.shape == (2, 3) and evidence.shape == (2,)
    assert torch.isinf(logits[1, 2])
    m.save_pretrained(str(tmp_path))
    assert isinstance(load_any(str(tmp_path), device=DEVICE), CrossEncoderModel)


def test_make_batches_respects_budget():
    records = [
        Record(state="x", questions=[type("LQ", (), {"question": q})()], meta={})
        for q in [TEAM, REFUND, ANGER, TEAM, TEAM]
    ]
    batches = make_batches(records, list(range(5)), batch_size=4, pair_budget=5)
    assert batches == [[0, 1], [2], [3], [4]]
    assert make_batches(records, list(range(5)), batch_size=2, pair_budget=100) == [
        [0, 1],
        [2, 3],
        [4],
    ]


def test_conditioned_forward_and_roundtrip(tmp_path):
    from assay.compiled import ConditionedModel

    m = ConditionedModel.from_encoder(ENCODER, device=DEVICE)
    m.eval()
    logits, evidence = m([STATE, STATE], [TEAM, REFUND])
    assert logits.shape == (2, 3) and evidence.shape == (2,)
    assert torch.isinf(logits[1, 2])
    answers = m.answer(STATE, {"team": TEAM, "anger": ANGER})
    assert abs(sum(answers["team"].probabilities.values()) - 1.0) < 1e-4
    m.save_pretrained(str(tmp_path))
    assert isinstance(load_any(str(tmp_path), device=DEVICE), ConditionedModel)


def test_late_interaction_rewards_option_text_present_in_state(tmp_path):
    m = CompiledModel.from_encoder(ENCODER, device=DEVICE, late_interaction=True)
    m.eval()
    with torch.no_grad():
        hidden, mask = m.encode_states([STATE])
        options = m.compile_options([TEAM])
        scores = m.late_interaction(hidden, mask, options)
    assert scores.shape == (1, 3)
    assert (
        scores[0, 0] > scores[0, 2]
    )  # "charges and refunds" appears in the state, "sales" does not
    logits, _ = m([STATE], [TEAM])
    assert logits.shape == (1, 3)
    m.save_pretrained(str(tmp_path))
    loaded = load_any(str(tmp_path), device=DEVICE)
    assert loaded.late_scale is not None
    assert loaded.late_scale.item() == pytest.approx(m.late_scale.item())


def test_joint_reader_forward_and_roundtrip(tmp_path):
    from assay.compiled import JointReaderModel

    m = JointReaderModel.from_encoder(ENCODER, device=DEVICE, reader_layers=2)
    m.eval()
    logits, evidence = m([STATE, STATE], [TEAM, REFUND])
    assert logits.shape == (2, 3) and evidence.shape == (2,)
    assert torch.isinf(logits[1, 2]) and torch.isfinite(logits[0]).all()
    answers = m.answer(STATE, {"team": TEAM, "anger": ANGER})
    assert abs(sum(answers["team"].probabilities.values()) - 1.0) < 1e-4
    m.save_pretrained(str(tmp_path))
    loaded = load_any(str(tmp_path), device=DEVICE)
    assert isinstance(loaded, JointReaderModel) and len(loaded.reader_layers) == 2
    a = m.answer(STATE, {"team": TEAM})["team"].probabilities
    b = loaded.answer(STATE, {"team": TEAM})["team"].probabilities
    for key in TEAM.keys:
        assert b[key] == pytest.approx(a[key], abs=1e-4)


def test_the_same_server_serves_the_encoder_tier(model):
    """Every tier goes through the same endpoints: the runner, not the endpoint, knows how a
    tier turns a request into a forward pass."""
    from fastapi.testclient import TestClient

    from assay.server import create_app

    client = TestClient(create_app(model, "encoder-tier"))
    body = {
        "state": STATE,
        "questions": {
            "team": {
                "type": "choice",
                "instructions": TEAM.instructions,
                "options": {"billing": "Charges and refunds", "technical": "Bugs and outages"},
            },
            "refund": {"type": "bool", "instructions": REFUND.instructions},
        },
    }
    r = client.post("/v1/decide", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["usage"]["input_tokens"] > 0
    assert out["answers"]["team"]["choice"] in ("billing", "technical")
    assert abs(sum(out["answers"]["team"]["probabilities"].values()) - 1.0) < 1e-3
    assert 0.0 <= out["answers"]["refund"]["p_true"] <= 1.0

    one = client.post(
        "/v1/systemone",
        json={
            "state": "x",
            "questions": {"short": {"type": "noul", "instructions": "Is this text short?"}},
        },
    )
    assert one.status_code == 200, one.text
    assert 0.0 <= one.json()["nouls"]["short"]["noul"] <= 1.0


def test_a_graph_over_the_encoder_tier_runs_in_one_pass(model):
    from fastapi.testclient import TestClient

    from assay.server import create_app

    client = TestClient(create_app(model, "encoder-tier"))
    graph = {
        "start": "triage",
        "nodes": {
            "triage": {
                "question": {
                    "type": "choice",
                    "instructions": "Which team should handle this?",
                    "options": {"billing": "Charges and refunds", "technical": "Bugs"},
                },
                "edges": {"billing": "refund", "technical": "outage"},
            },
            "refund": {
                "question": {"type": "bool", "instructions": REFUND.instructions},
                "edges": {"yes": "pay", "no": "reply"},
            },
            "outage": {
                "question": {"type": "bool", "instructions": "Is a service outage described?"},
                "edges": {"yes": "page", "no": "reply"},
            },
            "pay": {"outcome": "refund"},
            "reply": {"outcome": "reply"},
            "page": {"outcome": "page_oncall"},
        },
    }
    r = client.post("/v1/decide_graph", json={"state": STATE, "graph": graph})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["outcome"] in {"refund", "reply", "page_oncall"}
    assert out["usage"] == {
        "input_tokens": out["usage"]["input_tokens"],
        "questions": 3,
        "forward_passes": 1,
    }
