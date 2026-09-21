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
