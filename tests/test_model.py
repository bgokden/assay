import os

import pytest
import torch

from sezgi.encoding import encode, identity_order
from sezgi.model import SezgiModel
from sezgi.schema import Question

BASE = os.environ.get("SEZGI_TEST_BASE", "Qwen/Qwen3-0.6B-Base")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(scope="module")
def model():
    return SezgiModel.from_base(BASE, lora_r=None, dtype=torch.float32)


STATE = (
    "Hi, I've been trying to connect my Stripe account for 3 days and the integration keeps "
    "failing. I'm losing sales. Please help ASAP."
)
QUESTIONS = {
    "department": Question(
        type="choice",
        instructions="Which team should handle this?",
        options={
            "billing": "Payment or subscription issues",
            "technical": "Bugs or integration problems",
            "sales": "Pricing or account questions",
        },
    ),
    "frustration": Question(
        type="score",
        instructions="How frustrated does the customer appear?",
        levels=["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"],
    ),
    "is_urgent": Question(type="noul", instructions="Does the message convey urgency?"),
}


def test_packed_equals_separate(model):
    packed_all = model.answer(STATE, QUESTIONS)
    for name, q in QUESTIONS.items():
        single = model.answer(STATE, {name: q})[name]
        for key in q.keys:
            assert packed_all[name].probabilities[key] == pytest.approx(
                single.probabilities[key], abs=2e-4
            )


def test_zero_shot_is_sensible(model):
    answers = model.answer(STATE, QUESTIONS)
    assert answers["department"].argmax == "technical"
    assert answers["is_urgent"].noul > 0.5
    assert 0.0 <= answers["frustration"].score <= 2.0
    for a in answers.values():
        assert abs(sum(a.probabilities.values()) - 1.0) < 1e-5
        assert 0.0 <= a.confidence <= 1.0


def test_option_order_does_not_change_keys(model):
    q = QUESTIONS["department"]
    packed_a = encode(model.tokenizer, model.alphabet, STATE, [q], orders=[identity_order(q)])
    packed_b = encode(model.tokenizer, model.alphabet, STATE, [q], orders=[[2, 0, 1]])
    a = model.answer_packed([packed_a], [[q]])[0][0]
    b = model.answer_packed([packed_b], [[q]])[0][0]
    assert list(a.probabilities) == list(b.probabilities) == ["billing", "technical", "sales"]
    assert a.argmax == b.argmax == "technical"


def test_save_and_load_roundtrip(model, tmp_path):
    model.save_pretrained(str(tmp_path))
    assert (tmp_path / "sezgi_head.safetensors").exists()
    assert (tmp_path / "sezgi_config.json").exists()
