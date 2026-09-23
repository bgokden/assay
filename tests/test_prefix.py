import os

import pytest
import torch

from assay.encoding import encode, identity_order
from assay.model import AssayModel
from assay.prefix import prefix_answers, split_packed
from assay.schema import Question

BASE = os.environ.get("ASSAY_TEST_BASE", "Qwen/Qwen3-0.6B-Base")
DEVICE = os.environ.get("ASSAY_TEST_DEVICE", "cuda")

pytestmark = pytest.mark.skipif(
    DEVICE == "cuda" and not torch.cuda.is_available(), reason="needs a GPU"
)

STATE = {"channel": "email", "message": "I was charged twice for order A-104. Please refund it."}
QUESTIONS = [
    Question(
        type="choice",
        instructions="Which team should handle this?",
        options={"billing": "Charges and refunds", "technical": "Faults", "account": "Sign-in"},
    ),
    Question(type="bool", instructions="Is the customer asking for money back?"),
    Question(type="score", instructions="How urgent is this?", levels=["Can wait", "Today", "Now"]),
]


@pytest.fixture(scope="module")
def model():
    # float32: the two paths are mathematically identical, and bf16 kernels differ by ~1e-2
    # with the batch shape, which would make an equivalence test meaningless
    m = AssayModel.from_base(BASE, lora_r=None, dtype=torch.float32, device=DEVICE)
    m.eval()
    return m


def test_the_prefix_path_answers_as_the_packed_path(model):
    """One encoding of the state, reused per question, must give the block mask's answers."""
    names = [f"q{i}" for i in range(len(QUESTIONS))]
    packed = model.answer(STATE, dict(zip(names, QUESTIONS)))
    prefixed = prefix_answers(model, STATE, QUESTIONS)
    for name, question, answer in zip(names, QUESTIONS, prefixed):
        expected = packed[name]
        for key in expected.probabilities:
            assert answer.probabilities[key] == pytest.approx(expected.probabilities[key], abs=1e-4)
        assert answer.evidence == pytest.approx(expected.evidence, abs=1e-4)
        assert set(answer.probabilities) == set(question.keys)


def test_a_single_question_is_unchanged(model):
    one = QUESTIONS[0]
    assert prefix_answers(model, STATE, [one])[0].probabilities == pytest.approx(
        model.answer(STATE, {"q": one})["q"].probabilities, abs=1e-4
    )


def test_splitting_a_pack_keeps_the_state_and_every_block(model):
    """The split is the same tokens as the pack, cut at the block boundaries."""
    packed = encode(
        model.tokenizer,
        model.alphabet,
        STATE,
        QUESTIONS,
        orders=[identity_order(q) for q in QUESTIONS],
    )
    split = split_packed(packed)
    assert len(split.state_ids) == packed.state_len
    assert len(split.blocks) == len(QUESTIONS)
    for block, question, readout in zip(split.blocks, packed.questions, split.readouts):
        assert len(block) == question.end - question.start
        assert readout == question.readout - question.start
    assert split.state_ids + [t for b in split.blocks for t in b] == packed.input_ids
