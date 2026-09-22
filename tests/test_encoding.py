import os

import pytest
import torch

from assay.encoding import PAD_BLOCK, STATE_BLOCK, build_attention_mask, collate, encode
from assay.labels import LabelAlphabet
from assay.schema import Question, confidence_from_probabilities, render_state

BASE = os.environ.get("ASSAY_TEST_BASE", "Qwen/Qwen3-0.6B-Base")


def test_confidence_bounds():
    assert confidence_from_probabilities([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert confidence_from_probabilities([1 / 3] * 3) == pytest.approx(0.0)
    assert confidence_from_probabilities([0.85, 0.15, 0.0]) == pytest.approx(0.775)


def test_render_state_forms():
    assert render_state("  hi ") == "hi"
    assert render_state({"a": "x", "b": [1, 2]}) == "a: x\nb: [1, 2]"
    assert render_state(["p", {"k": 1}]) == '1. p\n2. {"k": 1}'


def test_question_validation():
    with pytest.raises(ValueError):
        Question(type="choice", instructions="x", options={"only": None})
    with pytest.raises(ValueError):
        Question(type="score", instructions="x", levels=["one"])
    with pytest.raises(ValueError):
        Question(type="bool", instructions="x", options={"a": None, "b": None})
    q = Question(type="score", instructions="x", levels=["low", "high"])
    assert q.keys == ["0", "1"]


def test_block_mask_rules():
    block_ids = torch.tensor([[STATE_BLOCK, STATE_BLOCK, 0, 0, 1, 1, PAD_BLOCK]])
    mask = build_attention_mask(block_ids)[0, 0]
    # state is causal
    assert mask[0, 0] and not mask[0, 1] and mask[1, 0]
    # block 0 sees state and itself, not block 1
    assert mask[3, 0] and mask[3, 2] and mask[3, 3] and not mask[3, 4]
    # block 1 sees state and itself, not block 0
    assert mask[5, 1] and mask[5, 4] and not mask[5, 2] and not mask[5, 3]
    # nobody sees padding except itself
    assert not mask[5, 6] and mask[6, 6]


@pytest.fixture(scope="module")
def tok_alphabet():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    return tok, LabelAlphabet(tok)


def test_encode_positions_restart(tok_alphabet):
    tok, alphabet = tok_alphabet
    qs = [
        Question(type="bool", instructions="Is it urgent?"),
        Question(type="choice", instructions="Team?", options={"billing": "money", "tech": "bugs"}),
    ]
    packed = encode(tok, alphabet, "The site is down.", qs)
    s = packed.state_len
    assert packed.position_ids[:s] == list(range(s))
    for q in packed.questions:
        assert packed.position_ids[q.start] == s
        assert q.readout == q.end - 1
        assert tok.decode([packed.input_ids[q.readout]]).endswith(":")
    assert packed.questions[0].option_token_ids == [alphabet.yes_id, alphabet.no_id]
    assert packed.questions[1].option_token_ids == alphabet.token_ids[:2]


def test_encode_shuffled_order_maps_labels(tok_alphabet):
    tok, alphabet = tok_alphabet
    q = Question(type="choice", instructions="Pick", options={"x": None, "y": None, "z": None})
    packed = encode(tok, alphabet, "s", [q], orders=[[2, 0, 1]])
    # display: A=z, B=x, C=y  -> canonical x->B, y->C, z->A
    ids = packed.questions[0].option_token_ids
    assert ids == [alphabet.token_ids[1], alphabet.token_ids[2], alphabet.token_ids[0]]
    text = tok.decode(packed.input_ids[packed.questions[0].start : packed.questions[0].end])
    assert "A. z" in text and "B. x" in text and "C. y" in text


def test_collate_shapes(tok_alphabet):
    tok, alphabet = tok_alphabet
    q1 = [Question(type="bool", instructions="a?")]
    q2 = [
        Question(type="choice", instructions="b?", options={"p": None, "q": None, "r": None}),
        Question(type="score", instructions="c?", levels=["lo", "hi"]),
    ]
    batch = collate(
        [encode(tok, alphabet, "one", q1), encode(tok, alphabet, "two two", q2)], tok.pad_token_id
    )
    assert batch.input_ids.shape[0] == 2
    assert batch.attention_mask.shape == (2, 1, batch.input_ids.shape[1], batch.input_ids.shape[1])
    assert batch.num_questions == 3
    assert batch.q_option_ids.shape == (3, 3)
    assert batch.q_num_options.tolist() == [2, 3, 2]
    assert (batch.q_option_ids[0, 2] == -1).item()
