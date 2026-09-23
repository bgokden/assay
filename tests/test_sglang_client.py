"""The SGLang client is tested against a stub server: the request fields are documented, the
response layout is not pinned by a published schema, so the parsing is what needs covering.
A live deployment must still be checked with assay.backends.sglang.verify_against_local.

The layouts here are the ones a live SGLang 0.5.9 server actually returned. In particular the
number of hidden-state rows is not the number of prompt tokens: RadixAttention recomputes only
the tail of a cached prefix, so the same prompt gave 54 rows cold and 1 row warm."""

import json
import math
import os

import numpy as np
import pytest

from assay.schema import Question

BASE = os.environ.get("ASSAY_TEST_BASE", "Qwen/Qwen3-0.6B-Base")
QUESTIONS = {
    "team": Question(
        type="choice",
        instructions="Which team should handle this?",
        options={"billing": "Charges", "technical": "Bugs"},
    ),
    "urgent": Question(type="bool", instructions="Is it urgent?"),
}


class StubResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class StubSession:
    """Answers /generate with the fields SGLang documents, recording what was asked."""

    def __init__(self, hidden_size, logprobs, style="tuple", hidden="nested", rows=1):
        self.hidden_size = hidden_size
        self.logprobs = logprobs
        self.style = style
        self.hidden = hidden
        self.rows = rows
        self.requests = []

    def post(self, url, json=None, timeout=None):
        self.requests.append(json)
        ids = json["token_ids_logprob"]
        values = [self.logprobs[i % len(self.logprobs)] for i in range(len(ids))]
        if self.style == "tuple":
            entries = [[v, i, None] for v, i in zip(values, ids)]
        else:
            entries = [{"logprob": v, "token_id": i} for v, i in zip(values, ids)]
        rows = [[float(i)] * self.hidden_size for i in range(self.rows)]
        meta = {"output_token_ids_logprobs": [entries]}
        if self.hidden == "nested":  # what a live server returns: [sequence][position][dim]
            meta["hidden_states"] = [rows]
        elif self.hidden == "flat":
            meta["hidden_states"] = rows
        return StubResponse({"text": " A", "meta_info": meta})


@pytest.fixture(scope="module")
def client(saved_model):
    """A client built against a saved model directory, with its HTTP session stubbed."""
    from assay.backends.sglang import SGLangClient

    return SGLangClient.__new__(SGLangClient), saved_model


def build(path, style="tuple", logprobs=(-0.2, -1.6), **kwargs):
    from assay.backends.sglang import SGLangClient

    c = SGLangClient("http://stub", path)
    c.session = StubSession(c.evidence_weight.shape[0], list(logprobs), style, **kwargs)
    return c


@pytest.mark.parametrize("style", ["tuple", "dict"])
def test_answers_match_the_returned_logprobs(client, style):
    _, path = client
    c = build(path, style)
    answers = c.answer({"message": "Refund me"}, QUESTIONS)
    assert set(answers) == {"team", "urgent"}
    for name, question in QUESTIONS.items():
        probs = answers[name].probabilities
        assert set(probs) == set(question.keys)
        assert abs(sum(probs.values()) - 1.0) < 1e-6
    # two options with logprobs -0.2 and -1.6, scaled by the fitted temperature
    expected = np.exp(np.array([-0.2, -1.6]) / c.temperature)
    expected /= expected.sum()
    assert answers["team"].probabilities["billing"] == pytest.approx(expected[0], abs=1e-6)


def test_request_carries_the_documented_fields(client):
    _, path = client
    c = build(path)
    c.answer("a state", {"urgent": QUESTIONS["urgent"]})
    sent = c.session.requests[0]
    assert sent["return_logprob"] is True
    assert sent["return_hidden_states"] is True
    assert sent["sampling_params"]["max_new_tokens"] == 1
    assert sent["token_ids_logprob"] == [c.alphabet.yes_id, c.alphabet.no_id]
    assert c.tokenizer.decode(sent["input_ids"]).rstrip().endswith("Answer (yes or no):")
    assert "a state" in c.tokenizer.decode(sent["input_ids"])


def test_the_prompt_is_sent_as_ids_not_text(client):
    """Letting the server retokenize merges the newline pair at the state boundary into one
    token. It still answers; it answers a differently tokenized prompt, and on assay-0.6b that
    moved probabilities by up to 6.1e-2."""
    _, path = client
    c = build(path)
    c.answer({"message": "Refund me"}, {"urgent": QUESTIONS["urgent"]})
    sent = c.session.requests[0]
    assert "text" not in sent
    expected, _ = c.tokens({"message": "Refund me"}, QUESTIONS["urgent"])
    assert sent["input_ids"] == expected
    # the readout is the last token, which is what the evidence head and the logprobs read
    assert c.tokenizer.decode(expected).endswith(":")


@pytest.mark.parametrize("hidden,rows", [("nested", 1), ("nested", 54), ("flat", 1), ("flat", 54)])
def test_the_readout_row_is_taken_whatever_the_nesting(client, hidden, rows):
    """A cache hit returns one row and a cold prefill returns one per prompt token; the last
    row of the last sequence is the readout in both cases."""
    _, path = client
    c = build(path, hidden=hidden, rows=rows)
    answers = c.answer("x", {"urgent": QUESTIONS["urgent"]})
    vector = [float(rows - 1)] * c.evidence_weight.shape[0]
    assert answers["urgent"].evidence == pytest.approx(c._evidence(vector), abs=1e-9)


def test_a_hidden_state_of_the_wrong_width_is_an_error(client):
    """Pointing the client at a different model than the server holds, which otherwise shows
    up as answers that are merely a bit off."""
    _, path = client
    c = build(path)
    c.session.hidden_size = c.evidence_weight.shape[0] // 2
    with pytest.raises(ValueError, match="evidence head expects"):
        c.answer("x", {"urgent": QUESTIONS["urgent"]})


def test_evidence_can_be_skipped(client):
    """A server started without --enable-return-hidden-states can still answer the readout."""
    _, path = client
    c = build(path)
    c.want_evidence = False
    answers = c.answer("x", {"urgent": QUESTIONS["urgent"]})
    assert answers["urgent"].evidence == 1.0
    assert c.session.requests[0]["return_hidden_states"] is False


def test_missing_hidden_states_is_an_explicit_error(client):
    _, path = client
    c = build(path)

    def post(url, json=None, timeout=None):
        return StubResponse(
            {
                "meta_info": {
                    "output_token_ids_logprobs": [
                        [[-0.1, i, None] for i in json["token_ids_logprob"]]
                    ]
                }
            }
        )

    c.session.post = post
    with pytest.raises(ValueError, match="hidden states"):
        c.answer("x", {"urgent": QUESTIONS["urgent"]})


def test_missing_option_logprob_is_an_explicit_error(client):
    _, path = client
    c = build(path)

    def post(url, json=None, timeout=None):
        return StubResponse(
            {
                "meta_info": {
                    "output_token_ids_logprobs": [[]],
                    "hidden_states": [[[0.0] * c.evidence_weight.shape[0]]],
                }
            }
        )

    c.session.post = post
    with pytest.raises(ValueError, match="no logprob"):
        c.answer("x", {"urgent": QUESTIONS["urgent"]})


def test_evidence_matches_the_local_head(client):
    _, path = client
    c = build(path)
    vector = [0.01] * c.evidence_weight.shape[0]
    expected = 1.0 / (
        1.0 + math.exp(-(float(np.asarray(vector) @ c.evidence_weight) + c.evidence_bias))
    )
    assert c._evidence(vector) == pytest.approx(expected, abs=1e-9)
    assert json.dumps({"ok": True})  # the module is importable without a live server
