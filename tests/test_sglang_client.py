"""The SGLang client is tested against a stub server: the request fields are documented, the
response layout is not pinned by a published schema, so the parsing is what needs covering.
A live deployment must still be checked with assay.backends.sglang.verify_against_local."""

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

    def __init__(self, hidden_size, logprobs, style="tuple"):
        self.hidden_size = hidden_size
        self.logprobs = logprobs
        self.style = style
        self.requests = []

    def post(self, url, json=None, timeout=None):
        self.requests.append(json)
        ids = json["token_ids_logprob"]
        values = [self.logprobs[i % len(self.logprobs)] for i in range(len(ids))]
        if self.style == "tuple":
            entries = [[v, i, None] for v, i in zip(values, ids)]
        else:
            entries = [{"logprob": v, "token_id": i} for v, i in zip(values, ids)]
        return StubResponse(
            {
                "text": " A",
                "meta_info": {"output_token_ids_logprobs": [entries]},
                "hidden_states": [[0.0] * self.hidden_size],
            }
        )


@pytest.fixture(scope="module")
def client(saved_model):
    """A client built against a saved model directory, with its HTTP session stubbed."""
    from assay.backends.sglang import SGLangClient

    return SGLangClient.__new__(SGLangClient), saved_model


def build(path, style="tuple", logprobs=(-0.2, -1.6)):
    from assay.backends.sglang import SGLangClient

    c = SGLangClient("http://stub", path)
    c.session = StubSession(c.evidence_weight.shape[0], list(logprobs), style)
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
    assert sent["text"].rstrip().endswith("Answer (yes or no):")
    assert "a state" in sent["text"]


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
                "meta_info": {"output_token_ids_logprobs": [[]]},
                "hidden_states": [[0.0] * c.evidence_weight.shape[0]],
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
