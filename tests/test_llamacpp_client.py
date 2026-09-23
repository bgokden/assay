"""The llama.cpp client against a stub server.

The request fields are documented; the response layout differs between builds (`top_probs` or
`top_logprobs`, `prob` or `logprob`), so the parsing is what needs covering, along with the
two failures a real deployment produces: a label token outside the candidate list, and a
server started without the embedding flags. A live deployment is checked with
`assay.backends.llamacpp.verify_against_local`.
"""

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
STATE = {"message": "Refund me"}


class StubResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class StubSession:
    """Answers /completion and /embeddings the way llama-server does."""

    def __init__(self, hidden_size, logprobs, key="logprob", field="top_probs", drop=()):
        self.hidden_size = hidden_size
        self.logprobs = logprobs
        self.key = key
        self.field = field
        self.drop = set(drop)
        self.requests = []
        self.embedding = [0.0] * hidden_size

    def post(self, url, json=None, timeout=None):
        self.requests.append((url, json))
        if url.endswith("/embeddings"):
            return StubResponse([{"index": 0, "embedding": self.embedding}])
        candidates = []
        for position, token in enumerate(range(10, 10 + len(self.logprobs))):
            if token in self.drop:
                continue
            value = self.logprobs[position]
            entry = {"id": token, "token": f"t{token}"}
            entry[self.key] = value if self.key == "logprob" else math.exp(value)
            candidates.append(entry)
        return StubResponse({"completion_probabilities": [{self.field: candidates}]})


@pytest.fixture(scope="module")
def model_path(saved_model):
    """The session-wide copy from conftest: saving another costs a copy of the weights."""
    return saved_model


def build(path, **kwargs):
    from assay.backends.llamacpp import LlamaCppClient

    client = LlamaCppClient("http://stub", path)
    ids = client.tokens(STATE, QUESTIONS["team"])[1]
    logprobs = kwargs.pop("logprobs", [-0.2, -1.6])
    session = StubSession(client.evidence_weight.shape[0], logprobs, **kwargs)
    # the stub hands back ids 10, 11, ...; point the client's labels at those
    session.token_ids = ids
    client.session = session
    client.tokens = lambda state, question, _ids=ids: (
        [1, 2, 3],
        list(range(10, 10 + len(question.keys))),
    )
    return client


@pytest.mark.parametrize("field", ["top_probs", "top_logprobs"])
@pytest.mark.parametrize("key", ["logprob", "prob"])
def test_answers_match_the_returned_candidates(model_path, field, key):
    client = build(model_path, field=field, key=key)
    answers = client.answer(STATE, QUESTIONS)
    assert set(answers) == {"team", "urgent"}
    expected = np.exp(np.array([-0.2, -1.6]) / client.temperature)
    expected /= expected.sum()
    for name, question in QUESTIONS.items():
        probabilities = [answers[name].probabilities[k] for k in question.keys]
        assert probabilities == pytest.approx(expected.tolist(), abs=1e-6)
        assert 0.0 <= answers[name].evidence <= 1.0


def test_the_fitted_temperature_is_applied(model_path):
    """Forgetting it is the mistake that makes a runtime look wrong: it sharpens everything."""
    client = build(model_path)
    client.temperature = 2.0
    answers = client.answer(STATE, QUESTIONS)
    expected = np.exp(np.array([-0.2, -1.6]) / 2.0)
    expected /= expected.sum()
    assert [answers["urgent"].probabilities[k] for k in ("yes", "no")] == pytest.approx(
        expected.tolist(), abs=1e-6
    )


def test_a_missing_label_is_an_error_not_a_zero(model_path):
    """A label outside the candidate list would silently distort the distribution."""
    client = build(model_path, drop=(11,))
    with pytest.raises(ValueError, match="were not among the top"):
        client.answer(STATE, QUESTIONS)


def test_a_server_without_embeddings_says_so(model_path):
    client = build(model_path)
    client.session.embedding = None
    with pytest.raises(ValueError, match="--embeddings --pooling last"):
        client.answer(STATE, QUESTIONS)


def test_a_wrong_pooling_is_caught(model_path):
    client = build(model_path)
    client.session.embedding = [0.0] * 7
    with pytest.raises(ValueError, match="check --pooling last"):
        client.answer(STATE, QUESTIONS)


def test_evidence_can_be_skipped(model_path):
    client = build(model_path)
    client.want_evidence = False
    answers = client.answer(STATE, QUESTIONS)
    assert all(a.evidence == 1.0 for a in answers.values())
    assert all(url.endswith("/completion") for url, _ in client.session.requests)


def test_the_state_prefix_is_cached_across_questions(model_path):
    client = build(model_path)
    client.answer(STATE, QUESTIONS)
    completions = [body for url, body in client.session.requests if url.endswith("/completion")]
    assert len(completions) == len(QUESTIONS)
    assert all(body["cache_prompt"] for body in completions)
    assert all(body["n_predict"] == 1 for body in completions)
