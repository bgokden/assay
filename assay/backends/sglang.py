"""Serve the decoder tier through an SGLang server instead of local transformers.

SGLang's /generate accepts `token_ids_logprob` (logprobs for a named set of token ids) and
`return_hidden_states` in the same request, so both halves of a decision come out of one
forward pass: the option-label logits are the readout, and the hidden state at the decision
position feeds the evidence head, which runs here rather than in the server. Temperature and
conformal thresholds also stay client-side, so calibration is unchanged.

    python -m sglang.launch_server --model-path Berk/assay-4b --port 30000 \\
        --enable-return-hidden-states
    client = SGLangClient("http://127.0.0.1:30000", "Berk/assay-4b")
    client.answer(state, questions)

`--enable-return-hidden-states` is required at launch: without it the server rejects a request
that asks for them, and the evidence head has no input. Pass `evidence=False` to answer from
the readout alone against a server started without it.

Questions about one state are sent as separate requests that share a prompt prefix, which
SGLang's RadixAttention caches, so the state is encoded once across them - the same saving our
own packing gives inside a single request, and it also works across requests. Prompts are sent
as token ids rather than text, so the sequence is the one the model was trained on.

Verified against local transformers on assay-0.6b (SGLang 0.5.9): probabilities agree to
1.2e-2 and evidence to 7.7e-5. The probability figure is bf16 rounding rather than a runtime
difference -- our own bf16 path differs from the same fp32 reference by 1.4e-2 -- and it is
worst on a near-uniform distribution, where a small logit shift moves the most probability.

The request fields are documented; the response layout is read defensively because it is not
pinned by a published schema. `verify_against_local` checks a live server against the local
model and is the way to confirm the parsing before trusting a deployment.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import requests
import safetensors.torch
from transformers import AutoTokenizer

from assay.encoding import encode, identity_order
from assay.labels import LabelAlphabet
from assay.model import CONFIG_FILE, HEAD_FILE
from assay.schema import Answer, Question, make_answer


def _resolve(model: str) -> str:
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model)


def _first(payload: Any) -> Any:
    """SGLang returns a dict for one prompt and a list for several."""
    return payload[0] if isinstance(payload, list) else payload


def _readout_vector(hidden: Any) -> np.ndarray:
    """The hidden state at the readout position, whatever nesting the server used.

    /generate returns a row per position it actually computed, inside an entry per sequence --
    which is not the same as a row per prompt token. RadixAttention recomputes only the tail of
    a cached prefix, so the same 25-token prompt comes back as 25 rows cold and 1 row warm. The
    last row of the last sequence is the readout position in both cases.
    """
    array = np.asarray(hidden, dtype=np.float64)
    if array.ndim == 0 or not array.size:
        raise ValueError("server returned an empty hidden state")
    while array.ndim > 1:
        array = array[-1]
    return array


def _logprob_map(entries: Any) -> dict[int, float]:
    """Entries are (logprob, token_id, text) triples in the documented order; accept dicts too."""
    out: dict[int, float] = {}
    for entry in entries or []:
        if isinstance(entry, dict):
            token_id, logprob = entry.get("token_id"), entry.get("logprob")
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            logprob, token_id = entry[0], entry[1]
        else:
            continue
        if token_id is not None and logprob is not None:
            out[int(token_id)] = float(logprob)
    return out


class SGLangClient:
    def __init__(
        self,
        url: str,
        model: str,
        timeout: float = 60.0,
        max_state_tokens: int = 2048,
        evidence: bool = True,
    ) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.max_state_tokens = max_state_tokens
        self.want_evidence = evidence
        path = _resolve(model)
        with open(os.path.join(path, CONFIG_FILE)) as f:
            config = json.load(f)
        # a published repository carries the tokenizer; a training run directory does not, so
        # fall back to the base model the run was trained from
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
        except (OSError, ValueError):
            self.tokenizer = AutoTokenizer.from_pretrained(config["base_model_id"])
        self.alphabet = LabelAlphabet(self.tokenizer)
        self.temperature = float(config.get("temperature", 1.0))
        self.normalize_evidence_input = bool(config.get("normalize_evidence_input", False))
        if config.get("content_term"):
            raise ValueError(
                "this model scores options with a content term, which needs the hidden states "
                "of the option spans, not just the readout; serve it with assay.server"
            )
        head = safetensors.torch.load_file(os.path.join(path, HEAD_FILE))
        self.evidence_weight = head["weight"].float().numpy().reshape(-1)
        self.evidence_bias = float(head["bias"].float().numpy().reshape(-1)[0])
        self.session = requests.Session()

    def tokens(self, state: Any, question: Question) -> tuple[list[int], list[int]]:
        """The token ids of one state-and-question sequence, and its option label ids.

        This is `assay.encoding.encode` for a single question, so the readout position is the
        last token and the prompt is identical to the one transformers sees. The ids are sent
        as ids for that reason: rendering the prompt to a string and letting the server
        retokenize merges the newline pair at the state boundary into one token, which moves
        probabilities by up to 6e-2 -- a wrong answer that looks like a rounding difference.
        """
        packed = encode(
            self.tokenizer,
            self.alphabet,
            state,
            [question],
            orders=[identity_order(question)],
            max_state_tokens=self.max_state_tokens,
        )
        return packed.input_ids, packed.questions[0].option_token_ids

    def _evidence(self, hidden: Any) -> float:
        h = _readout_vector(hidden)
        if h.shape != self.evidence_weight.shape:
            raise ValueError(
                f"hidden state has {h.shape[0]} dimensions, the evidence head expects "
                f"{self.evidence_weight.shape[0]}: this is not the model the client loaded"
            )
        if self.normalize_evidence_input:
            h = (h - h.mean()) / np.sqrt(h.var() + 1e-5)
        return float(1.0 / (1.0 + np.exp(-(float(h @ self.evidence_weight) + self.evidence_bias))))

    def _decide(self, state: Any, question: Question) -> tuple[list[float], float]:
        prompt, ids = self.tokens(state, question)
        response = self.session.post(
            f"{self.url}/generate",
            json={
                "input_ids": prompt,
                "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
                "return_logprob": True,
                "token_ids_logprob": ids,
                "return_hidden_states": self.want_evidence,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = _first(response.json())
        meta = payload.get("meta_info", payload)
        entries = meta.get("output_token_ids_logprobs") or meta.get("output_token_ids_logprob")
        logprobs = _logprob_map(_first(entries) if entries else None)
        missing = [i for i in ids if i not in logprobs]
        if missing:
            raise ValueError(f"server returned no logprob for option token ids {missing[:4]}")
        values = np.array([logprobs[i] for i in ids], dtype=np.float64) / self.temperature
        values -= values.max()
        probs = np.exp(values)
        probs /= probs.sum()
        if not self.want_evidence:
            return probs.tolist(), 1.0
        hidden = payload.get("hidden_states") or meta.get("hidden_states")
        if hidden is None:
            raise ValueError(
                "server returned no hidden states; launch it with --enable-return-hidden-states"
            )
        return probs.tolist(), self._evidence(hidden)

    def answer(self, state: Any, questions: dict[str, Question]) -> dict[str, Answer]:
        out: dict[str, Answer] = {}
        for name, question in questions.items():
            probs, evidence = self._decide(state, question)
            out[name] = make_answer(question, probs, evidence)
        return out


def verify_against_local(
    url: str,
    model: str,
    state: Any,
    questions: dict[str, Question],
    tolerance: float = 0.02,
    evidence: bool = True,
) -> dict[str, float]:
    """Compare a live SGLang deployment with the local transformers path.

    Returns the largest probability and evidence differences, and raises when either is beyond
    `tolerance`. Run this before trusting a deployment: the response layout is not pinned by a
    published schema, and a client that reads the wrong field is wrong quietly.
    """
    import torch

    from assay.model import AssayModel

    local = AssayModel.from_pretrained(_resolve(model), dtype=torch.float32, device="cpu")
    local.eval()
    reference = local.answer(state, questions)
    remote = SGLangClient(url, model, evidence=evidence).answer(state, questions)
    worst_probability = 0.0
    worst_evidence = 0.0
    for name, question in questions.items():
        a, b = reference[name], remote[name]
        worst_probability = max(
            worst_probability,
            max(abs(a.probabilities[k] - b.probabilities[k]) for k in question.keys),
        )
        if evidence:
            worst_evidence = max(worst_evidence, abs(a.evidence - b.evidence))
    if worst_probability > tolerance or worst_evidence > tolerance:
        raise AssertionError(
            f"SGLang disagrees with local transformers: probabilities differ by "
            f"{worst_probability:.3e}, evidence by {worst_evidence:.3e}, tolerance {tolerance}"
        )
    return {"probability": worst_probability, "evidence": worst_evidence}
