"""Serve the decoder tier through an SGLang server instead of local transformers.

SGLang's /generate accepts `token_ids_logprob` (logprobs for a named set of token ids) and
`return_hidden_states` in the same request, so both halves of a decision come out of one
forward pass: the option-label logits are the readout, and the hidden state at the decision
position feeds the evidence head, which runs here rather than in the server. Temperature and
conformal thresholds also stay client-side, so calibration is unchanged.

    python -m sglang.launch_server --model-path Berk/assay-4b --port 30000
    client = SGLangClient("http://127.0.0.1:30000", "Berk/assay-4b")
    client.answer(state, questions)

Questions about one state are sent as separate requests that share a prompt prefix, which
SGLang's RadixAttention caches, so the state is encoded once across them - the same saving our
own packing gives inside a single request, and it also works across requests.

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

from assay.encoding import render_question
from assay.labels import LabelAlphabet
from assay.model import CONFIG_FILE, HEAD_FILE
from assay.schema import Answer, Question, make_answer, render_state


def _resolve(model: str) -> str:
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model)


def _first(payload: Any) -> Any:
    """SGLang returns a dict for one prompt and a list for several."""
    return payload[0] if isinstance(payload, list) else payload


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
    def __init__(self, url: str, model: str, timeout: float = 60.0, max_state_tokens: int = 2048):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.max_state_tokens = max_state_tokens
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
        head = safetensors.torch.load_file(os.path.join(path, HEAD_FILE))
        self.evidence_weight = head["weight"].float().numpy().reshape(-1)
        self.evidence_bias = float(head["bias"].float().numpy().reshape(-1)[0])
        self.session = requests.Session()

    def prompt(self, state: Any, question: Question) -> str:
        """The same text `assay.encoding` builds, so the readout position is identical."""
        text = render_state(state) + "\n"
        ids = self.tokenizer.encode(text, add_special_tokens=False)[: self.max_state_tokens]
        return self.tokenizer.decode(ids) + render_question(
            question, self.alphabet, list(range(len(question.keys)))
        )

    def option_ids(self, question: Question) -> list[int]:
        if question.type == "bool":
            return [self.alphabet.yes_id, self.alphabet.no_id]
        return self.alphabet.token_ids[: len(question.keys)]

    def _evidence(self, hidden: list[float]) -> float:
        h = np.asarray(hidden, dtype=np.float64)
        if self.normalize_evidence_input:
            h = (h - h.mean()) / np.sqrt(h.var() + 1e-5)
        return float(1.0 / (1.0 + np.exp(-(float(h @ self.evidence_weight) + self.evidence_bias))))

    def _decide(self, state: Any, question: Question) -> tuple[list[float], float]:
        ids = self.option_ids(question)
        response = self.session.post(
            f"{self.url}/generate",
            json={
                "text": self.prompt(state, question),
                "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
                "return_logprob": True,
                "token_ids_logprob": ids,
                "return_hidden_states": True,
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
        hidden = payload.get("hidden_states") or meta.get("hidden_states")
        if hidden is None:
            raise ValueError("server did not return hidden states; start it so /generate can")
        vector = hidden[-1] if isinstance(hidden[0], list) else hidden
        return probs.tolist(), self._evidence(vector)

    def answer(self, state: Any, questions: dict[str, Question]) -> dict[str, Answer]:
        out: dict[str, Answer] = {}
        for name, question in questions.items():
            probs, evidence = self._decide(state, question)
            out[name] = make_answer(question, probs, evidence)
        return out


def verify_against_local(
    url: str, model: str, state: Any, questions: dict[str, Question], tolerance: float = 0.02
) -> dict[str, float]:
    """Compare a live SGLang deployment with the local transformers path; returns the largest
    probability difference per question. Run this before trusting a deployment: the response
    layout is not pinned by a published schema."""
    from assay.evaluate import load_model

    local = load_model(model)
    local.eval()
    reference = local.answer(state, questions)
    remote = SGLangClient(url, model).answer(state, questions)
    worst = {}
    for name, question in questions.items():
        diffs = [
            abs(reference[name].probabilities[k] - remote[name].probabilities[k])
            for k in question.keys
        ]
        worst[name] = max(diffs)
    bad = {n: d for n, d in worst.items() if d > tolerance}
    if bad:
        raise AssertionError(f"SGLang and local answers differ by more than {tolerance}: {bad}")
    return worst
