"""Serve the decoder tier through llama.cpp, which is how it runs on a laptop.

A decision needs two things from the runtime: the logits over the option-label tokens at the
readout position, and the hidden state there for the evidence head. `llama-server` gives both,
in two calls -- `/completion` with `n_probs` returns the token distribution at the position it
is about to generate, and `/embeddings` with last-token pooling returns that position's hidden
state. Everything else (temperature, the evidence head, conformal thresholds) runs here, so a
model behaves the same whether it is served by transformers, SGLang or llama.cpp.

    python convert_hf_to_gguf.py <merged model dir> --outfile assay.gguf --outtype f16
    llama-server -m assay.gguf --port 8081 --embeddings --pooling last --embd-normalize -1
    client = LlamaCppClient("http://127.0.0.1:8081", "Berk/assay-0.6b")
    client.answer(state, questions)

The server flags matter. Without `--embeddings --pooling last --embd-normalize -1` there is no
evidence score, and a normalised embedding would be the wrong input for the head; pass
`evidence=False` to skip it and answer from the readout alone.

llama.cpp cannot pack questions into one sequence, so each question is its own request. They
share the state prefix, and `cache_prompt` makes the server reuse it, so the state is encoded
once for the first question and reused by the rest.

Verified against local transformers on assay-0.6b (f16 GGUF, CPU): probabilities agree to
1.4e-3, the hidden state to a cosine similarity of 1.000000, and the evidence score to four
decimals. `verify_against_local` runs that check against your own deployment.
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

DEFAULT_CANDIDATES = 300


def _resolve(model: str) -> str:
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model)


def _logprobs(entry: dict[str, Any]) -> dict[int, float]:
    """The candidate list, whatever this build calls it, as token id -> log probability."""
    candidates = entry.get("top_probs") or entry.get("top_logprobs") or []
    out: dict[int, float] = {}
    for candidate in candidates:
        token = candidate.get("id")
        if token is None:
            continue
        if "logprob" in candidate:
            out[token] = float(candidate["logprob"])
        elif "prob" in candidate and candidate["prob"] > 0:
            out[token] = float(np.log(candidate["prob"]))
    return out


class LlamaCppClient:
    def __init__(
        self,
        url: str,
        model: str,
        timeout: float = 120.0,
        max_state_tokens: int = 2048,
        candidates: int = DEFAULT_CANDIDATES,
        evidence: bool = True,
    ) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.max_state_tokens = max_state_tokens
        self.candidates = candidates
        self.want_evidence = evidence
        path = _resolve(model)
        with open(os.path.join(path, CONFIG_FILE)) as f:
            config = json.load(f)
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
        last token and the prompt is identical to the one transformers sees.
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

    def _probabilities(self, prompt: list[int], option_ids: list[int]) -> list[float]:
        response = self.session.post(
            f"{self.url}/completion",
            json={
                "prompt": prompt,
                "n_predict": 1,
                "n_probs": self.candidates,
                "temperature": 0.0,
                "post_sampling_probs": False,
                "cache_prompt": True,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        entries = payload.get("completion_probabilities")
        if not entries:
            raise ValueError("no completion_probabilities: the server ignored n_probs")
        logprobs = _logprobs(entries[0])
        missing = [i for i in option_ids if i not in logprobs]
        if missing:
            raise ValueError(
                f"option label tokens {missing[:4]} were not among the top {self.candidates} "
                "candidates; raise `candidates`"
            )
        # the fitted temperature applies to the logits, and the softmax constant cancels when
        # the distribution is renormalised over the option labels
        values = np.array([logprobs[i] for i in option_ids], dtype=np.float64) / self.temperature
        values -= values.max()
        probs = np.exp(values)
        return (probs / probs.sum()).tolist()

    def _evidence(self, prompt: list[int]) -> float:
        response = self.session.post(
            f"{self.url}/embeddings",
            json={"input": prompt},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        row = payload[0] if isinstance(payload, list) else payload
        vector = row.get("embedding")
        if vector is None:
            raise ValueError("no embedding: start the server with --embeddings --pooling last")
        if isinstance(vector[0], list):
            vector = vector[-1]
        h = np.asarray(vector, dtype=np.float64)
        if h.shape != self.evidence_weight.shape:
            raise ValueError(
                f"embedding has {h.shape[0]} dimensions, the evidence head expects "
                f"{self.evidence_weight.shape[0]}; check --pooling last"
            )
        if self.normalize_evidence_input:
            h = (h - h.mean()) / np.sqrt(h.var() + 1e-5)
        return float(1.0 / (1.0 + np.exp(-(float(h @ self.evidence_weight) + self.evidence_bias))))

    def answer(self, state: Any, questions: dict[str, Question]) -> dict[str, Answer]:
        out: dict[str, Answer] = {}
        for name, question in questions.items():
            prompt, option_ids = self.tokens(state, question)
            probabilities = self._probabilities(prompt, option_ids)
            evidence = self._evidence(prompt) if self.want_evidence else 1.0
            out[name] = make_answer(question, probabilities, evidence)
        return out


def verify_against_local(
    url: str,
    model: str,
    state: Any,
    questions: dict[str, Question],
    tolerance: float = 0.01,
    evidence: bool = True,
) -> dict[str, float]:
    """Compare a live llama.cpp deployment with the local transformers path.

    Returns the largest probability and evidence differences, and raises when either is beyond
    `tolerance`. Run this before trusting a deployment: a GGUF is a converted, usually
    quantised copy of the weights, and the conversion is where a deployment goes quietly wrong.
    """
    import torch

    from assay.model import AssayModel

    client = LlamaCppClient(url, model, evidence=evidence)
    local = AssayModel.from_pretrained(_resolve(model), dtype=torch.float32, device="cpu")
    local.eval()
    theirs = client.answer(state, questions)
    ours = local.answer(state, questions)
    worst_probability = 0.0
    worst_evidence = 0.0
    for name, question in questions.items():
        a, b = ours[name], theirs[name]
        worst_probability = max(
            worst_probability,
            max(abs(a.probabilities[k] - b.probabilities[k]) for k in question.keys),
        )
        if evidence:
            worst_evidence = max(worst_evidence, abs(a.evidence - b.evidence))
    if worst_probability > tolerance or worst_evidence > tolerance:
        raise AssertionError(
            f"llama.cpp disagrees with local transformers: probabilities differ by "
            f"{worst_probability:.3e}, evidence by {worst_evidence:.3e}, tolerance {tolerance}"
        )
    return {"probability": worst_probability, "evidence": worst_evidence}
