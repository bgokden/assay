"""Compiled-function decision model: an encoder turns the question into parameters and the
state into token embeddings; a decision is a small computation between the two.

    state  --encoder-->  token embeddings H (L x d)              (once per state)
    question --encoder--> query vectors Q (M x d), option vectors O (K x d)   (once per question)
    decision: C = CrossAttention(Q, H); z = MLP(C); logits_k = z^T W o_k + b^T o_k  (microseconds)

Options are scored by their own encoded content, so there is no label alphabet, the answer is
permutation-equivariant by construction, and compiled questions can be cached and reused
across states. Evidence comes from a head on z and the attention entropy.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any

import safetensors.torch
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

from assay.schema import Answer, Question, make_answer, render_state

CONFIG_FILE = "assay_compiled_config.json"
WEIGHTS_FILE = "assay_compiled.safetensors"


def option_texts(q: Question) -> list[str]:
    """One text per option in canonical key order, carrying the instruction as context."""
    ins = q.instructions.strip()
    if q.type == "bool":
        yes = q.yes.strip() if q.yes else "the answer is yes"
        no = q.no.strip() if q.no else "the answer is no"
        return [f"{ins} Yes: {yes}", f"{ins} No: {no}"]
    if q.type == "choice":
        out = []
        for key, desc in q.options.items():
            shown = key.replace("_", " ")
            out.append(f"{ins} Option: {shown}" + (f": {desc}" if desc else ""))
        return out
    return [f"{ins} Level {i + 1} of {len(q.levels)}: {lvl}" for i, lvl in enumerate(q.levels)]


@dataclasses.dataclass
class CompiledQuestion:
    queries: torch.Tensor  # (M, d)
    options: torch.Tensor  # (K, d)
    keys: list[str]


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)


class CompiledModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        tokenizer,
        encoder_id: str,
        slots: int = 4,
        heads: int = 8,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.encoder_id = encoder_id
        d = encoder.config.hidden_size
        self.d = d
        self.slots = nn.Parameter(torch.randn(slots, d) * 0.02)
        self.query_proj = nn.Linear(d, d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.reader = nn.Sequential(nn.Linear(slots * d, d), nn.GELU(), nn.Linear(d, d))
        self.bilinear = nn.Linear(d, d, bias=False)
        self.option_bias = nn.Linear(d, 1, bias=False)
        self.evidence_head = nn.Linear(d + 1, 1)
        self.temperature = temperature
        self.max_state_tokens = 512
        self.max_option_tokens = 96
        self.hybrid = False  # for evaluate.predict compatibility

    # --- construction ---------------------------------------------------------------------

    @classmethod
    def from_encoder(cls, encoder_id: str, device: str = "cuda", **kw) -> CompiledModel:
        tokenizer = AutoTokenizer.from_pretrained(encoder_id)
        encoder = AutoModel.from_pretrained(encoder_id, dtype=torch.float32)
        model = cls(encoder, tokenizer, encoder_id, **kw)
        return model.to(device)

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cuda") -> CompiledModel:
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download

            path = snapshot_download(path)
        with open(os.path.join(path, CONFIG_FILE)) as f:
            config = json.load(f)
        tokenizer = AutoTokenizer.from_pretrained(path)
        encoder = AutoModel.from_pretrained(path, dtype=torch.float32)
        model = cls(
            encoder,
            tokenizer,
            config["encoder_id"],
            slots=config["slots"],
            heads=config["heads"],
            temperature=config.get("temperature", 1.0),
        )
        state = safetensors.torch.load_file(os.path.join(path, WEIGHTS_FILE))
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [m for m in missing if not m.startswith("encoder.")]
        if missing or unexpected:
            raise ValueError(f"weights mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
        return model.to(device)

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.encoder.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)
        head = {
            k: v.detach().cpu()
            for k, v in self.state_dict().items()
            if not k.startswith("encoder.")
        }
        safetensors.torch.save_file(head, os.path.join(path, WEIGHTS_FILE))
        with open(os.path.join(path, CONFIG_FILE), "w") as f:
            json.dump(
                {
                    "encoder_id": self.encoder_id,
                    "slots": self.slots.shape[0],
                    "heads": self.attn.num_heads,
                    "temperature": self.temperature,
                    "arch": "cross" if type(self).__name__ == "CrossEncoderModel" else "compiled",
                },
                f,
                indent=2,
            )

    @property
    def device(self) -> torch.device:
        return self.slots.device

    # --- encoding --------------------------------------------------------------------------

    def encode_tokens(self, texts: list[str], max_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self.tokenizer(
            texts, padding=True, truncation=True, max_length=max_tokens, return_tensors="pt"
        ).to(self.device)
        hidden = self.encoder(**batch).last_hidden_state
        return hidden, batch["attention_mask"]

    def encode_states(self, states: list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_tokens([render_state(s) for s in states], self.max_state_tokens)

    def compile_batch(
        self, questions: list[Question]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns queries (B, M, d), option vectors (B, K_max, d) and a validity mask (B, K_max)."""
        ins_hidden, ins_mask = self.encode_tokens(
            [q.instructions.strip() for q in questions], self.max_option_tokens
        )
        q_vec = mean_pool(ins_hidden, ins_mask)  # (B, d)
        flat: list[str] = []
        counts: list[int] = []
        for q in questions:
            texts = option_texts(q)
            flat.extend(texts)
            counts.append(len(texts))
        opt_hidden, opt_mask = self.encode_tokens(flat, self.max_option_tokens)
        opt_vec = mean_pool(opt_hidden, opt_mask)  # (sum K, d)
        k_max = max(counts)
        options = opt_vec.new_zeros((len(questions), k_max, self.d))
        valid = torch.zeros((len(questions), k_max), dtype=torch.bool, device=opt_vec.device)
        start = 0
        for i, k in enumerate(counts):
            options[i, :k] = opt_vec[start : start + k]
            valid[i, :k] = True
            start += k
        queries = self.slots.unsqueeze(0) + self.query_proj(q_vec).unsqueeze(1)  # (B, M, d)
        return queries, options, valid

    # --- decision --------------------------------------------------------------------------

    def decide(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
        queries: torch.Tensor,
        options: torch.Tensor,
        valid: torch.Tensor,
    ):
        """hidden (B, L, d) state tokens; queries (B, M, d); options (B, K, d) -> logits (B, K), evidence (B,)."""
        context, weights = self.attn(
            queries,
            hidden,
            hidden,
            key_padding_mask=~mask.bool(),
            need_weights=True,
            average_attn_weights=True,
        )
        z = self.reader(context.flatten(1))  # (B, d)
        logits = torch.einsum("bd,bkd->bk", self.bilinear(z), options) / (
            self.d**0.5
        ) + self.option_bias(options).squeeze(-1)
        logits = logits.masked_fill(~valid, float("-inf"))
        # attention entropy over state tokens, averaged over slots: low entropy = focused evidence
        ent = (
            -(weights.clamp(min=1e-9) * weights.clamp(min=1e-9).log())
            .sum(-1)
            .mean(-1, keepdim=True)
        )  # (B, 1)
        evidence = self.evidence_head(torch.cat([z, ent], dim=-1)).squeeze(-1)
        return logits, evidence

    def forward(self, states: list[Any], questions: list[Question]):
        hidden, mask = self.encode_states(states)
        queries, options, valid = self.compile_batch(questions)
        return self.decide(hidden, mask, queries, options, valid)

    @torch.no_grad()
    def answer(self, state: Any, questions: dict[str, Question]) -> dict[str, Answer]:
        names = list(questions)
        qs = [questions[n] for n in names]
        hidden, mask = self.encode_states([state])
        queries, options, valid = self.compile_batch(qs)
        hidden = hidden.expand(len(qs), -1, -1)
        mask = mask.expand(len(qs), -1)
        logits, evidence = self.decide(hidden, mask, queries, options, valid)
        probs = torch.softmax(logits / self.temperature, dim=-1)
        ev = torch.sigmoid(evidence)
        out = {}
        for i, (n, q) in enumerate(zip(names, qs)):
            k = len(q.keys)
            out[n] = make_answer(q, probs[i, :k].tolist(), ev[i].item())
        return out

    @torch.no_grad()
    def zero_shot_logits(self, states: list[Any], questions: list[Question], scale: float = 20.0):
        """Untrained baseline: cosine similarity between the pooled state and each option text."""
        hidden, mask = self.encode_states(states)
        s = F.normalize(mean_pool(hidden, mask), dim=-1)
        _, options, valid = self.compile_batch(questions)
        o = F.normalize(options, dim=-1)
        logits = torch.einsum("bd,bkd->bk", s, o) * scale
        return logits.masked_fill(~valid, float("-inf"))


class CrossEncoderModel(CompiledModel):
    """Comparison tier: one encoder pass per (state, option) pair, scored by a linear head.
    Joint attention between state and option, at K passes per question."""

    def __init__(
        self,
        encoder: nn.Module,
        tokenizer,
        encoder_id: str,
        slots: int = 4,
        heads: int = 8,
        temperature: float = 1.0,
    ):
        super().__init__(
            encoder, tokenizer, encoder_id, slots=slots, heads=heads, temperature=temperature
        )
        self.pair_head = nn.Linear(self.d, 1)
        self.pair_evidence = nn.Linear(self.d, 1)
        self.max_pair_tokens = 640

    def forward(self, states: list[Any], questions: list[Question]):
        state_texts = [render_state(s) for s in states]
        pairs: list[str] = []
        counts: list[int] = []
        for text, q in zip(state_texts, questions):
            opts = option_texts(q)
            pairs.extend(f"{o}{self.tokenizer.sep_token or ' | '}{text}" for o in opts)
            counts.append(len(opts))
        hidden, mask = self.encode_tokens(pairs, self.max_pair_tokens)
        pooled = mean_pool(hidden, mask)
        scores = self.pair_head(pooled).squeeze(-1)
        k_max = max(counts)
        logits = scores.new_full((len(questions), k_max), float("-inf"))
        evidence = scores.new_zeros(len(questions))
        start = 0
        for i, k in enumerate(counts):
            logits[i, :k] = scores[start : start + k]
            evidence[i] = self.pair_evidence(pooled[start : start + k].mean(0)).squeeze(-1)
            start += k
        return logits, evidence

    @torch.no_grad()
    def answer(self, state: Any, questions: dict[str, Question]) -> dict[str, Answer]:
        names = list(questions)
        qs = [questions[n] for n in names]
        logits, evidence = self.forward([state] * len(qs), qs)
        probs = torch.softmax(logits / self.temperature, dim=-1)
        ev = torch.sigmoid(evidence)
        return {
            n: make_answer(q, probs[i, : len(q.keys)].tolist(), ev[i].item())
            for i, (n, q) in enumerate(zip(names, qs))
        }


def load_any(path: str, device: str = "cuda") -> CompiledModel:
    """Load a compiled or cross-encoder model saved by either class."""
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download

        path = snapshot_download(path)
    with open(os.path.join(path, CONFIG_FILE)) as f:
        arch = json.load(f).get("arch", "compiled")
    cls = CrossEncoderModel if arch == "cross" else CompiledModel
    return cls.from_pretrained(path, device=device)
