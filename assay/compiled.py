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
from assay.tiers import CONFIG_FILES, WEIGHTS_FILES

CONFIG_FILE = CONFIG_FILES["encoder"]
WEIGHTS_FILE = WEIGHTS_FILES["encoder"]


def option_texts(q: Question, with_instructions: bool = False) -> list[str]:
    """One text per option in canonical key order. The compiled tiers encode option content
    only (the instruction goes into the queries, so option vectors stay distinct); the
    cross-encoder puts the instruction in front because its pair text is all it sees."""
    prefix = f"{q.instructions.strip()} " if with_instructions else ""
    if q.type == "bool":
        yes = q.yes.strip() if q.yes else "the answer is yes"
        no = q.no.strip() if q.no else "the answer is no"
        return [f"{prefix}Yes: {yes}", f"{prefix}No: {no}"]
    if q.type == "choice":
        out = []
        for key, desc in q.options.items():
            shown = key.replace("_", " ")
            out.append(f"{prefix}{shown}" + (f": {desc}" if desc else ""))
        return out
    return [f"{prefix}Level {i + 1} of {len(q.levels)}: {lvl}" for i, lvl in enumerate(q.levels)]


@dataclasses.dataclass
class Options:
    """Compiled options of a batch of questions, padded to K_max."""

    vectors: torch.Tensor  # (B, K, d) mean-pooled option encodings
    valid: torch.Tensor  # (B, K) bool
    counts: list[int]  # options per question
    tokens: torch.Tensor | None = None  # (sum K, T, d) token encodings, for late interaction
    token_mask: torch.Tensor | None = None  # (sum K, T)


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)


def last_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The last real token's state: the summary position of a causal (decoder) backbone."""
    last = mask.sum(1).clamp(min=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


POOLING = {"mean": mean_pool, "last": last_pool}


def truncate_layers(encoder: nn.Module, layers: int) -> None:
    """Keep the first `layers` transformer blocks; mid-depth features are cheaper and, for a
    decoder backbone, less specialised to next-token prediction."""
    encoder.layers = encoder.layers[:layers]
    encoder.config.num_hidden_layers = layers
    if getattr(encoder.config, "layer_types", None):
        encoder.config.layer_types = list(encoder.config.layer_types[:layers])


class CompiledModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        tokenizer,
        encoder_id: str,
        slots: int = 4,
        heads: int = 8,
        temperature: float = 1.0,
        late_interaction: bool = False,
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
        # late interaction (ColBERT-style): each option token finds its best-matching state
        # token; the mean of those cosine similarities, scaled, is added to the option logit
        self.late_scale = nn.Parameter(torch.tensor(10.0)) if late_interaction else None
        self.temperature = temperature
        self.max_state_tokens = 512
        self.max_option_tokens = 96
        self.hybrid = False  # for evaluate.predict compatibility
        # models saved before 2026-09-22 were trained with the instruction inside every option
        self.option_instructions = False
        self.pooling = "mean"
        self.keeps_option_tokens = False  # readers that need option tokens set this

    # --- construction ---------------------------------------------------------------------

    @classmethod
    def from_encoder(
        cls,
        encoder_id: str,
        device: str = "cuda",
        encoder_layers: int | None = None,
        pooling: str = "mean",
        lora_r: int | None = None,
        **kw,
    ) -> CompiledModel:
        """encoder_layers: keep only the first N backbone blocks. pooling: "mean" for bidirectional encoders,
        "last" for causal backbones. lora_r: train the backbone through a LoRA adapter (merged
        into the weights on save) instead of fully."""
        tokenizer = AutoTokenizer.from_pretrained(encoder_id)
        dtype = torch.bfloat16 if lora_r else torch.float32
        encoder = AutoModel.from_pretrained(encoder_id, dtype=dtype)
        if encoder_layers is not None:
            truncate_layers(encoder, encoder_layers)
        if lora_r:
            from peft import LoraConfig, get_peft_model

            config = LoraConfig(
                r=lora_r,
                lora_alpha=2 * lora_r,
                lora_dropout=0.05,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                bias="none",
            )
            encoder = get_peft_model(encoder, config)
        model = cls(encoder, tokenizer, encoder_id, **kw)
        model.pooling = pooling
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
        extra = {"reader_layers": config["reader_layers"]} if config.get("reader_layers") else {}
        model = cls(
            encoder,
            tokenizer,
            config["encoder_id"],
            slots=config["slots"],
            heads=config["heads"],
            temperature=config.get("temperature", 1.0),
            late_interaction=config.get("late_interaction", False),
            **extra,
        )
        model.option_instructions = config.get("option_instructions", True)
        model.pooling = config.get("pooling", "mean")
        state = safetensors.torch.load_file(os.path.join(path, WEIGHTS_FILE))
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [m for m in missing if not m.startswith("encoder.")]
        if missing or unexpected:
            raise ValueError(f"weights mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
        model.eval()
        return model.to(device)

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        if hasattr(self.encoder, "merge_and_unload"):
            self.encoder = self.encoder.merge_and_unload()
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
                    "option_instructions": self.option_instructions,
                    "pooling": self.pooling,
                    "late_interaction": self.late_scale is not None,
                    "arch": arch_name(self),
                    "reader_layers": len(self.reader_layers)
                    if hasattr(self, "reader_layers")
                    else None,
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
        )
        ids, mask = batch["input_ids"], batch["attention_mask"]
        if ids.shape[1] == 0:  # every text empty: give the batch one position to attend to
            ids = torch.full((len(texts), 1), self.tokenizer.pad_token_id, dtype=ids.dtype)
            mask = torch.zeros((len(texts), 1), dtype=mask.dtype)
        # a text that tokenizes to nothing (tokenizers without special tokens) would leave a
        # row with every key masked and a NaN softmax; let it attend to one pad token instead
        mask[mask.sum(1) == 0, 0] = 1
        batch = {"input_ids": ids.to(self.device), "attention_mask": mask.to(self.device)}
        hidden = self.encoder(**batch).last_hidden_state
        # the heads are fp32; a bf16 backbone (LoRA case) hands over fp32 features
        return hidden.to(self.slots.dtype), batch["attention_mask"]

    def pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return POOLING[self.pooling](hidden, mask)

    def encode_unique(
        self, texts: list[str], max_tokens: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Token encodings (N, T, d) and mask (N, T) for texts, encoding each distinct text
        once."""
        unique = list(dict.fromkeys(texts))
        index = {t: i for i, t in enumerate(unique)}
        hidden, mask = self.encode_tokens(unique, max_tokens or self.max_option_tokens)
        gather = torch.tensor([index[t] for t in texts], device=hidden.device)
        return hidden[gather], mask[gather]

    def encode_states(self, states: list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_tokens([render_state(s) for s in states], self.max_state_tokens)

    def compile_batch(
        self, questions: list[Question]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns queries (B, M, d), option vectors (B, K_max, d) and a validity mask (B, K_max).
        Each distinct text is encoded once per batch: records that share a question share
        its compiled parameters."""
        options = self.compile_options(questions)
        q_vec = self.pool(*self.encode_unique([q.instructions.strip() for q in questions]))
        queries = self.slots.unsqueeze(0) + self.query_proj(q_vec).unsqueeze(1)  # (B, M, d)
        return queries, options

    def compile_options(self, questions: list[Question]) -> Options:
        flat: list[str] = []
        counts: list[int] = []
        for q in questions:
            texts = option_texts(q, self.option_instructions)
            flat.extend(texts)
            counts.append(len(texts))
        tokens, token_mask = self.encode_unique(flat)  # (sum K, T, d), (sum K, T)
        opt_vec = self.pool(tokens, token_mask)  # (sum K, d)
        b, k_max = len(questions), max(counts)
        vectors = opt_vec.new_zeros((b, k_max, self.d))
        valid = torch.zeros((b, k_max), dtype=torch.bool, device=opt_vec.device)
        start = 0
        for i, k in enumerate(counts):
            vectors[i, :k] = opt_vec[start : start + k]
            valid[i, :k] = True
            start += k
        if self.late_scale is None and not self.keeps_option_tokens:
            return Options(vectors, valid, counts)
        return Options(vectors, valid, counts, tokens, token_mask)

    def late_interaction(
        self, hidden: torch.Tensor, mask: torch.Tensor, options: Options
    ) -> torch.Tensor:
        """(B, K) mean over option tokens of the best cosine match among state tokens. Works
        on each question's own options, so memory follows the number of options in the
        batch rather than batch size times the widest question."""
        h = F.normalize(hidden, dim=-1)
        o = F.normalize(options.tokens, dim=-1).to(h.dtype)
        out = h.new_zeros(options.valid.shape)
        start = 0
        for i, k in enumerate(options.counts):
            sim = torch.einsum("ktd,ld->ktl", o[start : start + k], h[i])  # (k, T, L)
            sim = sim.masked_fill(~mask[i].bool()[None, None, :], -1.0)
            best = sim.max(-1).values  # (k, T)
            tm = options.token_mask[start : start + k].to(best.dtype)
            out[i, :k] = (best * tm).sum(-1) / tm.sum(-1).clamp(min=1.0)
            start += k
        return out

    # --- decision --------------------------------------------------------------------------

    def decide(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
        queries: torch.Tensor,
        options: Options,
    ):
        """hidden (B, L, d) state tokens; queries (B, M, d) -> logits (B, K), evidence (B,)."""
        context, weights = self.attn(
            queries,
            hidden,
            hidden,
            key_padding_mask=~mask.bool(),
            need_weights=True,
            average_attn_weights=True,
        )
        z = self.reader(context.flatten(1))  # (B, d)
        logits = torch.einsum("bd,bkd->bk", self.bilinear(z), options.vectors) / (
            self.d**0.5
        ) + self.option_bias(options.vectors).squeeze(-1)
        if self.late_scale is not None:
            logits = logits + self.late_scale * self.late_interaction(hidden, mask, options)
        logits = logits.masked_fill(~options.valid, float("-inf"))
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
        queries, options = self.compile_batch(questions)
        return self.decide(hidden, mask, queries, options)

    @torch.no_grad()
    def answer(self, state: Any, questions: dict[str, Question]) -> dict[str, Answer]:
        names = list(questions)
        qs = [questions[n] for n in names]
        hidden, mask = self.encode_states([state])
        queries, options = self.compile_batch(qs)
        hidden = hidden.expand(len(qs), -1, -1)
        mask = mask.expand(len(qs), -1)
        logits, evidence = self.decide(hidden, mask, queries, options)
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
        s = F.normalize(self.pool(hidden, mask), dim=-1)
        _, options = self.compile_batch(questions)
        o = F.normalize(options.vectors, dim=-1)
        logits = torch.einsum("bd,bkd->bk", s, o) * scale
        return logits.masked_fill(~options.valid, float("-inf"))


class CrossEncoderModel(CompiledModel):
    """Comparison tier: one encoder pass per (state, option) pair, scored by a linear head.
    Joint attention between state and option, at K passes per question."""

    def __init__(self, encoder: nn.Module, tokenizer, encoder_id: str, **kw):
        kw.pop("late_interaction", None)  # the joint pass already relates option and state
        super().__init__(encoder, tokenizer, encoder_id, **kw)
        self.pair_head = nn.Linear(self.d, 1)
        self.pair_evidence = nn.Linear(self.d, 1)
        self.max_pair_tokens = 640

    def forward(self, states: list[Any], questions: list[Question]):
        state_texts = [render_state(s) for s in states]
        pairs: list[str] = []
        counts: list[int] = []
        for text, q in zip(state_texts, questions):
            opts = option_texts(q, with_instructions=True)
            pairs.extend(f"{o}{self.tokenizer.sep_token or ' | '}{text}" for o in opts)
            counts.append(len(opts))
        hidden, mask = self.encode_tokens(pairs, self.max_pair_tokens)
        pooled = self.pool(hidden, mask)
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


class ConditionedModel(CompiledModel):
    """Middle tier: the state is encoded once per question with the instruction in front, so
    the encoder's own attention relates question and state; the reader's slots then read the
    joint sequence, and options are compiled separately and scored by content. One encoder
    pass per (state, question) rather than per state (compiled) or per option (cross)."""

    def forward(self, states: list[Any], questions: list[Question]):
        sep = self.tokenizer.sep_token or " | "
        texts = [
            f"{q.instructions.strip()}{sep}{render_state(s)}" for s, q in zip(states, questions)
        ]
        hidden, mask = self.encode_tokens(texts, self.max_state_tokens + self.max_option_tokens)
        options = self.compile_options(questions)
        queries = self.slots.unsqueeze(0).expand(len(questions), -1, -1)
        return self.decide(hidden, mask, queries, options)

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


class JointReaderModel(CompiledModel):
    """Compiled tier with a deeper reader: the instruction tokens and every option's tokens form
    one sequence that self-attends (options compare with each other and with the instruction)
    and cross-attends into the cached state tokens, over a few Transformer layers. The state
    is still encoded once and never updated; each option is scored from its own tokens after
    the reader, plus the late-interaction term. One reader pass per question with all its
    options together, so the cost is one encoder pass per state and a small Transformer per
    question, never a pass per option."""

    def __init__(
        self, encoder: nn.Module, tokenizer, encoder_id: str, reader_layers: int = 3, **kw
    ):
        kw.setdefault("late_interaction", True)
        super().__init__(encoder, tokenizer, encoder_id, **kw)
        self.reader_layers = nn.ModuleList(
            nn.TransformerDecoderLayer(
                self.d,
                self.attn.num_heads,
                2 * self.d,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(reader_layers)
        )
        for layer in self.reader_layers:  # each block starts as the identity on its input
            nn.init.zeros_(layer.multihead_attn.out_proj.weight)
            nn.init.zeros_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.linear2.weight)
        self.option_head = nn.Linear(self.d, 1)
        self.joint_evidence = nn.Linear(self.d, 1)
        self.max_instruction_tokens = 48
        self.max_option_tokens = 32
        self.keeps_option_tokens = True

    def read(
        self, hidden: torch.Tensor, mask: torch.Tensor, options: Options, ins_tokens, ins_mask
    ):
        """Per question: run the reader over [instruction tokens; option tokens] against the
        state tokens; returns logits (B, K_max) and evidence (B,)."""
        b = hidden.shape[0]
        logits = hidden.new_full(options.valid.shape, float("-inf"))
        evidence = hidden.new_zeros(b)
        state_pad = ~mask.bool()
        start = 0
        for i, k in enumerate(options.counts):
            opt = options.tokens[start : start + k]  # (k, T, d)
            opt_mask = options.token_mask[start : start + k].bool()  # (k, T)
            t_q = int(ins_mask[i].sum())
            seq = torch.cat([ins_tokens[i, :t_q], opt.flatten(0, 1)], dim=0).unsqueeze(
                0
            )  # (1, L, d)
            pad = torch.cat([ins_mask.new_zeros(t_q, dtype=torch.bool), ~opt_mask.flatten()])[None]
            x = seq
            for layer in self.reader_layers:
                x = layer(
                    x,
                    hidden[i : i + 1],
                    tgt_key_padding_mask=pad,
                    memory_key_padding_mask=state_pad[i : i + 1],
                )
            x = x[0]
            q_vec = x[:t_q].mean(0)
            o = x[t_q:].view(k, -1, self.d)
            om = opt_mask.to(o.dtype).unsqueeze(-1)
            pooled = (o * om).sum(1) / om.sum(1).clamp(min=1.0)  # (k, d)
            logits[i, :k] = self.option_head(pooled).squeeze(-1)
            evidence[i] = self.joint_evidence(q_vec).squeeze(-1)
            start += k
        if self.late_scale is not None:
            logits = logits + self.late_scale * self.late_interaction(hidden, mask, options)
        return logits.masked_fill(~options.valid, float("-inf")), evidence

    def forward(self, states: list[Any], questions: list[Question]):
        hidden, mask = self.encode_states(states)
        options = self.compile_options(questions)
        ins_tokens, ins_mask = self.encode_unique(
            [q.instructions.strip() for q in questions], max_tokens=self.max_instruction_tokens
        )
        return self.read(hidden, mask, options, ins_tokens, ins_mask)

    @torch.no_grad()
    def answer(self, state: Any, questions: dict[str, Question]) -> dict[str, Answer]:
        names = list(questions)
        qs = [questions[n] for n in names]
        hidden, mask = self.encode_states([state])
        options = self.compile_options(qs)
        ins_tokens, ins_mask = self.encode_unique(
            [q.instructions.strip() for q in qs], max_tokens=self.max_instruction_tokens
        )
        logits, evidence = self.read(
            hidden.expand(len(qs), -1, -1), mask.expand(len(qs), -1), options, ins_tokens, ins_mask
        )
        probs = torch.softmax(logits / self.temperature, dim=-1)
        ev = torch.sigmoid(evidence)
        return {
            n: make_answer(q, probs[i, : len(q.keys)].tolist(), ev[i].item())
            for i, (n, q) in enumerate(zip(names, qs))
        }


ARCHITECTURES: dict[str, type[CompiledModel]] = {
    "compiled": CompiledModel,
    "conditioned": ConditionedModel,
    "cross": CrossEncoderModel,
    "joint": JointReaderModel,
}


def arch_name(model: CompiledModel) -> str:
    return next(name for name, cls in ARCHITECTURES.items() if type(model) is cls)


def load_any(path: str, device: str = "cuda") -> CompiledModel:
    """Load a model saved by any of the encoder-tier classes."""
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download

        path = snapshot_download(path)
    with open(os.path.join(path, CONFIG_FILE)) as f:
        arch = json.load(f).get("arch", "compiled")
    return ARCHITECTURES[arch].from_pretrained(path, device=device)
