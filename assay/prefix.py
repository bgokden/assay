"""Answer many questions from one encoding of the state, for backbones that cannot pack.

The decoder tier normally packs every question of a request into one sequence and isolates
them with a block mask, so the state is encoded once. A hybrid backbone cannot do that: its
linear-attention layers carry state along the sequence, so a mask cannot stop one question
from seeing another, and until now each question was answered in its own sequence -- the state
re-encoded once per question, which is where the 27B's cost went.

This is the other way to share the state. Run the state alone with the cache on, copy that
cache once per question, and run the question blocks as a batch that continues from it. Each
question sees the state and itself, which is exactly what the block mask gave us, and the
state is encoded once either way.

The encoding comes from `assay.encoding.encode`, split back into its prefix and its question
blocks, so both paths tokenise identically.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

from assay.encoding import Packed, encode, identity_order
from assay.schema import Answer, Question, make_answer

# Below this many questions the two extra passes cost more than re-encoding the state does.
# Measured on assay-27b (4-bit): one question 155 ms against 109 ms for a sequence per
# question, six questions a wash, twelve 342 ms against 458 ms, twenty-four 532 against 888.
PREFIX_MIN_QUESTIONS = 8


@dataclasses.dataclass
class Split:
    """One packed encoding taken apart: the shared prefix and one block per question."""

    state_ids: list[int]
    blocks: list[list[int]]
    readouts: list[int]  # index of the readout token inside its own block
    option_token_ids: list[list[int]]


def split_packed(packed: Packed) -> Split:
    state_ids = packed.input_ids[: packed.state_len]
    blocks, readouts, options = [], [], []
    for q in packed.questions:
        blocks.append(packed.input_ids[q.start : q.end])
        readouts.append(q.readout - q.start)
        options.append(q.option_token_ids)
    return Split(state_ids=state_ids, blocks=blocks, readouts=readouts, option_token_ids=options)


def _pad(blocks: list[list[int]], pad_token_id: int, device: torch.device):
    width = max(len(b) for b in blocks)
    ids = torch.full((len(blocks), width), pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(blocks), width), dtype=torch.long, device=device)
    for i, block in enumerate(blocks):
        ids[i, : len(block)] = torch.tensor(block, dtype=torch.long, device=device)
        mask[i, : len(block)] = 1
    return ids, mask


def expand_cache(cache, copies: int, device: torch.device) -> None:
    """Turn a cache built from one sequence into `copies` identical rows.

    `reorder_cache` is the beam-search hook: every layer type implements it, including the
    linear-attention layers a hybrid backbone uses, so selecting row zero `copies` times is a
    supported way to fan the state out.
    """
    index = torch.zeros(copies, dtype=torch.long, device=device)
    cache.reorder_cache(index)


@torch.no_grad()
def prefix_answers(
    model,
    state: Any,
    questions: list[Question],
    max_state_tokens: int = 2048,
) -> list[Answer]:
    """One encoding of the state, then every question answered from it."""
    if not questions:
        return []
    packed = encode(
        model.tokenizer,
        model.alphabet,
        state,
        questions,
        orders=[identity_order(q) for q in questions],
        max_state_tokens=max_state_tokens,
    )
    logits, evidence = prefix_logits(model, packed)
    probabilities = torch.softmax(logits / model.temperature, dim=-1)
    scores = torch.sigmoid(evidence)
    return [
        make_answer(q, probabilities[i, : len(q.keys)].tolist(), scores[i].item())
        for i, q in enumerate(questions)
    ]


@torch.no_grad()
def prefix_logits(model, packed: Packed) -> tuple[torch.Tensor, torch.Tensor]:
    """(option logits, evidence logits) for one packed request, via the prefix cache."""
    split = split_packed(packed)
    device = model.device
    backbone = model._backbone()
    pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id or 0

    state = torch.tensor([split.state_ids], dtype=torch.long, device=device)
    state_mask = torch.ones_like(state)
    prefix = backbone(
        input_ids=state,
        position_ids=torch.arange(state.shape[1], device=device).unsqueeze(0),
        attention_mask=_mask_for(model, causal_mask(state.shape[1], device), state_mask),
        use_cache=True,
    )
    cache = prefix.past_key_values
    count = len(split.blocks)
    expand_cache(cache, count, device)

    ids, block_mask = _pad(split.blocks, pad_id, device)
    positions = state.shape[1] + torch.arange(ids.shape[1], device=device).unsqueeze(0).expand(
        count, -1
    )
    flat_mask = torch.cat(
        [torch.ones(count, state.shape[1], dtype=torch.long, device=device), block_mask], dim=1
    )
    prepared = suffix_mask(block_mask, state.shape[1]) if model.hybrid else flat_mask
    hidden = backbone(
        input_ids=ids,
        position_ids=positions,
        attention_mask=_mask_for(model, prepared, block_mask) if model.hybrid else flat_mask,
        past_key_values=cache,
        use_cache=True,
    ).last_hidden_state

    rows = torch.arange(count, device=device)
    readouts = torch.tensor(split.readouts, dtype=torch.long, device=device)
    decision = hidden[rows, readouts]  # (Q, H)
    vocabulary = model._lm_head()(decision).float()
    width = max(len(o) for o in split.option_token_ids)
    option_ids = torch.full((count, width), -1, dtype=torch.long, device=vocabulary.device)
    for i, options in enumerate(split.option_token_ids):
        option_ids[i, : len(options)] = torch.tensor(options, device=vocabulary.device)
    valid = option_ids >= 0
    option_logits = vocabulary.gather(1, option_ids.clamp(min=0)).masked_fill(~valid, float("-inf"))

    head_input = decision.float().to(model.evidence_head.weight.device)
    if model.normalize_evidence_input:
        head_input = torch.nn.functional.layer_norm(head_input, head_input.shape[-1:])
    evidence = model.evidence_head(head_input).squeeze(-1).to(option_logits.device)
    return option_logits, evidence


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """(1, 1, L, L) boolean mask for a sequence attending to itself, causally."""
    idx = torch.arange(length, device=device)
    return (idx[None, :] <= idx[:, None])[None, None]


def suffix_mask(block_mask: torch.Tensor, state_len: int) -> torch.Tensor:
    """(Q, 1, B, S + B): every question block sees all of the state and, causally, itself.

    This is the block mask of the packed path written against a cached prefix: the state is
    the keys before the block, and padding is never visible. The diagonal is always allowed so
    a padded row still has something to attend to and its softmax stays finite.
    """
    count, width = block_mask.shape
    device = block_mask.device
    state = torch.ones(count, 1, width, state_len, dtype=torch.bool, device=device)
    within = causal_mask(width, device).expand(count, 1, width, width)
    real = block_mask.bool()[:, None, None, :].expand(count, 1, width, width)
    block = within & real
    block = block | torch.eye(width, dtype=torch.bool, device=device)[None, None]
    return torch.cat([state, block], dim=-1)


def _mask_for(model, prepared: torch.Tensor, padding: torch.Tensor):
    """Hybrid backbones want a mask per layer family: the attention layers take the prepared
    boolean mask, the linear-attention layers only the padding of the tokens being given."""
    if getattr(model, "hybrid", False):
        return {"full_attention": prepared, "linear_attention": padding}
    return padding
