"""Packing a state and its questions into one sequence with isolated question branches.

Layout of the packed sequence:

    [ state tokens ][ question 1 block ][ question 2 block ] ... [ question N block ]

The attention mask lets every token see the state (causally) and its own block (causally),
never another block. Position ids restart right after the state for every block, so each
question is processed exactly as if it were the only question following the state. The
decision token of a block is its last token; the answer distribution is read from the next-token
logits at that position, restricted to the option label tokens.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from sezgi.labels import LabelAlphabet
from sezgi.schema import Question, render_state

STATE_BLOCK = -1
PAD_BLOCK = -2


@dataclasses.dataclass
class EncodedQuestion:
    start: int
    end: int
    readout: int
    option_token_ids: list[int]


@dataclasses.dataclass
class Packed:
    input_ids: list[int]
    position_ids: list[int]
    block_ids: list[int]
    state_len: int
    questions: list[EncodedQuestion]

    def __len__(self) -> int:
        return len(self.input_ids)


def _display_key(key: str) -> str:
    return key.replace("_", " ")


def _option_line(label: str, key: str, description: str | None) -> str:
    if description is None or not str(description).strip():
        return f"{label}. {_display_key(key)}"
    description = str(description).strip()
    if len(key) <= 2:
        return f"{label}. {description}"
    return f"{label}. {_display_key(key)}: {description}"


def render_question(question: Question, alphabet: LabelAlphabet, order: list[int]) -> str:
    """Text of one question block. `order` lists canonical option indices in display order."""
    lines = [f"\nQuestion: {question.instructions.strip()}"]
    if question.type == "noul":
        if question.yes is not None:
            lines.append(f"yes: {question.yes.strip()}")
        if question.no is not None:
            lines.append(f"no: {question.no.strip()}")
        lines.append("Answer (yes or no):")
        return "\n".join(lines)
    if question.type == "choice":
        keys = list(question.options.keys())
        lines.append("Options:")
        for pos, idx in enumerate(order):
            key = keys[idx]
            lines.append(_option_line(alphabet.labels[pos], key, question.options[key]))
    else:
        lines.append("Levels (lowest to highest):")
        for pos, idx in enumerate(order):
            lines.append(f"{alphabet.labels[pos]}. {question.levels[idx].strip()}")
    lines.append("Answer:")
    return "\n".join(lines)


def identity_order(question: Question) -> list[int]:
    return list(range(len(question.keys)))


def encode(
    tokenizer: PreTrainedTokenizerBase,
    alphabet: LabelAlphabet,
    state: Any,
    questions: list[Question],
    orders: list[list[int]] | None = None,
    max_state_tokens: int = 2048,
) -> Packed:
    """Pack one state and its questions. `orders[i]` gives the display order of question i's
    options (canonical indices); options of the answer distribution stay in canonical order."""
    if orders is None:
        orders = [identity_order(q) for q in questions]
    state_text = render_state(state) + "\n"
    state_ids = tokenizer.encode(state_text, add_special_tokens=False)
    if len(state_ids) > max_state_tokens:
        state_ids = state_ids[:max_state_tokens]
    input_ids = list(state_ids)
    position_ids = list(range(len(state_ids)))
    block_ids = [STATE_BLOCK] * len(state_ids)
    state_len = len(state_ids)
    encoded: list[EncodedQuestion] = []
    for k, (question, order) in enumerate(zip(questions, orders)):
        if sorted(order) != list(range(len(question.keys))):
            raise ValueError("order must be a permutation of the question's options")
        block = tokenizer.encode(
            render_question(question, alphabet, order), add_special_tokens=False
        )
        start = len(input_ids)
        input_ids.extend(block)
        position_ids.extend(range(state_len, state_len + len(block)))
        block_ids.extend([k] * len(block))
        end = len(input_ids)
        if question.type == "noul":
            option_token_ids = [alphabet.yes_id, alphabet.no_id]
        else:
            # canonical index -> display position -> label token
            position_of = {idx: pos for pos, idx in enumerate(order)}
            option_token_ids = [
                alphabet.token_ids[position_of[idx]] for idx in range(len(question.keys))
            ]
        encoded.append(
            EncodedQuestion(start=start, end=end, readout=end - 1, option_token_ids=option_token_ids)
        )
    return Packed(
        input_ids=input_ids,
        position_ids=position_ids,
        block_ids=block_ids,
        state_len=state_len,
        questions=encoded,
    )


def build_attention_mask(block_ids: torch.Tensor) -> torch.Tensor:
    """(B, L) block ids -> (B, 1, L, L) boolean mask, True where query i may attend key j.

    Rules: causal; state tokens are visible to all later tokens; a block sees only itself and
    the state; padding is never visible; every token sees itself (keeps softmax finite)."""
    batch, length = block_ids.shape
    idx = torch.arange(length, device=block_ids.device)
    causal = idx[None, :] <= idx[:, None]
    key_is_state = (block_ids == STATE_BLOCK)[:, None, :]
    same_block = block_ids[:, :, None] == block_ids[:, None, :]
    key_is_pad = (block_ids == PAD_BLOCK)[:, None, :]
    allowed = causal[None] & (key_is_state | same_block) & ~key_is_pad
    allowed = allowed | torch.eye(length, dtype=torch.bool, device=block_ids.device)[None]
    return allowed[:, None, :, :]


@dataclasses.dataclass
class Batch:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    # flat over all questions in the batch
    q_batch_index: torch.Tensor
    q_readout: torch.Tensor
    q_option_ids: torch.Tensor  # (Q, K_max), padded with -1
    q_num_options: torch.Tensor
    num_questions: int

    def to(self, device: torch.device | str) -> "Batch":
        return Batch(
            input_ids=self.input_ids.to(device),
            position_ids=self.position_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            q_batch_index=self.q_batch_index.to(device),
            q_readout=self.q_readout.to(device),
            q_option_ids=self.q_option_ids.to(device),
            q_num_options=self.q_num_options.to(device),
            num_questions=self.num_questions,
        )


def collate(packed: list[Packed], pad_token_id: int) -> Batch:
    length = max(len(p) for p in packed)
    batch = len(packed)
    input_ids = torch.full((batch, length), pad_token_id, dtype=torch.long)
    position_ids = torch.zeros((batch, length), dtype=torch.long)
    block_ids = torch.full((batch, length), PAD_BLOCK, dtype=torch.long)
    q_batch_index, q_readout, q_option_ids, q_num = [], [], [], []
    for b, p in enumerate(packed):
        n = len(p)
        input_ids[b, :n] = torch.tensor(p.input_ids)
        position_ids[b, :n] = torch.tensor(p.position_ids)
        block_ids[b, :n] = torch.tensor(p.block_ids)
        for q in p.questions:
            q_batch_index.append(b)
            q_readout.append(q.readout)
            q_option_ids.append(q.option_token_ids)
            q_num.append(len(q.option_token_ids))
    k_max = max(q_num) if q_num else 1
    option_ids = torch.full((len(q_num), k_max), -1, dtype=torch.long)
    for i, ids in enumerate(q_option_ids):
        option_ids[i, : len(ids)] = torch.tensor(ids)
    return Batch(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=build_attention_mask(block_ids),
        q_batch_index=torch.tensor(q_batch_index, dtype=torch.long),
        q_readout=torch.tensor(q_readout, dtype=torch.long),
        q_option_ids=option_ids,
        q_num_options=torch.tensor(q_num, dtype=torch.long),
        num_questions=len(q_num),
    )
