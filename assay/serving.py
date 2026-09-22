"""What the server needs from a model, for every tier.

The decoder tier answers a packed batch in one forward pass: every question of a request is
an isolated branch over one shared prefill. The encoder and encoder-decoder tiers answer a
list of (state, question) rows instead. Both are wrapped here, so the batcher and the
endpoints hold one interface rather than a branch on the model type.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

from assay.encoding import Packed, encode, identity_order
from assay.schema import Answer, Question, make_answer, render_state


@dataclasses.dataclass
class Direct:
    """One request for a tier that takes rows: the state and the size a batch budget sees."""

    state: Any
    tokens: int


class PackedRunner:
    def __init__(self, model, max_state_tokens: int = 4096) -> None:
        self.model = model
        self.max_state_tokens = max_state_tokens

    def prepare(self, state: Any, questions: list[Question]) -> Packed:
        return encode(
            self.model.tokenizer,
            self.model.alphabet,
            state,
            questions,
            orders=[identity_order(q) for q in questions],
            max_state_tokens=self.max_state_tokens,
        )

    def size(self, item: Packed) -> int:
        return len(item)

    def answer_batch(
        self, items: list[Packed], questions: list[list[Question]]
    ) -> list[list[Answer]]:
        return self.model.answer_packed(items, questions)


class DirectRunner:
    """One row per (state, question); the rows of a whole batch run in one forward pass."""

    def __init__(self, model, max_state_tokens: int = 4096) -> None:
        self.model = model
        self.max_state_tokens = max_state_tokens

    def prepare(self, state: Any, questions: list[Question]) -> Direct:
        ids = self.model.tokenizer(
            render_state(state), truncation=True, max_length=self.max_state_tokens
        )["input_ids"]
        return Direct(state, max(1, len(ids)) * max(1, len(questions)))

    def size(self, item: Direct) -> int:
        return item.tokens

    def answer_batch(
        self, items: list[Direct], questions: list[list[Question]]
    ) -> list[list[Answer]]:
        states: list[Any] = []
        flat: list[Question] = []
        for item, qs in zip(items, questions):
            states += [item.state] * len(qs)
            flat += qs
        logits, evidence = self.model(states, flat)
        probs = torch.softmax(logits.float() / self.model.temperature, dim=-1)
        ev = torch.sigmoid(evidence.float())
        out: list[list[Answer]] = []
        row = 0
        for qs in questions:
            answers = []
            for q in qs:
                answers.append(make_answer(q, probs[row, : len(q.keys)].tolist(), ev[row].item()))
                row += 1
            out.append(answers)
        return out


def runner_for(model, max_state_tokens: int = 4096):
    """The decoder tier packs; everything else answers rows."""
    if hasattr(model, "answer_packed"):
        return PackedRunner(model, max_state_tokens)
    return DirectRunner(model, max_state_tokens)
