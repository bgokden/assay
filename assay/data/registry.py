"""Task specifications: how a public dataset row becomes a state plus a typed question.

Each task provides a builder that turns one dataset row into an `Example` or None (skip).
Instructions are sampled from a few paraphrases so the model does not bind to one wording.
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Callable
from typing import Any

from assay.schema import Question


@dataclasses.dataclass
class Example:
    state: Any
    question: Question
    label: Any  # option key, bool or level index
    target: dict[str, float] | None = None  # soft distribution over keys
    answerable: bool = True
    name: str = "answer"


Builder = Callable[[dict[str, Any], random.Random], Example | None]


@dataclasses.dataclass
class TaskSpec:
    name: str
    hf_id: str
    config: str | None
    split: str
    build: Builder | None = None
    # synthetic tasks: generator(n, seed) -> examples, no dataset download
    generator: Callable[[int, int], list[Example]] | None = None
    max_train: int = 1500
    max_eval: int = 200
    eval_split: str | None = None
    holdout: bool = False
    revision: str | None = None
    data_dir: str | None = None
    # builders that need the whole table (for example to aggregate annotators) get it here
    prepare: Callable[[Any], list[dict[str, Any]]] | None = None
    # builders that need dataset features (for example ClassLabel names) are made from it
    build_factory: Callable[[Any], Builder] | None = None
    # fraction of reading-comprehension examples to turn into unanswerable pairs by
    # swapping in another row's passage
    swap_state_fraction: float = 0.0
    swap_state_key: str | None = None


def pick(rng: random.Random, options: list[str]) -> str:
    return options[rng.randrange(len(options))]


def choice_builder(
    instructions: list[str],
    options: dict[str, str | None] | Callable[[dict[str, Any]], dict[str, str | None] | None],
    state_fn: Callable[[dict[str, Any]], Any],
    label_fn: Callable[[dict[str, Any]], str | None],
    target_fn: Callable[[dict[str, Any]], dict[str, float] | None] | None = None,
    name: str = "answer",
) -> Builder:
    """`options` is a fixed mapping or a function of the row (per-example answer sets)."""

    def build(row: dict[str, Any], rng: random.Random) -> Example | None:
        label = label_fn(row)
        opts = options(row) if callable(options) else options
        if label is None or opts is None or label not in opts or len(opts) < 2:
            return None
        state = state_fn(row)
        if state is None:
            return None
        q = Question(type="choice", instructions=pick(rng, instructions), options=dict(opts))
        target = target_fn(row) if target_fn else None
        return Example(state=state, question=q, label=label, target=target, name=name)

    return build


def bool_builder(
    instructions: list[str],
    state_fn: Callable[[dict[str, Any]], Any],
    label_fn: Callable[[dict[str, Any]], bool | None],
    yes: str | None = None,
    no: str | None = None,
    p_yes_fn: Callable[[dict[str, Any]], float | None] | None = None,
    name: str = "answer",
) -> Builder:
    def build(row: dict[str, Any], rng: random.Random) -> Example | None:
        label = label_fn(row)
        if label is None:
            return None
        state = state_fn(row)
        if state is None:
            return None
        q = Question(type="bool", instructions=pick(rng, instructions), yes=yes, no=no)
        target = None
        if p_yes_fn is not None:
            p = p_yes_fn(row)
            if p is not None:
                p = min(max(float(p), 0.0), 1.0)
                target = {"yes": p, "no": 1.0 - p}
        return Example(state=state, question=q, label=bool(label), target=target, name=name)

    return build


def score_builder(
    instructions: list[str],
    levels: list[str],
    state_fn: Callable[[dict[str, Any]], Any],
    level_fn: Callable[[dict[str, Any]], int | None],
    value_fn: Callable[[dict[str, Any]], float | None] | None = None,
    counts_fn: Callable[[dict[str, Any]], list[float] | None] | None = None,
    name: str = "answer",
) -> Builder:
    """value_fn returns a continuous position on the level axis (0..K-1) for a soft target;
    counts_fn returns per-level annotator counts. Either produces a soft target; the
    training code turns hard levels into SORD targets itself."""

    def build(row: dict[str, Any], rng: random.Random) -> Example | None:
        level = level_fn(row)
        if level is None or not (0 <= level < len(levels)):
            return None
        state = state_fn(row)
        if state is None:
            return None
        q = Question(type="score", instructions=pick(rng, instructions), levels=list(levels))
        target = None
        if counts_fn is not None:
            counts = counts_fn(row)
            if counts is not None and sum(counts) > 0:
                total = float(sum(counts))
                target = {str(i): c / total for i, c in enumerate(counts)}
        elif value_fn is not None:
            v = value_fn(row)
            if v is not None:
                target = sord_target(float(v), len(levels))
        return Example(state=state, question=q, label=int(level), target=target, name=name)

    return build


def sord_target(center: float, num_levels: int, sigma: float = 0.4) -> dict[str, float]:
    """Soft ordinal target: probability decays with squared distance from `center` on the
    level axis (Diaz and Marques 2019). `center` may be fractional."""
    import math

    weights = [math.exp(-((k - center) ** 2) / (2 * sigma * sigma)) for k in range(num_levels)]
    total = sum(weights)
    return {str(k): w / total for k, w in enumerate(weights)}


def clip_text(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."
