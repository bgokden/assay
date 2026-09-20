"""Question and answer types.

A request is a state (text, object or list) plus named typed questions. Every question is
answered with a probability distribution over a closed set of options; nothing is generated.
"""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any

MAX_OPTIONS = 255
MAX_LEVELS = 10


@dataclasses.dataclass(frozen=True)
class Question:
    """One typed question.

    type: "noul" (is the statement true), "choice" (pick one option) or "score" (ordered levels).
    instructions: the question in plain language, may reference state fields.
    options: choice only, ordered mapping option key -> description (description may be None).
    levels: score only, ordered level descriptions from lowest to highest.
    yes, no: noul only, optional descriptions of what makes the answer yes or no.
    """

    type: str
    instructions: str
    options: dict[str, str | None] | None = None
    levels: list[str] | None = None
    yes: str | None = None
    no: str | None = None

    def __post_init__(self) -> None:
        if self.type not in ("noul", "choice", "score"):
            raise ValueError(f"unknown question type {self.type!r}")
        if not self.instructions or not self.instructions.strip():
            raise ValueError("instructions must not be empty")
        if self.type == "choice":
            if not self.options or len(self.options) < 2:
                raise ValueError("choice needs at least two options")
            if len(self.options) > MAX_OPTIONS:
                raise ValueError(f"choice supports at most {MAX_OPTIONS} options")
            if self.levels is not None:
                raise ValueError("choice takes options, not levels")
        elif self.type == "score":
            if not self.levels or len(self.levels) < 2:
                raise ValueError("score needs at least two levels")
            if len(self.levels) > MAX_LEVELS:
                raise ValueError(f"score supports at most {MAX_LEVELS} levels")
            if self.options is not None:
                raise ValueError("score takes levels, not options")
        else:
            if self.options is not None or self.levels is not None:
                raise ValueError("noul takes neither options nor levels")
        if self.type != "noul" and (self.yes is not None or self.no is not None):
            raise ValueError("yes/no descriptions apply to noul only")

    @property
    def keys(self) -> list[str]:
        """Option identifiers in the order the answer distribution uses."""
        if self.type == "noul":
            return ["yes", "no"]
        if self.type == "choice":
            return list(self.options.keys())
        return [str(i) for i in range(len(self.levels))]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Question":
        options = d.get("options")
        if options is not None and isinstance(options, list):
            options = {str(k): None for k in options}
        return cls(
            type=d["type"],
            instructions=d["instructions"],
            options=options,
            levels=d.get("levels"),
            yes=d.get("yes"),
            no=d.get("no"),
        )


@dataclasses.dataclass(frozen=True)
class Answer:
    """The answer distribution for one question and the quantities derived from it.

    probabilities: option key -> probability, in the question's key order.
    confidence: (p_max - 1/K) / (1 - 1/K); 0 for a uniform distribution, 1 for a one-hot.
    evidence: P(the state contains what is needed to answer), from the evidence head.
    """

    type: str
    probabilities: dict[str, float]
    confidence: float
    evidence: float

    @property
    def argmax(self) -> str:
        return max(self.probabilities, key=self.probabilities.get)

    @property
    def noul(self) -> float:
        return self.probabilities["yes"]

    @property
    def score(self) -> float:
        return sum(int(k) * p for k, p in self.probabilities.items())

    def to_dict(self, question: Question) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type}
        if self.type == "noul":
            out["noul"] = round(self.noul, 4)
        elif self.type == "choice":
            out["choice"] = self.argmax
        else:
            out["score"] = round(self.score, 4)
            out["level"] = int(self.argmax)
            out["legend"] = {str(i): lvl for i, lvl in enumerate(question.levels)}
        out["probabilities"] = {k: round(p, 4) for k, p in self.probabilities.items()}
        out["confidence"] = round(self.confidence, 4)
        out["evidence"] = round(self.evidence, 4)
        return out


def confidence_from_probabilities(probs: list[float]) -> float:
    k = len(probs)
    if k < 2:
        return 1.0
    p_max = max(probs)
    return max(0.0, min(1.0, (p_max - 1.0 / k) / (1.0 - 1.0 / k)))


def make_answer(question: Question, probs: list[float], evidence: float) -> Answer:
    keys = question.keys
    if len(keys) != len(probs):
        raise ValueError("probability vector does not match the question's options")
    total = sum(probs)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("probabilities must be finite and positive")
    probs = [p / total for p in probs]
    return Answer(
        type=question.type,
        probabilities=dict(zip(keys, probs)),
        confidence=confidence_from_probabilities(probs),
        evidence=float(evidence),
    )


def render_state(state: Any) -> str:
    """Render the state as text the model reads. Strings pass through; objects become
    `key: value` lines with nested values as JSON; lists become numbered JSON items."""
    if isinstance(state, str):
        return state.strip()
    if isinstance(state, dict):
        lines = []
        for k, v in state.items():
            if isinstance(v, str):
                lines.append(f"{k}: {v.strip()}")
            else:
                lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
        return "\n".join(lines)
    if isinstance(state, list):
        lines = []
        for i, item in enumerate(state, start=1):
            if isinstance(item, str):
                lines.append(f"{i}. {item.strip()}")
            else:
                lines.append(f"{i}. {json.dumps(item, ensure_ascii=False)}")
        return "\n".join(lines)
    return json.dumps(state, ensure_ascii=False)
