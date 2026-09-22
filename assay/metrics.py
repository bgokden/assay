"""Accuracy and calibration metrics over a set of answered questions."""

from __future__ import annotations

import collections
import itertools
import json
import math
from collections.abc import Iterable
from typing import Any

import numpy as np


def brier(probs: np.ndarray, target: np.ndarray) -> float:
    """Multi-class Brier score, sum over options of (p - t)^2. Range 0..2."""
    return float(((probs - target) ** 2).sum())


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, bins: int = 15
) -> float:
    if len(confidences) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in itertools.pairwise(edges):
        mask = (confidences > lo) & (confidences <= hi)
        if lo == 0.0:
            mask |= confidences == 0.0
        if mask.any():
            ece += mask.sum() / n * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def reliability_table(
    confidences: np.ndarray, correct: np.ndarray, bins: int = 10
) -> list[dict[str, float]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for lo, hi in itertools.pairwise(edges):
        mask = (confidences > lo) & (confidences <= hi)
        if lo == 0.0:
            mask |= confidences == 0.0
        rows.append(
            {
                "bin": f"{lo:.1f}-{hi:.1f}",
                "count": int(mask.sum()),
                "confidence": float(confidences[mask].mean()) if mask.any() else float("nan"),
                "accuracy": float(correct[mask].mean()) if mask.any() else float("nan"),
            }
        )
    return rows


class Scored:
    """One answered question with everything the metrics need."""

    __slots__ = ("answerable", "evidence", "label_index", "probs", "qtype", "source", "target")

    def __init__(
        self,
        probs: list[float],
        target: list[float],
        label_index: int,
        qtype: str,
        source: str,
        answerable: bool,
        evidence: float,
    ) -> None:
        self.probs = np.asarray(probs, dtype=np.float64)
        self.target = np.asarray(target, dtype=np.float64)
        self.label_index = label_index
        self.qtype = qtype
        self.source = source
        self.answerable = answerable
        self.evidence = evidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "probs": [round(float(p), 6) for p in self.probs],
            "target": [round(float(t), 6) for t in self.target],
            "label_index": self.label_index,
            "qtype": self.qtype,
            "source": self.source,
            "answerable": self.answerable,
            "evidence": round(float(self.evidence), 6),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Scored:
        return cls(
            d["probs"],
            d["target"],
            d["label_index"],
            d["qtype"],
            d["source"],
            d["answerable"],
            d["evidence"],
        )


def write_scored(path: str, scored: Iterable[Scored]) -> None:
    with open(path, "w") as f:
        f.writelines(json.dumps(s.to_dict()) + "\n" for s in scored)


def read_scored(path: str) -> list[Scored]:
    with open(path) as f:
        return [Scored.from_dict(json.loads(line)) for line in f]


def summarize(items: Iterable[Scored], bins: int = 15) -> dict[str, Any]:
    items = list(items)
    if not items:
        return {"count": 0}
    p_max = np.array([s.probs.max() for s in items])
    pred = np.array([int(s.probs.argmax()) for s in items])
    label = np.array([s.label_index for s in items])
    correct = (pred == label).astype(np.float64)
    eps = 1e-12
    nll_hard = np.array([-math.log(max(s.probs[s.label_index], eps)) for s in items])
    nll_soft = np.array([-(s.target * np.log(np.maximum(s.probs, eps))).sum() for s in items])
    brier_hard = np.array([brier(s.probs, np.eye(len(s.probs))[s.label_index]) for s in items])
    brier_soft = np.array([brier(s.probs, s.target) for s in items])
    score_items = [s for s in items if s.qtype == "score"]
    out: dict[str, Any] = {
        "count": len(items),
        "accuracy": float(correct.mean()),
        "brier": float(brier_hard.mean()),
        "brier_soft": float(brier_soft.mean()),
        "nll": float(nll_hard.mean()),
        "nll_soft": float(nll_soft.mean()),
        "ece": expected_calibration_error(p_max, correct, bins),
        "mean_confidence": float(p_max.mean()),
        "confident_error_rate": float(((p_max >= 0.9) & (correct == 0)).mean()),
    }
    if score_items:
        levels = np.arange(max(len(s.probs) for s in score_items))
        mae = [
            abs(float((s.probs * levels[: len(s.probs)]).sum()) - s.label_index)
            for s in score_items
        ]
        out["score_mae"] = float(np.mean(mae))
    evid = np.array([s.evidence for s in items])
    ans = np.array([1.0 if s.answerable else 0.0 for s in items])
    if ans.min() < 1.0:
        out["evidence_auc"] = auc(evid, ans)
    return out


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC by rank statistic."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def summarize_by(items: Iterable[Scored], key: str, bins: int = 15) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Scored]] = collections.defaultdict(list)
    for s in items:
        groups[getattr(s, key)].append(s)
    return {k: summarize(v, bins) for k, v in sorted(groups.items())}
