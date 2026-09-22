"""Abstention with a finite-sample guarantee, fitted on the calibration split of the seen tasks.

    uv run python -m assay.conformal --model runs/assay-4b-v4 --alpha 0.1

Two thresholds per question type, fitted on the per-item calibration predictions that
`assay.calibrate` saves (temperature-scaled here with the fitted temperature):

- a prediction-set threshold (split conformal, least-ambiguous-set score 1 - p_k): the set
  {k : p_k >= 1 - q} contains the label with probability at least 1 - alpha on inputs
  distributed like the calibration split;
- an act threshold on the top probability: among calibration items with p_max >= t, the
  error rate is at most alpha at confidence 1 - delta (one-sided binomial tests over a grid
  of thresholds with a Bonferroni correction, so the family-wise error is delta).

Both are properties of the calibration distribution; the reports show how they transfer to
unseen tasks. The evidence head still says "the state does not say"; this layer says "the
model does not know".
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any

import numpy as np

from assay.calibrate import CALIBRATION_SCORED, rescale
from assay.evaluate import scored_path
from assay.metrics import Scored, read_scored
from assay.schema import Question
from assay.tiers import temperature_of

CONFORMAL_FILE = "conformal.json"
TYPES = ("bool", "choice", "score")
ACT_GRID = [round(0.5 + 0.01 * i, 2) for i in range(50)]  # 0.50 .. 0.99


def binomial_cdf(successes: int, n: int, p: float) -> float:
    """P(X <= successes) for X ~ Binomial(n, p), in log space."""
    if successes >= n:
        return 1.0
    total = 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    for k in range(successes + 1):
        total += math.exp(
            math.lgamma(n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)
            + k * log_p
            + (n - k) * log_q
        )
    return min(1.0, total)


def set_threshold(scored: list[Scored], alpha: float) -> float:
    """Conformal quantile of the nonconformity score 1 - p_label."""
    scores = np.sort(np.array([1.0 - s.probs[s.label_index] for s in scored]))
    n = len(scores)
    rank = min(n, math.ceil((n + 1) * (1.0 - alpha)))
    return float(scores[rank - 1])


def act_threshold(scored: list[Scored], alpha: float, delta: float) -> float | None:
    """Lowest threshold t on p_max, from a fixed grid, such that the error rate among items
    with p_max >= t is at most alpha at confidence 1 - delta. Each grid point is a one-sided
    binomial test at level delta / len(grid) (Bonferroni), so any choice among the passing
    points keeps the guarantee. Returns None when no point passes."""
    p_max = np.array([s.probs.max() for s in scored])
    wrong = np.array([int(s.probs.argmax()) != s.label_index for s in scored])
    level = delta / len(ACT_GRID)
    for t in ACT_GRID:
        mask = p_max >= t
        m = int(mask.sum())
        if m and binomial_cdf(int(wrong[mask].sum()), m, alpha) <= level:
            return t
    return None


def fit(scored: list[Scored], alpha: float, delta: float) -> dict[str, Any]:
    out: dict[str, Any] = {"alpha": alpha, "delta": delta, "types": {}}
    for qtype in TYPES:
        items = [s for s in scored if s.qtype == qtype and s.answerable]
        if not items:
            continue
        out["types"][qtype] = {
            "count": len(items),
            "set_threshold": set_threshold(items, alpha),
            "act_threshold": act_threshold(items, alpha, delta),
        }
    return out


def prediction_set(probs: list[float], qtype: str, thresholds: dict[str, Any]) -> list[int]:
    q = thresholds["types"][qtype]["set_threshold"]
    return [k for k, p in enumerate(probs) if 1.0 - p <= q]


def should_act(probs: list[float], qtype: str, thresholds: dict[str, Any]) -> bool:
    t = thresholds["types"][qtype]["act_threshold"]
    return t is not None and max(probs) >= t


def decorate(answer: dict[str, Any], question: Question, thresholds: dict[str, Any]) -> None:
    """Add "act" and "set" to a serialised answer (keys as the question names them)."""
    probs = [answer["probabilities"][k] for k in question.keys]
    if question.type not in thresholds["types"]:
        return
    answer["act"] = should_act(probs, question.type, thresholds)
    answer["set"] = [question.keys[k] for k in prediction_set(probs, question.type, thresholds)]


def evaluate(scored: list[Scored], thresholds: dict[str, Any]) -> dict[str, Any]:
    """Coverage and size of the sets, rate and error of acting, per question type."""
    out: dict[str, Any] = {}
    for qtype in TYPES:
        items = [s for s in scored if s.qtype == qtype and s.answerable]
        if not items or qtype not in thresholds["types"]:
            continue
        sets = [prediction_set(s.probs.tolist(), qtype, thresholds) for s in items]
        covered = np.array([s.label_index in ps for s, ps in zip(items, sets)])
        acts = np.array([should_act(s.probs.tolist(), qtype, thresholds) for s in items])
        correct = np.array([int(s.probs.argmax()) == s.label_index for s in items])
        out[qtype] = {
            "count": len(items),
            "coverage": float(covered.mean()),
            "mean_set_size": float(np.mean([len(ps) for ps in sets])),
            "singleton_rate": float(np.mean([len(ps) == 1 for ps in sets])),
            "act_rate": float(acts.mean()),
            "act_error": float(1.0 - correct[acts].mean()) if acts.any() else float("nan"),
            "abstain_accuracy": float(correct[~acts].mean()) if (~acts).any() else float("nan"),
        }
    return out


def format_evaluation(name: str, ev: dict[str, Any]) -> str:
    lines = [name]
    for qtype, v in ev.items():
        lines.append(
            f"  {qtype:<7} n={v['count']:<5} coverage={v['coverage']:.3f} set_size={v['mean_set_size']:.2f} "
            f"singleton={v['singleton_rate']:.3f} act={v['act_rate']:.3f} act_err={v['act_error']:.3f} "
            f"abstained_acc={v['abstain_accuracy']:.3f}"
        )
    return "\n".join(lines)


def load_conformal(model: str) -> dict[str, Any] | None:
    """Thresholds from a model directory or a Hub repository, when it has them."""
    if model.startswith("base:"):
        return None
    if not os.path.isdir(model):
        from huggingface_hub import snapshot_download

        model = snapshot_download(model)
    path = os.path.join(model, CONFORMAL_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model directory with calibration.scored.jsonl")
    ap.add_argument("--alpha", type=float, default=0.1, help="target error rate")
    ap.add_argument(
        "--delta", type=float, default=0.05, help="confidence level for the act threshold"
    )
    ap.add_argument(
        "--evaluate",
        nargs="*",
        default=["holdout", "transfer-v4"],
        help="eval names with saved predictions",
    )
    args = ap.parse_args()
    temperature = temperature_of(args.model)
    calibration = rescale(read_scored(os.path.join(args.model, CALIBRATION_SCORED)), temperature)
    thresholds = fit(calibration, args.alpha, args.delta)
    thresholds["temperature"] = temperature
    with open(os.path.join(args.model, CONFORMAL_FILE), "w") as f:
        json.dump(thresholds, f, indent=2)
    print(json.dumps(thresholds["types"], indent=2))
    print(format_evaluation("calibration", evaluate(calibration, thresholds)))
    results = {}
    for name in args.evaluate:
        path = scored_path(os.path.join(args.model, f"eval-{name}.json"))
        if not os.path.exists(path):
            print(f"{name}: no predictions at {path}")
            continue
        scored = rescale(read_scored(path), temperature)
        results[name] = evaluate(scored, thresholds)
        print(format_evaluation(name, results[name]))
    with open(os.path.join(args.model, "conformal-eval.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
