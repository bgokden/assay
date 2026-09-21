"""Post-hoc temperature scaling on the calibration split of the seen tasks.

    uv run python -m assay.calibrate --model runs/assay-1.7b --data data/v1/calibration.jsonl

Fits one temperature T minimising soft-target NLL, writes it into the model directory's
assay_config.json, and reports metrics before and after. The same T is applied unchanged to
unseen tasks at evaluation time, so it never sees holdout data.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from assay.evaluate import format_report, load_model, predict, report
from assay.metrics import Scored, write_scored
from assay.model import CONFIG_FILE
from assay.records import read_records

CALIBRATION_SCORED = "calibration.scored.jsonl"


def nll_at_temperature(scored: list[Scored], temperature: float) -> float:
    total = 0.0
    for s in scored:
        logits = np.log(np.maximum(s.probs, 1e-12)) / temperature
        logits -= logits.max()
        logp = logits - math.log(np.exp(logits).sum())
        total += -(s.target * logp).sum()
    return total / max(1, len(scored))


def fit_temperature(scored: list[Scored], lo: float = 0.25, hi: float = 5.0) -> float:
    """Golden-section search on log T; the NLL is unimodal in T for a fixed set of logits."""
    a, b = math.log(lo), math.log(hi)
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = nll_at_temperature(scored, math.exp(c))
    fd = nll_at_temperature(scored, math.exp(d))
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = nll_at_temperature(scored, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = nll_at_temperature(scored, math.exp(d))
    return math.exp((a + b) / 2.0)


def rescale(scored: list[Scored], temperature: float) -> list[Scored]:
    out = []
    for s in scored:
        logits = np.log(np.maximum(s.probs, 1e-12)) / temperature
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        out.append(
            Scored(
                probs=probs.tolist(),
                target=s.target.tolist(),
                label_index=s.label_index,
                qtype=s.qtype,
                source=s.source,
                answerable=s.answerable,
                evidence=s.evidence,
            )
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="saved model directory")
    ap.add_argument("--data", required=True, help="calibration.jsonl")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-state-tokens", type=int, default=2048)
    args = ap.parse_args()
    model = load_model(args.model)
    records = list(read_records(args.data))
    scored = predict(
        model,
        records,
        batch_size=args.batch_size,
        max_state_tokens=args.max_state_tokens,
        temperature=1.0,
    )
    write_scored(os.path.join(args.model, CALIBRATION_SCORED), scored)
    before = report(scored)
    temperature = fit_temperature(scored)
    after = report(rescale(scored, temperature))
    print(f"fitted temperature: {temperature:.4f}")
    print("before:", format_report(before).splitlines()[0])
    print("after: ", format_report(after).splitlines()[0])
    config_path = os.path.join(args.model, CONFIG_FILE)
    with open(config_path) as f:
        config = json.load(f)
    config["temperature"] = temperature
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(args.model, "calibration.json"), "w") as f:
        json.dump({"temperature": temperature, "before": before["overall"], "after": after["overall"]}, f, indent=2)


if __name__ == "__main__":
    main()
