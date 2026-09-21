"""Score a model on a record file: accuracy, Brier, NLL, ECE, confident errors, per source.

    uv run python -m assay.evaluate --model base:Qwen/Qwen3-1.7B-Base --data suite.jsonl
    uv run python -m assay.evaluate --model runs/assay-1.7b --data suite.jsonl --out res.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable

import torch

from assay.encoding import Packed, encode, identity_order
from assay.metrics import Scored, reliability_table, summarize, summarize_by
from assay.model import AssayModel
from assay.records import Record, read_records


def load_model(spec: str, dtype: torch.dtype = torch.bfloat16) -> AssayModel:
    if spec.startswith("base:"):
        return AssayModel.from_base(spec[len("base:") :], lora_r=None, dtype=dtype)
    return AssayModel.from_pretrained(spec, dtype=dtype)


def predict(
    model: AssayModel,
    records: Iterable[Record],
    batch_size: int = 16,
    max_state_tokens: int = 2048,
    temperature: float | None = None,
) -> list[Scored]:
    """Answer every question of every record; returns Scored items in record order."""
    if temperature is not None:
        model.temperature = temperature
    records = list(records)
    packed: list[Packed] = []
    for r in records:
        qs = [lq.question for lq in r.questions]
        packed.append(
            encode(
                model.tokenizer,
                model.alphabet,
                r.state,
                qs,
                orders=[identity_order(q) for q in qs],
                max_state_tokens=max_state_tokens,
            )
        )
    order = sorted(range(len(records)), key=lambda i: len(packed[i]))
    results: list[list] = [None] * len(records)
    model.eval()
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        answers = model.answer_packed(
            [packed[i] for i in idx], [[lq.question for lq in records[i].questions] for i in idx]
        )
        for i, a in zip(idx, answers):
            results[i] = a
    scored: list[Scored] = []
    for r, answers in zip(records, results):
        for lq, a in zip(r.questions, answers):
            scored.append(
                Scored(
                    probs=[a.probabilities[k] for k in lq.question.keys],
                    target=lq.target,
                    label_index=lq.label_index,
                    qtype=lq.question.type,
                    source=lq.source,
                    answerable=lq.answerable,
                    evidence=a.evidence,
                )
            )
    return scored


def report(scored: list[Scored]) -> dict:
    import numpy as np

    p_max = np.array([s.probs.max() for s in scored])
    correct = np.array([float(int(s.probs.argmax()) == s.label_index) for s in scored])
    return {
        "overall": summarize(scored),
        "by_type": summarize_by(scored, "qtype"),
        "by_source": summarize_by(scored, "source"),
        "reliability": reliability_table(p_max, correct),
    }


def format_report(rep: dict) -> str:
    lines = []
    o = rep["overall"]
    lines.append(
        f"overall n={o['count']} acc={o['accuracy']:.3f} brier={o['brier']:.3f} "
        f"nll={o['nll']:.3f} ece={o['ece']:.3f} conf={o['mean_confidence']:.3f} "
        f"conf_err={o['confident_error_rate']:.3f}"
        + (f" score_mae={o['score_mae']:.3f}" if "score_mae" in o else "")
        + (f" evidence_auc={o['evidence_auc']:.3f}" if "evidence_auc" in o else "")
    )
    for name, group in (("type", rep["by_type"]), ("source", rep["by_source"])):
        for k, v in group.items():
            lines.append(
                f"  {name}={k:<24} n={v['count']:<5} acc={v['accuracy']:.3f} "
                f"brier={v['brier']:.3f} ece={v['ece']:.3f} conf_err={v['confident_error_rate']:.3f}"
            )
    lines.append("  reliability (confidence -> accuracy):")
    for row in rep["reliability"]:
        if row["count"]:
            lines.append(
                f"    {row['bin']} n={row['count']:<5} conf={row['confidence']:.3f} acc={row['accuracy']:.3f}"
            )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base:<hf id> or a saved model directory")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--max-state-tokens", type=int, default=2048)
    args = ap.parse_args()
    model = load_model(args.model)
    records = list(read_records(args.data, limit=args.limit))
    t0 = time.time()
    scored = predict(
        model,
        records,
        batch_size=args.batch_size,
        max_state_tokens=args.max_state_tokens,
        temperature=args.temperature,
    )
    elapsed = time.time() - t0
    rep = report(scored)
    rep["seconds"] = elapsed
    rep["model"] = args.model
    rep["data"] = args.data
    print(format_report(rep))
    print(f"  {len(scored)} questions in {elapsed:.1f}s")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()
