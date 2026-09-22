"""Unlabelled states for distillation: texts drawn from the training tasks' source datasets
(never from holdout tasks or the transfer suite's sources), many more per source than the
labelled build takes.

    uv run python -m assay.data.corpus --out data/distill/corpus.jsonl --per-task 40000
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any

from assay.data.build import build_examples
from assay.data.tasks import TASKS
from assay.schema import render_state


def collect_texts(
    per_task: int, seed: int, min_chars: int = 40, max_chars: int = 1500
) -> list[dict[str, Any]]:
    """States from every non-holdout, non-synthetic task, deduplicated on their rendered text."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for spec in TASKS:
        if spec.holdout or spec.generator is not None:
            continue
        examples = build_examples(spec, spec.split, per_task, seed)
        kept = 0
        for ex in examples:
            text = render_state(ex.state).strip()
            if not (min_chars <= len(text) <= max_chars) or text in seen:
                continue
            seen.add(text)
            out.append({"state": ex.state, "source": spec.name})
            kept += 1
        print(f"{spec.name}: {kept} states", flush=True)
    random.Random(seed).shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-task", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    states = collect_texts(args.per_task, args.seed)
    with open(args.out, "w") as f:
        f.writelines(json.dumps(s) + "\n" for s in states)
    print(f"wrote {len(states)} states to {args.out}")


if __name__ == "__main__":
    main()
