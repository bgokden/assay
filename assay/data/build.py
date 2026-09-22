"""Build the training, calibration, dev and holdout record files from the task registry.

    uv run python -m assay.data.build --out data/v1 [--tasks ag_news,boolq] [--smoke]

Per task: rows are shuffled with a fixed seed, `max_train` become training examples and
`max_eval` become evaluation examples (from `eval_split` when the dataset has one). Evaluation
examples of training tasks are split half into calibration.jsonl (temperature fitting) and half
into dev.jsonl; holdout tasks go entirely to holdout.jsonl and never to train.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import os
import random
from typing import Any

from datasets import load_dataset

from assay.data.registry import Example, TaskSpec
from assay.data.tasks import TASKS
from assay.records import write_record


def load_rows(spec: TaskSpec, split: str):
    kwargs = {}
    if spec.revision:
        kwargs["revision"] = spec.revision
    if spec.data_dir:
        kwargs["data_dir"] = spec.data_dir
    ds = load_dataset(spec.hf_id, spec.config, split=split, **kwargs)
    return ds


def build_examples(spec: TaskSpec, split: str, limit: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    if spec.generator is not None:
        return spec.generator(limit, seed)
    ds = load_rows(spec, split)
    build = spec.build_factory(ds) if spec.build_factory is not None else spec.build
    if spec.prepare is not None:
        rows = spec.prepare(ds)
    else:
        rows = ds
    n = len(rows)
    order = list(range(n))
    rng.shuffle(order)
    examples: list[Example] = []
    # keep the per-row states around for unanswerable swaps
    for i in order:
        ex = build(rows[i], rng)
        if ex is None:
            continue
        examples.append(ex)
        if len(examples) >= limit:
            break
    if spec.swap_state_fraction > 0 and len(examples) > 1 and spec.swap_state_key:
        examples = add_unanswerable(examples, spec, rng)
    return examples


def add_unanswerable(examples: list[Example], spec: TaskSpec, rng: random.Random) -> list[Example]:
    """Pair some questions with another example's passage; those become answerable=False."""
    n_swap = int(len(examples) * spec.swap_state_fraction)
    key = spec.swap_state_key
    out = list(examples)
    for _ in range(n_swap):
        a, b = rng.sample(range(len(examples)), 2)
        src, other = examples[a], examples[b]
        if not isinstance(src.state, dict) or not isinstance(other.state, dict):
            continue
        state = dict(src.state)
        state[key] = other.state[key]
        out.append(
            Example(
                state=state,
                question=src.question,
                label=src.label,
                target=None,
                answerable=False,
                name=src.name,
            )
        )
    return out


def write_examples(f, examples: list[Example], spec: TaskSpec, split: str) -> None:
    for i, ex in enumerate(examples):
        write_record(
            f,
            ex.state,
            {ex.name: (ex.question, ex.label, ex.target, ex.answerable)},
            {"source": spec.name, "split": split, "id": f"{spec.name}/{split}/{i}"},
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", help="comma separated task names (default all)")
    ap.add_argument("--smoke", action="store_true", help="20 train / 10 eval per task")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    selected = [t for t in TASKS if not args.tasks or t.name in args.tasks.split(",")]
    stats: dict[str, dict[str, int]] = collections.defaultdict(dict)
    with contextlib.ExitStack() as stack:
        files = {
            name: stack.enter_context(open(os.path.join(args.out, f"{name}.jsonl"), "w"))
            for name in ("train", "calibration", "dev", "holdout")
        }
        for spec in selected:
            build_task(spec, args, files, stats)
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump({"tasks": stats}, f, indent=2)
    totals = collections.Counter()
    for s in stats.values():
        totals.update(s)
    print("totals:", dict(totals))


def build_task(
    spec: TaskSpec, args, files: dict[str, Any], stats: dict[str, dict[str, int]]
) -> None:
    """Build one task's splits; a source that fails to load fails the build (rerun with
    --tasks for the rest)."""
    max_train = 20 if args.smoke else spec.max_train
    max_eval = 10 if args.smoke else spec.max_eval
    if spec.eval_split:
        train = [] if spec.holdout else build_examples(spec, spec.split, max_train, args.seed)
        evals = build_examples(spec, spec.eval_split, max_eval, args.seed + 1)
    else:
        budget = max_eval if spec.holdout else max_train + max_eval
        pool = build_examples(spec, spec.split, budget, args.seed)
        evals = pool[:max_eval]
        train = [] if spec.holdout else pool[max_eval:]
    if spec.holdout:
        write_examples(files["holdout"], evals, spec, "holdout")
        stats[spec.name]["holdout"] = len(evals)
    else:
        write_examples(files["train"], train, spec, "train")
        half = len(evals) // 2
        write_examples(files["calibration"], evals[:half], spec, "calibration")
        write_examples(files["dev"], evals[half:], spec, "dev")
        stats[spec.name]["train"] = len(train)
        stats[spec.name]["calibration"] = half
        stats[spec.name]["dev"] = len(evals) - half
    print(f"{spec.name:<28} {stats[spec.name]}", flush=True)


if __name__ == "__main__":
    main()
