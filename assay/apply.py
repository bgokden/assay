"""Answer the same questions over a file of states.

    python -m assay.apply --model runs/assay-0.6b --questions questions.json \\
        --states states.jsonl --out answers.jsonl

`questions.json` holds the question set in the request shape (`{"name": {"type": ..., ...}}`),
`--states` is one state per line -- a bare JSON value, or an object with a `state` field, which
lets an existing dataset file be scored as it is -- and each output line carries the state's
identifier, if it had one, and the answers. Any tier works, and when the model directory holds
conformal thresholds each answer also carries `act` and `set`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterator
from typing import Any

import torch

from assay.conformal import decorate, load_conformal
from assay.evaluate import load_model
from assay.schema import Question
from assay.serving import runner_for


def read_states(path: str, limit: int | None = None) -> Iterator[tuple[Any, Any]]:
    """Yields (state, id) pairs. A line that is an object with a `state` field is read as a
    record, so training and evaluation files can be scored without being rewritten."""
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if limit is not None and i >= limit:
                return
            value = json.loads(line)
            if isinstance(value, dict) and "state" in value:
                yield value["state"], (value.get("meta") or {}).get("id", i)
            else:
                yield value, i


def load_questions(path: str) -> dict[str, Question]:
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path} must hold a non-empty object of named questions")
    return {name: Question.from_dict(q) for name, q in raw.items()}


def apply_model(
    model,
    questions: dict[str, Question],
    states: Iterator[tuple[Any, Any]],
    batch_size: int = 8,
    max_state_tokens: int = 4096,
    conformal: dict | None = None,
) -> Iterator[dict[str, Any]]:
    """One record per state, batched the way the server batches concurrent requests."""
    runner = runner_for(model, max_state_tokens)
    names = list(questions)
    qs = [questions[n] for n in names]
    batch: list[tuple[Any, Any]] = []
    for state, identifier in states:
        batch.append((state, identifier))
        if len(batch) == batch_size:
            yield from _answer(runner, batch, names, qs, conformal)
            batch = []
    if batch:
        yield from _answer(runner, batch, names, qs, conformal)


def _answer(runner, batch, names, qs, conformal) -> Iterator[dict[str, Any]]:
    items = [runner.prepare(state, qs) for state, _ in batch]
    with torch.inference_mode():
        results = runner.answer_batch(items, [qs] * len(batch))
    for (state, identifier), answers in zip(batch, results):
        out = {n: a.to_dict(q) for n, a, q in zip(names, answers, qs)}
        if conformal is not None:
            for n, q in zip(names, qs):
                decorate(out[n], q, conformal)
        yield {"id": identifier, "state": state, "answers": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="a run directory, a Hub id, or base:<hf id>")
    ap.add_argument("--questions", required=True, help="JSON object of named questions")
    ap.add_argument("--states", required=True, help="one state per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8, help="states per forward pass")
    ap.add_argument("--max-state-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, help="score only the first N lines")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--progress-every", type=int, default=1000)
    args = ap.parse_args()

    questions = load_questions(args.questions)
    model = load_model(args.model, device=args.device)
    model.eval()
    conformal = load_conformal(args.model) if os.path.isdir(args.model) else None
    t0 = time.time()
    written = 0
    with open(args.out, "w") as out:
        for record in apply_model(
            model,
            questions,
            read_states(args.states, args.limit),
            batch_size=args.batch_size,
            max_state_tokens=args.max_state_tokens,
            conformal=conformal,
        ):
            out.write(json.dumps(record) + "\n")
            written += 1
            if args.progress_every and written % args.progress_every == 0:
                rate = written / max(1e-9, time.time() - t0)
                print(f"{written} states, {rate:.1f}/s", flush=True)
    print(f"wrote {written} states with {len(questions)} questions each to {args.out}")


if __name__ == "__main__":
    main()
