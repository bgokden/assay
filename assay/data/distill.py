"""Distillation data: generic rubric questions over existing texts, labelled by a teacher model
with its temperature-scaled probabilities, plus exact-labelled synthetic sets.

    uv run python -m assay.data.distill --teacher runs/assay-27b --train data/v2/train.jsonl \\
        --out data/distill --generic 40000 --policy-hard 5000 --dates 6000

Writes generic.jsonl (teacher targets), policy_hard.jsonl and dates.jsonl (exact labels), each
with a small *_dev.jsonl slice, to be concatenated with an existing data version.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any

import torch

from assay.data.dates import generate as generate_dates
from assay.data.policy import generate_hard
from assay.data.registry import Example
from assay.data.rubrics import question_bank
from assay.encoding import encode, identity_order
from assay.records import write_record
from assay.schema import Question

GENERIC: list[Question] = question_bank()


def texts_from_train(
    path: str, rng: random.Random, min_chars: int = 40, max_chars: int = 1500
) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("meta", {}).get("source") in ("policy_application",):
                continue
            state = d["state"]
            if isinstance(state, dict):
                strings = [v for v in state.values() if isinstance(v, str)]
                if not strings:
                    continue
                text = max(strings, key=len)
            elif isinstance(state, str):
                text = state
            else:
                continue
            text = text.strip()
            if min_chars <= len(text) <= max_chars and text not in seen:
                seen.add(text)
                texts.append(text)
    rng.shuffle(texts)
    return texts


def example_from_probs(text: str, q: Question, p: list[float]) -> Example:
    keys = q.keys
    best = max(range(len(keys)), key=lambda i: p[i])
    label: Any = keys[best]
    if q.type == "bool":
        label = best == 0
    elif q.type == "score":
        label = best
    return Example(
        state=text,
        question=q,
        label=label,
        target={k: float(v) for k, v in zip(keys, p)},
        name="answer",
    )


@torch.no_grad()
def teacher_label(
    model,
    pairs: list[tuple[str, Question]],
    batch_size: int,
    max_state_tokens: int,
    out_path: str,
    meta: dict[str, Any],
) -> int:
    """Label pairs in length-sorted batches, appending each batch to out_path as it completes.
    Already written lines are skipped, so a crashed run resumes where it stopped."""
    done = 0
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = sum(1 for _ in f)
    packed = [
        encode(
            model.tokenizer,
            model.alphabet,
            text,
            [q],
            orders=[identity_order(q)],
            max_state_tokens=max_state_tokens,
        )
        for text, q in pairs
    ]
    order = sorted(range(len(pairs)), key=lambda i: len(packed[i]))
    model.eval()
    written = done
    with open(out_path, "a") as f, torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(done, len(order), batch_size):
            idx = order[start : start + batch_size]
            answers = model.answer_packed([packed[i] for i in idx], [[pairs[i][1]] for i in idx])
            for i, a in zip(idx, answers):
                text, q = pairs[i]
                ex = example_from_probs(text, q, [a[0].probabilities[k] for k in q.keys])
                write_record(
                    f,
                    ex.state,
                    {ex.name: (ex.question, ex.label, ex.target, ex.answerable)},
                    {**meta, "id": f"{meta['source']}/{i}"},
                )
                written += 1
            f.flush()
            if start % (batch_size * 200) == 0:
                print(f"labelled {start}/{len(pairs)}", flush=True)
    return written


@torch.no_grad()
def teacher_label_packed(
    model,
    states: list[Any],
    questions_per_state: list[list[Question]],
    batch_size: int,
    max_state_tokens: int,
    out_path: str,
    meta: dict[str, Any],
) -> int:
    """Label every question of a state in one packed forward pass (the state is encoded once
    per state, not once per question). One record per state, all its questions inside.
    Resumable: states already written are skipped."""
    done = 0
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = sum(1 for _ in f)
    model.eval()
    written = done
    with open(out_path, "a") as f, torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(done, len(states), batch_size):
            idx = list(range(start, min(start + batch_size, len(states))))
            packed = [
                encode(
                    model.tokenizer,
                    model.alphabet,
                    states[i],
                    questions_per_state[i],
                    orders=[identity_order(q) for q in questions_per_state[i]],
                    max_state_tokens=max_state_tokens,
                )
                for i in idx
            ]
            answers = model.answer_packed(packed, [questions_per_state[i] for i in idx])
            for i, per_state in zip(idx, answers):
                record: dict[str, Any] = {}
                for j, (q, a) in enumerate(zip(questions_per_state[i], per_state)):
                    ex = example_from_probs(states[i], q, [a.probabilities[k] for k in q.keys])
                    record[f"q{j}"] = (ex.question, ex.label, ex.target, ex.answerable)
                write_record(f, states[i], record, {**meta, "id": f"{meta['source']}/{i}"})
                written += 1
            f.flush()
            if start % (batch_size * 200) == 0:
                print(f"labelled {start}/{len(states)} states", flush=True)
    return written


def write_examples(
    path: str, examples: list[Example], source: str, extra_meta: dict[str, Any] | None = None
) -> None:
    with open(path, "w") as f:
        for i, ex in enumerate(examples):
            meta = {"source": source, "id": f"{source}/{i}"}
            if extra_meta:
                meta.update(extra_meta)
            write_record(
                f, ex.state, {ex.name: (ex.question, ex.label, ex.target, ex.answerable)}, meta
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--train", required=True, help="existing train.jsonl to take texts from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--generic", type=int, default=40000)
    ap.add_argument("--policy-hard", type=int, default=5000)
    ap.add_argument("--dates", type=int, default=6000)
    ap.add_argument("--dev", type=int, default=300, help="dev slice per synthetic set")
    ap.add_argument("--questions-per-text", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-state-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument(
        "--corpus", help="states file from assay.data.corpus: label these with the rubric bank"
    )
    ap.add_argument("--corpus-out", default="corpus_labels.jsonl", help="output name under --out")
    ap.add_argument(
        "--corpus-limit", type=int, help="label only the first N states (the file is shuffled)"
    )
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    if args.corpus:
        bank = GENERIC
        states: list[Any] = []
        with open(args.corpus) as f:
            for line in f:
                states.append(json.loads(line)["state"])
                if args.corpus_limit and len(states) >= args.corpus_limit:
                    break
        per_state = [rng.sample(bank, args.questions_per_text) for _ in states]
        print(
            f"{len(states)} states x {args.questions_per_text} questions from a bank of {len(bank)}; loading teacher",
            flush=True,
        )
        from assay.evaluate import load_model

        model = load_model(args.teacher)
        written = teacher_label_packed(
            model,
            states,
            per_state,
            args.batch_size,
            args.max_state_tokens,
            os.path.join(args.out, args.corpus_out),
            {
                "source": "distill_corpus",
                "teacher": args.teacher,
                "teacher_temperature": model.temperature,
            },
        )
        print(f"wrote {written} labelled states to {args.out}/{args.corpus_out}")
        return

    if not os.path.exists(os.path.join(args.out, "dates_dev.jsonl")):
        hard = generate_hard(args.policy_hard + args.dev, args.seed)
        write_examples(os.path.join(args.out, "policy_hard.jsonl"), hard[args.dev :], "policy_hard")
        write_examples(
            os.path.join(args.out, "policy_hard_dev.jsonl"), hard[: args.dev], "policy_hard"
        )
        dates = generate_dates(args.dates + args.dev, args.seed + 1)
        write_examples(os.path.join(args.out, "dates.jsonl"), dates[args.dev :], "dates")
        write_examples(os.path.join(args.out, "dates_dev.jsonl"), dates[: args.dev], "dates")
        print(f"wrote {len(hard)} policy_hard and {len(dates)} dates examples", flush=True)

    texts = texts_from_train(args.train, rng)
    pairs: list[tuple[str, Question]] = []
    for text in texts:
        for q in rng.sample(GENERIC, args.questions_per_text):
            pairs.append((text, q))
        if len(pairs) >= args.generic:
            break
    pairs = pairs[: args.generic]
    print(f"{len(texts)} texts, {len(pairs)} generic pairs; loading teacher", flush=True)

    from assay.evaluate import load_model

    model = load_model(args.teacher)
    written = teacher_label(
        model,
        pairs,
        args.batch_size,
        args.max_state_tokens,
        os.path.join(args.out, "generic.jsonl"),
        {
            "source": "distill_generic",
            "teacher": args.teacher,
            "teacher_temperature": model.temperature,
        },
    )
    print(f"wrote {written} teacher-labelled examples to {args.out}/generic.jsonl")


if __name__ == "__main__":
    main()
