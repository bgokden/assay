"""Latency of one request as the number of questions grows, packed against separate.

    uv run python scripts/bench_latency.py --model runs/assay-4b
    uv run python scripts/bench_latency.py --model runs/seq2seq-enc --device cpu

Any tier: the decoder packs every question into one prefill, the encoder and encoder-decoder
tiers run them as rows of one batch. Write the output to `latency.txt` (or `latency-cpu.txt`)
in the run directory and it appears in the family table.
"""

from __future__ import annotations

import argparse
import time

import torch

from assay.evaluate import load_model
from assay.schema import Question

STATE = (
    "Hi, I've been trying to connect my Stripe account for 3 days and the integration keeps "
    "failing. Support told me to reinstall the plugin, which I did twice. I'm losing sales every "
    "hour this stays broken and my subscription renews tomorrow. Please help ASAP."
)


def questions(n: int) -> dict[str, Question]:
    pool = [
        Question(type="bool", instructions="Does the message convey urgency?"),
        Question(
            type="choice",
            instructions="Which team should handle this?",
            options={
                "billing": "Payments and refunds",
                "technical": "Bugs and integrations",
                "sales": "Pricing",
            },
        ),
        Question(
            type="score",
            instructions="How frustrated is the customer?",
            levels=["Calm", "Frustrated but civil", "Very angry"],
        ),
        Question(type="bool", instructions="Has the customer already tried a fix?"),
        Question(type="bool", instructions="Does the customer mention money?"),
        Question(
            type="choice",
            instructions="What does the customer want most?",
            options={
                "fix": "The integration working",
                "refund": "Money back",
                "information": "An explanation",
            },
        ),
    ]
    return {f"q{i}": pool[i % len(pool)] for i in range(n)}


def timed(fn, device: str, repeats: int = 20) -> float:
    fn()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    model = load_model(args.model, device=args.device)
    model.eval()
    print("questions  packed_ms  separate_ms")
    for n in (1, 3, 6, 12, 24):
        qs = questions(n)
        with (
            torch.inference_mode(),
            torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device == "cuda"),
        ):
            packed = timed(lambda qs=qs: model.answer(STATE, qs), args.device)
            separate = timed(
                lambda qs=qs: [model.answer(STATE, {k: v}) for k, v in qs.items()],
                args.device,
                repeats=5,
            )
        print(f"{n:>9}  {packed:>9.1f}  {separate:>11.1f}")


if __name__ == "__main__":
    main()
