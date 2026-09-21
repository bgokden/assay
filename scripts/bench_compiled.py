"""Latency of the compiled tier, split into its three parts: encode a state once, compile
questions once, decide (the small computation that runs per state x question).

    uv run python scripts/bench_compiled.py --model runs/compiled-gte-base --device cpu
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

sys.path.insert(0, "scripts")
from bench_latency import STATE, questions

from assay.compiled import load_any


def timed(fn, device: str, repeats: int) -> float:
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
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, help="CPU threads (default: torch's choice)")
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    model = load_any(args.model, device=args.device)
    model.eval()
    print(f"{args.model} on {args.device}, {torch.get_num_threads()} threads")
    print("questions  encode_state_ms  compile_ms  decide_ms  end_to_end_ms")
    with torch.inference_mode():
        for n in (1, 3, 6, 12, 24):
            qs = list(questions(n).values())
            encode = timed(lambda: model.encode_states([STATE]), args.device, args.repeats)
            compile_ = timed(lambda qs=qs: model.compile_batch(qs), args.device, args.repeats)
            hidden, mask = model.encode_states([STATE])
            queries, options, valid = model.compile_batch(qs)
            h = hidden.expand(n, -1, -1)
            m = mask.expand(n, -1)
            parts = (h, m, queries, options, valid)
            decide = timed(lambda parts=parts: model.decide(*parts), args.device, args.repeats)
            total = timed(
                lambda qs=qs: model.answer(STATE, {f"q{i}": q for i, q in enumerate(qs)}),
                args.device,
                args.repeats,
            )
            print(f"{n:>9}  {encode:>15.2f}  {compile_:>10.2f}  {decide:>9.3f}  {total:>13.2f}")


if __name__ == "__main__":
    main()
