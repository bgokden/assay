"""Train and evaluate the compiled-function model on the same record files as the decoder.

    uv run python -m assay.train_compiled --encoder Alibaba-NLP/gte-modernbert-base \\
        --data data/v4 --extra data/distill/generic.jsonl --out runs/compiled-gte-base

Loss: soft-target cross-entropy over options (answerable only) + evidence BCE, as for the
decoder. The whole encoder is fine-tuned (it is small), with a higher learning rate on the
reader and heads.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from assay.calibrate import CALIBRATION_SCORED, fit_temperature, rescale
from assay.compiled import CompiledModel, CrossEncoderModel
from assay.evaluate import format_report, report
from assay.metrics import Scored, write_scored
from assay.records import Record, read_records
from assay.train import target_vector


def flatten(records: list[Record]) -> list[Record]:
    return [
        Record(state=r.state, questions=[lq], meta=r.meta) for r in records for lq in r.questions
    ]


def epoch_order(seed: int, epoch: int, n: int) -> list[int]:
    """Deterministic shuffle per epoch, so a resumed run sees the same batches."""
    order = list(range(n))
    random.Random(seed * 1009 + epoch).shuffle(order)
    return order


def training_schedule(
    records: list[Record], epochs: float, batch_size: int, pair_budget: int, seed: int
) -> list[list[int]]:
    """Every batch of the run in order: whole epochs, each with its own shuffle, and a
    proportional slice of the last one when `epochs` is fractional. Resuming at step k means
    continuing from schedule[k]."""
    schedule: list[list[int]] = []
    whole = int(epochs)
    for epoch in range(whole):
        schedule += make_batches(
            records, epoch_order(seed, epoch, len(records)), batch_size, pair_budget
        )
    fraction = epochs - whole
    if fraction > 0:
        last = make_batches(
            records, epoch_order(seed, whole, len(records)), batch_size, pair_budget
        )
        schedule += last[: max(1, int(len(last) * fraction))]
    return schedule


def save_checkpoint(path: str, model, optimizer, scheduler, step: int) -> None:
    state = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
    }
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_checkpoint(path: str, model, optimizer, scheduler) -> int:
    if not os.path.exists(path):
        return 0
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return int(state["step"])


def build_targets(batch: list[Record], k_max: int, hard: bool, sigma: float):
    target = torch.zeros((len(batch), k_max))
    answerable = torch.zeros(len(batch))
    for i, r in enumerate(batch):
        lq = r.questions[0]
        t = target_vector(lq, hard, sigma)
        target[i, : len(t)] = torch.tensor(t)
        answerable[i] = float(lq.answerable)
    return target, answerable


def make_batches(
    records: list[Record], order: list[int], batch_size: int, pair_budget: int
) -> list[list[int]]:
    """Group indices in the given order so a batch has at most batch_size records and at most
    pair_budget options in total; a record larger than the budget forms a batch on its own."""
    batches: list[list[int]] = []
    current: list[int] = []
    pairs = 0
    for i in order:
        k = len(records[i].questions[0].question.keys)
        if current and (len(current) >= batch_size or pairs + k > pair_budget):
            batches.append(current)
            current, pairs = [], 0
        current.append(i)
        pairs += k
    if current:
        batches.append(current)
    return batches


@torch.no_grad()
def predict(
    model: CompiledModel,
    records: list[Record],
    batch_size: int = 32,
    zero_shot: bool = False,
    temperature: float | None = None,
    pair_budget: int = 512,
) -> list[Scored]:
    records = flatten(records)
    model.eval()
    temp = model.temperature if temperature is None else temperature
    order = sorted(range(len(records)), key=lambda i: len(json.dumps(records[i].state)))
    scored: list[Scored | None] = [None] * len(records)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for idx in make_batches(records, order, batch_size, pair_budget):
            batch = [records[i] for i in idx]
            qs = [r.questions[0].question for r in batch]
            states = [r.state for r in batch]
            if zero_shot:
                logits = model.zero_shot_logits(states, qs)
                evidence = torch.zeros(len(batch), device=logits.device)
            else:
                logits, evidence = model(states, qs)
            probs = torch.softmax(logits.float() / temp, dim=-1)
            ev = torch.sigmoid(evidence.float())
            for j, i in enumerate(idx):
                lq = batch[j].questions[0]
                k = len(lq.question.keys)
                scored[i] = Scored(
                    probs=probs[j, :k].tolist(),
                    target=lq.target,
                    label_index=lq.label_index,
                    qtype=lq.question.type,
                    source=lq.source,
                    answerable=lq.answerable,
                    evidence=ev[j].item(),
                )
    return scored  # type: ignore[return-value]


def evaluate_split(
    model: CompiledModel,
    path: str,
    name: str,
    out_dir: str,
    batch_size: int,
    temperature: float | None = None,
    zero_shot: bool = False,
    pair_budget: int = 512,
) -> dict:
    records = list(read_records(path))
    scored = predict(
        model,
        records,
        batch_size=batch_size,
        zero_shot=zero_shot,
        temperature=temperature,
        pair_budget=pair_budget,
    )
    rep = report(scored)
    with open(os.path.join(out_dir, f"eval-{name}.json"), "w") as f:
        json.dump(rep, f, indent=2)
    write_scored(os.path.join(out_dir, f"eval-{name}.scored.jsonl"), scored)
    o = rep["overall"]
    print(
        f"{name}: acc={o['accuracy']:.3f} brier={o['brier']:.3f} nll={o['nll']:.3f} ece={o['ece']:.3f} conf_err={o['confident_error_rate']:.3f}",
        flush=True,
    )
    with open(os.path.join(out_dir, f"eval-{name}.txt"), "w") as f:
        f.write(format_report(rep))
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--extra", nargs="*", default=[], help="additional train jsonl files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--evidence-weight", type=float, default=0.5)
    ap.add_argument("--score-sigma", type=float, default=0.5)
    ap.add_argument("--hard-targets", action="store_true")
    ap.add_argument("--max-state-tokens", type=int, default=512)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--zero-shot-only",
        action="store_true",
        help="evaluate the untrained cosine baseline and exit",
    )
    ap.add_argument("--arch", choices=["compiled", "cross"], default="compiled")
    ap.add_argument(
        "--pair-budget", type=int, help="max options per batch (default 256 compiled, 64 cross)"
    )
    ap.add_argument("--checkpoint-every", type=int, default=500)
    args = ap.parse_args()
    if args.pair_budget is None:
        args.pair_budget = 64 if args.arch == "cross" else 256

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    cls = CrossEncoderModel if args.arch == "cross" else CompiledModel
    model = cls.from_encoder(args.encoder, slots=args.slots)
    model.max_state_tokens = args.max_state_tokens
    if args.arch == "cross":
        model.encoder.gradient_checkpointing_enable()

    if args.zero_shot_only:
        for split in ("dev", "holdout"):
            evaluate_split(
                model,
                os.path.join(args.data, f"{split}.jsonl"),
                f"{split}-zeroshot",
                args.out,
                args.batch_size,
                temperature=1.0,
                zero_shot=True,
                pair_budget=args.pair_budget,
            )
        evaluate_split(
            model,
            "data/suites/kev-transfer-v4-dev.jsonl",
            "transfer-v4-zeroshot",
            args.out,
            args.batch_size,
            temperature=1.0,
            zero_shot=True,
            pair_budget=args.pair_budget,
        )
        return

    records = flatten(list(read_records(os.path.join(args.data, "train.jsonl"), limit=args.limit)))
    for extra in args.extra:
        records += flatten(list(read_records(extra)))
    print(f"train records: {len(records)}", flush=True)
    schedule = training_schedule(records, args.epochs, args.batch_size, args.pair_budget, args.seed)
    total = len(schedule)
    warmup = int(total * args.warmup)

    enc_params = list(model.encoder.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": args.lr, "weight_decay": 0.01},
            {"params": head_params, "lr": args.head_lr, "weight_decay": 0.0},
        ]
    )

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    checkpoint_path = os.path.join(args.out, "checkpoint.pt")
    step = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
    if step:
        print(f"resumed from step {step}", flush=True)
    t0 = time.time()
    running = {"loss": 0.0, "answer": 0.0, "evidence": 0.0, "n": 0}
    model.train()
    for idx in schedule[step:]:
        batch = [records[i] for i in idx]
        qs = [r.questions[0].question for r in batch]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, evidence = model([r.state for r in batch], qs)
        k_max = logits.shape[1]
        target, answerable = build_targets(batch, k_max, args.hard_targets, args.score_sigma)
        target = target.to(logits.device)
        answerable = answerable.to(logits.device)
        logp = F.log_softmax(logits.float(), dim=-1)
        ce = -(target * logp.masked_fill(target == 0, 0.0)).sum(-1)
        answer_loss = (ce * answerable).sum() / answerable.sum().clamp(min=1.0)
        evidence_loss = F.binary_cross_entropy_with_logits(evidence.float(), answerable)
        loss = answer_loss + args.evidence_weight * evidence_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        running["loss"] += loss.item()
        running["answer"] += answer_loss.item()
        running["evidence"] += evidence_loss.item()
        running["n"] += 1
        if step % args.log_every == 0:
            n = running["n"]
            entry = {
                "step": step,
                "loss": running["loss"] / n,
                "answer_loss": running["answer"] / n,
                "evidence_loss": running["evidence"] / n,
                "lr": scheduler.get_last_lr()[0],
                "elapsed": time.time() - t0,
            }
            print(json.dumps(entry), flush=True)
            with open(os.path.join(args.out, "train_log.jsonl"), "a") as log:
                log.write(json.dumps(entry) + "\n")
            running = {"loss": 0.0, "answer": 0.0, "evidence": 0.0, "n": 0}
        if args.checkpoint_every and step % args.checkpoint_every == 0:
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, step)
    model.save_pretrained(args.out)
    print(f"saved to {args.out} after {step} steps, {time.time() - t0:.0f}s", flush=True)

    # temperature on the calibration split, then the three evaluations raw and scaled
    cal = list(read_records(os.path.join(args.data, "calibration.jsonl")))
    scored = predict(
        model, cal, batch_size=args.batch_size, temperature=1.0, pair_budget=args.pair_budget
    )
    write_scored(os.path.join(args.out, CALIBRATION_SCORED), scored)
    temperature = fit_temperature(scored)
    model.temperature = temperature
    before, after = report(scored)["overall"], report(rescale(scored, temperature))["overall"]
    print(
        f"fitted temperature: {temperature:.4f} (calibration ece {before['ece']:.3f} -> {after['ece']:.3f})",
        flush=True,
    )
    model.save_pretrained(args.out)
    with open(os.path.join(args.out, "calibration.json"), "w") as f:
        json.dump({"temperature": temperature, "before": before, "after": after}, f, indent=2)
    for split, path in (
        ("dev", os.path.join(args.data, "dev.jsonl")),
        ("holdout", os.path.join(args.data, "holdout.jsonl")),
        ("transfer-v4", "data/suites/kev-transfer-v4-dev.jsonl"),
    ):
        evaluate_split(
            model,
            path,
            split,
            args.out,
            args.batch_size,
            temperature=1.0,
            pair_budget=args.pair_budget,
        )
        evaluate_split(
            model, path, f"{split}-scaled", args.out, args.batch_size, pair_budget=args.pair_budget
        )


if __name__ == "__main__":
    main()
