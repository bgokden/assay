"""Train the LoRA adapter and evidence head with soft-target cross-entropy.

    uv run python -m assay.train --base Qwen/Qwen3-1.7B-Base --data data/v1 --out runs/assay-1.7b

Loss = sum over questions of CE(target, softmax(option logits)) [answerable only]
     + evidence_weight * BCE(evidence logit, answerable)
Targets: the record's soft target if present; SORD-smoothed levels for hard-labelled score
questions; one-hot otherwise. Choice options are shuffled per example so label tokens carry
no positional prior.
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

from assay.data.registry import sord_target
from assay.encoding import Packed, collate, encode, identity_order
from assay.evaluate import format_report, predict, report
from assay.model import AssayModel
from assay.records import Record, read_records


def target_vector(lq, hard_targets: bool, score_sigma: float) -> list[float]:
    keys = lq.question.keys
    if hard_targets or lq.target is None or max(lq.target) >= 0.999:
        if lq.question.type == "score" and not hard_targets:
            t = sord_target(float(lq.label_index), len(keys), sigma=score_sigma)
            return [t[k] for k in keys]
        one_hot = [0.0] * len(keys)
        one_hot[lq.label_index] = 1.0
        return one_hot
    return list(lq.target)


class TrainBatchBuilder:
    def __init__(self, model: AssayModel, max_state_tokens: int, hard_targets: bool, score_sigma: float, seed: int):
        self.model = model
        self.max_state_tokens = max_state_tokens
        self.hard_targets = hard_targets
        self.score_sigma = score_sigma
        self.rng = random.Random(seed)

    def encode_record(self, r: Record) -> tuple[Packed, list[list[float]], list[bool]]:
        qs = [lq.question for lq in r.questions]
        orders = []
        for q in qs:
            order = identity_order(q)
            if q.type == "choice":
                self.rng.shuffle(order)
            orders.append(order)
        packed = encode(
            self.model.tokenizer,
            self.model.alphabet,
            r.state,
            qs,
            orders=orders,
            max_state_tokens=self.max_state_tokens,
        )
        targets = [target_vector(lq, self.hard_targets, self.score_sigma) for lq in r.questions]
        answerable = [lq.answerable for lq in r.questions]
        return packed, targets, answerable

    def build(self, records: list[Record]):
        packed, targets, answerable = [], [], []
        for r in records:
            p, t, a = self.encode_record(r)
            packed.append(p)
            targets.extend(t)
            answerable.extend(a)
        batch = collate(packed, self.model.tokenizer.pad_token_id)
        k_max = batch.q_option_ids.shape[1]
        target = torch.zeros((len(targets), k_max), dtype=torch.float32)
        for i, t in enumerate(targets):
            target[i, : len(t)] = torch.tensor(t)
        return batch, target, torch.tensor(answerable, dtype=torch.float32)


def bucket_batches(records: list[Record], batch_size: int, rng: random.Random, length_key) -> list[list[Record]]:
    order = list(range(len(records)))
    rng.shuffle(order)
    chunk = batch_size * 16
    batches = []
    for start in range(0, len(order), chunk):
        idx = sorted(order[start : start + chunk], key=lambda i: length_key(records[i]))
        for b in range(0, len(idx), batch_size):
            batches.append([records[i] for i in idx[b : b + batch_size]])
    rng.shuffle(batches)
    return batches


def approx_length(r: Record) -> int:
    state = r.state if isinstance(r.state, str) else json.dumps(r.state)
    extra = sum(len(json.dumps(lq.question.options or lq.question.levels or "")) for lq in r.questions)
    return len(state) + extra


def compute_loss(out, target, answerable, evidence_weight: float):
    logp = F.log_softmax(out.option_logits, dim=-1)
    ce = -(target * logp.masked_fill(target == 0, 0.0)).sum(-1)
    n_ans = answerable.sum().clamp(min=1.0)
    answer_loss = (ce * answerable).sum() / n_ans
    evidence_loss = F.binary_cross_entropy_with_logits(out.evidence_logits, answerable)
    return answer_loss + evidence_weight * evidence_loss, answer_loss.detach(), evidence_loss.detach()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True, help="directory with train.jsonl and dev.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--evidence-weight", type=float, default=0.5)
    ap.add_argument("--score-sigma", type=float, default=0.5)
    ap.add_argument("--hard-targets", action="store_true", help="ablation: one-hot targets only")
    ap.add_argument("--max-state-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, help="use only the first N training records")
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gradient-checkpointing", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--resume", action="store_true", help="continue from <out>/checkpoint if present")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    model = AssayModel.from_base(args.base, lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    causal_lm = model._causal_lm()
    if not args.no_gradient_checkpointing:
        causal_lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        causal_lm.enable_input_require_grads()
    model.train()

    records = list(read_records(os.path.join(args.data, "train.jsonl"), limit=args.limit))
    dev = list(read_records(os.path.join(args.data, "dev.jsonl")))
    print(f"train records: {len(records)}  dev records: {len(dev)}")

    builder = TrainBatchBuilder(model, args.max_state_tokens, args.hard_targets, args.score_sigma, args.seed)
    steps_per_epoch = math.ceil(len(records) / args.batch_size)
    total_updates = math.ceil(steps_per_epoch * args.epochs / args.grad_accum)
    warmup_updates = int(total_updates * args.warmup)

    lora_params = [p for n, p in model.lm.named_parameters() if p.requires_grad]
    head_params = list(model.evidence_head.parameters())
    print(f"trainable: lora {sum(p.numel() for p in lora_params)/1e6:.1f}M, head {sum(p.numel() for p in head_params)}")
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": head_params, "lr": args.head_lr, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.98),
    )

    def lr_lambda(update: int) -> float:
        if update < warmup_updates:
            return (update + 1) / max(1, warmup_updates)
        progress = (update - warmup_updates) / max(1, total_updates - warmup_updates)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    log_path = os.path.join(args.out, "train_log.jsonl")
    log = open(log_path, "a")
    step = 0
    update = 0
    checkpoint_dir = os.path.join(args.out, "checkpoint")
    if args.resume and os.path.exists(os.path.join(checkpoint_dir, "state.pt")):
        step, update = load_checkpoint(checkpoint_dir, model, optimizer, scheduler)
        print(f"resumed from step {step} (update {update})")
    t0 = time.time()
    running = {"loss": 0.0, "answer": 0.0, "evidence": 0.0, "n": 0}
    epochs_int = math.ceil(args.epochs)
    max_steps = int(steps_per_epoch * args.epochs)
    done = False
    seen = 0
    for epoch in range(epochs_int):
        batches = bucket_batches(records, args.batch_size, rng, approx_length)
        for batch_records in batches:
            if seen < step:
                seen += 1
                continue
            if step >= max_steps:
                done = True
                break
            batch, target, answerable = builder.build(batch_records)
            batch = batch.to(model.device)
            target = target.to(model.device)
            answerable = answerable.to(model.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch)
            loss, answer_loss, evidence_loss = compute_loss(out, target, answerable, args.evidence_weight)
            (loss / args.grad_accum).backward()
            step += 1
            seen += 1
            running["loss"] += loss.item()
            running["answer"] += answer_loss.item()
            running["evidence"] += evidence_loss.item()
            running["n"] += 1
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(lora_params + head_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
            if step % args.log_every == 0:
                n = max(1, running["n"])
                entry = {
                    "step": step,
                    "update": update,
                    "epoch": epoch,
                    "loss": running["loss"] / n,
                    "answer_loss": running["answer"] / n,
                    "evidence_loss": running["evidence"] / n,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed": time.time() - t0,
                }
                print(json.dumps(entry), flush=True)
                log.write(json.dumps(entry) + "\n")
                log.flush()
                running = {"loss": 0.0, "answer": 0.0, "evidence": 0.0, "n": 0}
            if args.eval_every and step % args.eval_every == 0:
                evaluate_and_log(model, dev, log, step, args)
                model.train()
            if args.checkpoint_every and step % args.checkpoint_every == 0 and step % args.grad_accum == 0:
                save_checkpoint(checkpoint_dir, model, optimizer, scheduler, step, update)
        if done:
            break
    model.save_pretrained(args.out)
    print(f"saved to {args.out} after {step} steps, {time.time() - t0:.0f}s")
    evaluate_and_log(model, dev, log, step, args)
    log.close()


def trainable_state(model: AssayModel) -> dict[str, torch.Tensor]:
    state = {f"lm.{n}": p.detach().cpu() for n, p in model.lm.named_parameters() if p.requires_grad}
    state.update({f"evidence_head.{k}": v.detach().cpu() for k, v in model.evidence_head.state_dict().items()})
    return state


def save_checkpoint(path: str, model: AssayModel, optimizer, scheduler, step: int, update: int) -> None:
    os.makedirs(path, exist_ok=True)
    tmp = os.path.join(path, "state.pt.tmp")
    torch.save(
        {
            "params": trainable_state(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "update": update,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        },
        tmp,
    )
    os.replace(tmp, os.path.join(path, "state.pt"))


def load_checkpoint(path: str, model: AssayModel, optimizer, scheduler) -> tuple[int, int]:
    state = torch.load(os.path.join(path, "state.pt"), map_location="cpu", weights_only=False)
    params = dict(model.lm.named_parameters())
    head = model.evidence_head.state_dict()
    with torch.no_grad():
        for name, value in state["params"].items():
            if name.startswith("lm."):
                params[name[3:]].copy_(value)
            else:
                head[name[len("evidence_head.") :]].copy_(value)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])
    return state["step"], state["update"]


def evaluate_and_log(model: AssayModel, dev: list[Record], log, step: int, args) -> None:
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        scored = predict(model, dev, batch_size=16, max_state_tokens=args.max_state_tokens)
    rep = report(scored)
    print(f"--- dev at step {step} ---")
    print(format_report(rep))
    log.write(json.dumps({"step": step, "dev": rep["overall"], "dev_by_source": rep["by_source"]}) + "\n")
    log.flush()
    with open(os.path.join(args.out, f"dev_step{step}.json"), "w") as f:
        json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()
