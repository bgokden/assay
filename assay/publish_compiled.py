"""Publish an encoder-tier model (compiled, conditioned or cross) to the Hugging Face Hub.

    uv run python -m assay.publish_compiled --run runs/compiled-late-gte-base --repo Berk/assay-compiled-base

The run directory already holds the encoder, tokenizer, head weights and config as saved by
`assay.train_compiled`; this stages them with the evaluation files and a model card.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

from huggingface_hub import HfApi

from assay.compiled import CONFIG_FILE, WEIGHTS_FILE
from assay.publish import FAMILY_SECTION, metrics_row

MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    WEIGHTS_FILE,
    CONFIG_FILE,
)

ARCH_TEXT = {
    "compiled": (
        "The state is encoded once into token embeddings. Each question is compiled once: the "
        "instruction becomes a set of query vectors (learned slots plus a projection of its pooled "
        "encoding), each option becomes a vector (its pooled encoding) and its token encodings. A "
        "decision is cross-attention from the queries over the state tokens, a small reader MLP, a "
        "bilinear score against each option vector plus a per-option bias, and a late-interaction "
        "term: the mean over option tokens of the best cosine match among state tokens, times a "
        "learned scale. Compiled questions can be cached and reused across states; after the state "
        "encode, a decision is a few small matrix products."
    ),
    "conditioned": (
        "The instruction and the state are encoded together (one encoder pass per question), so "
        "the encoder's own attention relates them; learned query slots read the joint sequence, and "
        "options are compiled separately and scored by content (bilinear score plus bias, with a "
        "late-interaction term when enabled)."
    ),
    "cross": (
        "One encoder pass per (option, state) pair with the instruction in front, scored by a linear "
        "head over the pooled encoding; softmax over a question's options."
    ),
}


def write_card(run: str, repo: str, out_path: str) -> None:
    with open(os.path.join(run, CONFIG_FILE)) as f:
        config = json.load(f)
    with open(os.path.join(run, "train_args.json")) as f:
        targs = json.load(f)
    arch = config.get("arch", "compiled")
    late = config.get("late_interaction", False)
    rows = [
        metrics_row(name, os.path.join(run, fn))
        for name, fn in (
            ("seen tasks (dev), scaled", "eval-dev-scaled.json"),
            ("unseen tasks (holdout), raw", "eval-holdout.json"),
            ("unseen tasks (holdout), scaled", "eval-holdout-scaled.json"),
            ("kev transfer-v4 dev, raw", "eval-transfer-v4.json"),
            ("kev transfer-v4 dev, scaled", "eval-transfer-v4-scaled.json"),
        )
    ]
    table = "\n".join(r for r in rows if r)
    latency = ""
    path = os.path.join(run, "latency-cpu.txt")
    if os.path.exists(path):
        with open(path) as f:
            latency = (
                "CPU latency (milliseconds; the state is encoded once, questions are compiled once and cached, `decide` runs per state x question set):\n\n```\n"
                + f.read().strip()
                + "\n```\n"
            )
    card = f"""---
license: apache-2.0
base_model: {config["encoder_id"]}
language: en
tags:
- text-classification
- calibration
- decision-model
- assay
---

# {repo.split("/")[-1]}

An encoder-tier decision model from the [Assay](https://github.com/bgokden/assay) project:
typed questions (`bool`, `choice`, `score`) over a state produce calibrated probability
distributions with an `evidence` signal, with no text generation and no language model. It
is the small, CPU-friendly tier; the decoder models ([Berk/assay-4b](https://huggingface.co/Berk/assay-4b),
[Berk/assay-27b](https://huggingface.co/Berk/assay-27b)) are far more accurate.

Architecture `{arch}`{" with late interaction" if late else ""} on `{config["encoder_id"]}`
({config["slots"]} query slots). {ARCH_TEXT[arch]}

## Evaluation

| split | n | accuracy | Brier | NLL | ECE | confident errors |
|---|---|---|---|---|---|---|
{table}

"Scaled" applies the temperature {config.get("temperature", 1.0):.3f} fitted on the seen-task
calibration split. Unseen tasks are eleven datasets never trained on; the transfer suite is
`jaredpalmer/kev-suites` transfer-v4 dev, whose sources are excluded from training. Single-text
classification (topic, sentiment, spam) is strong; questions that need knowledge (MMLU) or
multi-step reasoning are near chance. See the repository's `docs/roadmap.md` for the full
comparison against the cross-encoder and the decoders.

{latency}
{FAMILY_SECTION}
## Usage

```python
from assay.compiled import load_any
from assay.schema import Question

model = load_any("{repo}", device="cpu")
answers = model.answer(
    "My card was charged twice for order A-104.",
    {{
        "refund": Question(type="bool", instructions="Does the customer ask for money back?"),
        "team": Question(type="choice", instructions="Which team should handle this?",
                         options={{"billing": "Charges and refunds", "technical": "Bugs"}}),
    }},
)
print(answers["team"].probabilities, answers["refund"].p_true)
```

Trained with `assay.train_compiled` ({targs.get("epochs")} epochs, lr {targs.get("lr")}, head lr
{targs.get("head_lr")}, batch {targs.get("batch_size")}) on the Assay data (55 public datasets
rendered as typed questions, synthetic policy and date cases, and 40k generic questions labelled
by assay-27b). Each dataset keeps its own licence; the list is in
[docs/datasets.md](https://github.com/bgokden/assay/blob/main/docs/datasets.md).

## Limitations

English only. No knowledge beyond what the encoder carries, no arithmetic, no multi-hop
reasoning. Calibrated in aggregate on the evaluated distributions, not per answer; check on
your own labels before acting on thresholds.

## Relationship to other work

Assay is an independent project. Jev and System One are names of TypeSafe AI's products and
are mentioned only to describe and compare; kev-suites is Jared Palmer's evaluation data.
Assay is not affiliated with or endorsed by either.
"""
    with open(out_path, "w") as f:
        f.write(card)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--card-only", action="store_true", help="upload just the model card")
    args = ap.parse_args()
    staging = os.path.join(args.run, "hub")
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)
    for fn in os.listdir(args.run):
        if (
            fn in MODEL_FILES
            or (fn.startswith("eval-") and fn.endswith(".json"))
            or fn
            in (
                "calibration.json",
                "conformal.json",
                "train_args.json",
                "train_log.jsonl",
                "latency-cpu.txt",
            )
        ):
            shutil.copy(os.path.join(args.run, fn), os.path.join(staging, fn))
    write_card(args.run, args.repo, os.path.join(staging, "README.md"))
    print(f"staged {staging}: {sorted(os.listdir(staging))}")
    if args.dry_run:
        return
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    if args.card_only:
        api.upload_file(
            path_or_fileobj=os.path.join(staging, "README.md"),
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="model",
        )
    else:
        api.upload_folder(folder_path=staging, repo_id=args.repo, repo_type="model")
    print(f"uploaded to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
