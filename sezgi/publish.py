"""Merge the adapter into the base model, write a model card and upload to the Hugging Face Hub.

    uv run python -m sezgi.publish --run runs/sezgi-4b --repo Berk/sezgi-4b [--no-merge]

The published repository contains the merged weights (plain Qwen3 checkpoint loadable with
transformers), the LoRA adapter, the evidence head, sezgi_config.json and the evaluation
results the card cites.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from huggingface_hub import HfApi

from sezgi.model import CONFIG_FILE, HEAD_FILE, SezgiModel


def metrics_row(name: str, path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        o = json.load(f)["overall"]
    return (
        f"| {name} | {o['count']} | {o['accuracy']:.3f} | {o['brier']:.3f} | {o['nll']:.3f} | "
        f"{o['ece']:.3f} | {o['confident_error_rate']:.3f} |"
    )


def write_model_card(run: str, repo: str, base: str, out_path: str) -> None:
    with open(os.path.join(run, "train_args.json")) as f:
        targs = json.load(f)
    calibration = {}
    if os.path.exists(os.path.join(run, "calibration.json")):
        with open(os.path.join(run, "calibration.json")) as f:
            calibration = json.load(f)
    rows = []
    for name, fn in [
        ("seen tasks (dev), raw", "eval-dev.json"),
        ("seen tasks (dev), scaled", "eval-dev-scaled.json"),
        ("unseen tasks (holdout), raw", "eval-holdout.json"),
        ("unseen tasks (holdout), scaled", "eval-holdout-scaled.json"),
        ("kev transfer-v4 dev, raw", "eval-transfer-v4.json"),
        ("kev transfer-v4 dev, scaled", "eval-transfer-v4-scaled.json"),
    ]:
        row = metrics_row(name, os.path.join(run, fn))
        if row:
            rows.append(row)
    table = "\n".join(
        ["| split | n | accuracy | Brier | NLL | ECE | confident errors |", "|---|---|---|---|---|---|---|"] + rows
    )
    temperature = calibration.get("temperature", 1.0)
    card = f"""---
license: apache-2.0
base_model: {base}
library_name: transformers
pipeline_tag: text-classification
tags:
  - sezgi
  - calibrated
  - decision-model
  - zero-shot-classification
  - system-one
language:
  - en
---

# {repo.split('/')[-1]}

Calibrated typed decisions from one forward pass. Send a state and named typed questions
(`noul` yes/no, `choice` over 2..255 described options, `score` over 2..10 ordered levels);
get a probability distribution per question, a confidence and an evidence score. No text is
generated, so nothing can come back off-schema.

Code, server and training recipe: https://github.com/bgokden/sezgi

## How it is built

- Backbone `{base}` with a LoRA adapter (r={targs['lora_r']}, alpha={targs['lora_alpha']},
  lr={targs['lr']}, {targs['epochs']} epoch, batch {targs['batch_size']}); merged weights are in
  this repository, the adapter is in `adapter/`.
- The answer is read from the model's own next-token logits over option label tokens at a
  single decision position, so the base model's zero-shot competence is the starting point.
- Questions are isolated branches over a shared state (block attention mask, restarted
  positions): packed and separate requests agree exactly.
- Trained with cross-entropy against soft targets: human label distributions where the source
  has them, SORD-smoothed levels for ordinal questions, one-hot otherwise. Choice options are
  shuffled per example.
- An evidence head (linear on the decision token, `sezgi_head.safetensors`) predicts whether the
  state supports the question, trained on passage-swapped negatives.
- Global temperature {temperature:.3f} fitted on the calibration split of the training tasks and
  applied unchanged everywhere else.

## Evaluation

{table}

"Unseen tasks" are eleven datasets never used in training (bbc_news, app_reviews, scitail,
medical_questions_pairs, tweet_irony, ethos, stance_climate, dream, copa, truthful_qa,
hh_rlhf). "kev transfer-v4 dev" is the public suite from
[jaredpalmer/kev-suites](https://huggingface.co/datasets/jaredpalmer/kev-suites) (mmlu,
emotion, sciq, tweet_offensive, qnli, paws and synthetic rule holdouts); none of its sources
are in the training data. Brier is the multi-class sum of squared errors (0..2), ECE uses 15
bins, confident errors are answers with p >= 0.9 that are wrong.

## Usage

```python
from sezgi.model import SezgiModel
from sezgi.schema import Question

model = SezgiModel.from_pretrained("{repo}")
answers = model.answer(
    state="My card was charged twice for order A-104.",
    questions={{
        "refund": Question(type="noul", instructions="Does the customer ask for money back?"),
        "team": Question(type="choice", instructions="Which team should handle this?",
                         options={{"billing": "Charges and refunds", "technical": "Bugs"}}),
    }},
)
print(answers["team"].probabilities, answers["refund"].noul, answers["refund"].evidence)
```

## Limitations

Text only, English training data. No arithmetic, counting, date comparison or multi-hop
reasoning in one pass; keep those in code. Accuracy drops with unrelated state. The evidence
head is trained on coarse swapped-passage negatives. Probabilities are calibrated in aggregate
on the evaluated distributions, which is not a guarantee about any single answer or about your
data; check calibration on your own labels before acting on thresholds.

## Training data

Fifty-five public classification, inference, reading-comprehension and preference datasets
rendered as typed questions with described options (see `sezgi/data/tasks.py` in the
repository for the full list and rubrics). Each dataset keeps its own license.
"""
    with open(out_path, "w") as f:
        f.write(card)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="prepare the folder, do not upload")
    args = ap.parse_args()

    with open(os.path.join(args.run, CONFIG_FILE)) as f:
        config = json.load(f)
    base = config["base_model_id"]
    staging = os.path.join(args.run, "hub")
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)

    model = SezgiModel.from_pretrained(args.run, dtype=torch.bfloat16, device="cpu")
    if not args.no_merge and hasattr(model.lm, "merge_and_unload"):
        merged = model.lm.merge_and_unload()
        merged.save_pretrained(staging, safe_serialization=True)
        model.tokenizer.save_pretrained(staging)
        hub_config = dict(config)
        hub_config["base_model_id"] = args.repo
        hub_config["adapter"] = None
        hub_config["merged_from"] = base
    else:
        hub_config = dict(config)
    if config.get("adapter"):
        shutil.copytree(os.path.join(args.run, config["adapter"]), os.path.join(staging, "adapter"))
    shutil.copy(os.path.join(args.run, HEAD_FILE), os.path.join(staging, HEAD_FILE))
    with open(os.path.join(staging, CONFIG_FILE), "w") as f:
        json.dump(hub_config, f, indent=2)
    for fn in os.listdir(args.run):
        if fn.startswith("eval-") and fn.endswith(".json") or fn in ("calibration.json", "train_args.json", "train_log.jsonl"):
            shutil.copy(os.path.join(args.run, fn), os.path.join(staging, fn))
    write_model_card(args.run, args.repo, base, os.path.join(staging, "README.md"))
    print(f"staged {staging}: {sorted(os.listdir(staging))}")
    if args.dry_run:
        return
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=staging, repo_id=args.repo, repo_type="model")
    print(f"uploaded to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
