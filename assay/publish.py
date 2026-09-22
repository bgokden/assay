"""Merge the adapter into the base model, write a model card and upload to the Hugging Face Hub.

    uv run python -m assay.publish --run runs/assay-4b --repo Berk/assay-4b [--no-merge]

The published repository contains the merged weights (plain Qwen3 checkpoint loadable with
transformers), the LoRA adapter, the evidence head, assay_config.json and the evaluation
results the card cites.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from huggingface_hub import HfApi

from assay.family import short_table
from assay.model import CONFIG_FILE, HEAD_FILE, AssayModel


def family_section() -> str:
    """The card's family table, generated from the same source as docs/models.md."""
    return f"""## The family

{short_table()}

Accuracy / Brier after temperature scaling. Same recipe, same splits, different backbones;
per-tier abstention and latency are in
[docs/models.md](https://github.com/bgokden/assay/blob/main/docs/models.md).

## Serving

`assay.server` exposes `POST /v1/decide`, the System One style `POST /v1/systemone`
(questions typed `choice`/`noul`/`score` with options under `criteria`, plus a batch
endpoint), and `POST /v1/decide_graph`, which walks a decision tree in a single forward pass.
Requests arriving together are answered in one pass; `/health` and `/metrics` are for
operations. `assay.backends.sglang` runs the same model on an SGLang deployment.

## Train one on your own data

`python -m assay.pipeline --config <your>.json` runs training, temperature calibration,
evaluation and the conformal thresholds over your own records, and writes a directory this
same server and publisher accept. The repository's
[examples/](https://github.com/bgokden/assay/tree/main/examples) has a configuration per tier
and a dataset in the record format; records written for other decision models (`criteria`
options, `noul` booleans) load unchanged.

"""


def metrics_row(name: str, path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        o = json.load(f)["overall"]
    return (
        f"| {name} | {o['count']} | {o['accuracy']:.3f} | {o['brier']:.3f} | {o['nll']:.3f} | "
        f"{o['ece']:.3f} | {o['confident_error_rate']:.3f} |"
    )


def abstention_section(run: str) -> str:
    """Model-card paragraph on the conformal thresholds, when the run has them."""
    path = os.path.join(run, "conformal-eval.json")
    conf_path = os.path.join(run, "conformal.json")
    if not (os.path.exists(path) and os.path.exists(conf_path)):
        return ""
    with open(conf_path) as f:
        conf = json.load(f)
    with open(path) as f:
        ev = json.load(f)
    rows = []
    for split, name in (("holdout", "unseen tasks"), ("transfer-v4", "transfer suite")):
        for qtype, v in ev.get(split, {}).items():
            act = (
                f"{v['act_rate']:.0%} / {v['act_error']:.1%}"
                if v["act_rate"] > 0
                else "no threshold"
            )
            rows.append(
                f"| {name} | {qtype} | {v['coverage']:.2f} | {v['mean_set_size']:.2f} | {act} |"
            )
    table = "\n".join(rows)
    return f"""## Abstention

`conformal.json` holds per-question-type thresholds fitted on the seen-task calibration split
(alpha {conf["alpha"]}, delta {conf["delta"]}): a prediction-set threshold with coverage at least
1 - alpha and an act threshold on the top probability whose acted-on error rate is at most alpha
at confidence 1 - delta, both on inputs distributed like the calibration split. `assay.server`
returns them as `act` and `set` on every answer. How they carry over to other tasks:

| split | type | coverage | set size | act rate / error among acted |
|---|---|---|---|---|
{table}

Score questions get prediction sets but usually no act threshold, because exact-level accuracy
is the wrong error notion for ordinal answers. Refit on your own labelled data with
`python -m assay.conformal` for a guarantee about your distribution.
"""


def write_model_card(run: str, repo: str, base: str, out_path: str, merged: bool = True) -> None:
    with open(os.path.join(run, "train_args.json")) as f:
        targs = json.load(f)
    quant = targs.get("quant")
    if merged:
        weights_line = "merged weights are in this repository, the adapter is in `adapter/`."
        load_line = "The merged weights load with transformers like any Qwen checkpoint."
    else:
        weights_line = (
            f"this repository holds the adapter (`adapter/`) and the evidence head; the base is loaded "
            f"from `{base}`" + (f" in {quant} (bitsandbytes)" if quant else "") + " at load time."
        )
        load_line = "Loading downloads the base model separately; the 4-bit base needs about 15 GB of GPU memory."
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
        [
            "| split | n | accuracy | Brier | NLL | ECE | confident errors |",
            "|---|---|---|---|---|---|---|",
        ]
        + rows
    )
    by_source = ""
    path = os.path.join(run, "eval-transfer-v4-scaled.json")
    if os.path.exists(path):
        with open(path) as f:
            groups = json.load(f)["by_source"]
        lines = ["| transfer-v4 source | n | accuracy | Brier | ECE |", "|---|---|---|---|---|"]
        for name, o in groups.items():
            lines.append(
                f"| {name} | {o['count']} | {o['accuracy']:.3f} | {o['brier']:.3f} | {o['ece']:.3f} |"
            )
        by_source = "\n".join(lines)
    latency = ""
    path = os.path.join(run, "latency.txt")
    if os.path.exists(path):
        with open(path) as f:
            latency = (
                "Latency on one RTX 5090 (bf16, transformers, packed questions over one state versus separate requests):\n\n```\n"
                + f.read().strip()
                + "\n```\n"
            )
    temperature = calibration.get("temperature", 1.0)
    abstention = abstention_section(run)
    card = f"""---
license: apache-2.0
base_model: {base}
library_name: transformers
pipeline_tag: text-classification
tags:
  - assay
  - calibrated
  - decision-model
  - zero-shot-classification
language:
  - en
---

# {repo.split("/")[-1]}

Calibrated typed decisions from one forward pass. Send a state and named typed questions
(`bool` yes/no, `choice` over 2..255 described options, `score` over 2..10 ordered levels);
get a probability distribution per question, a confidence and an evidence score. No text is
generated, so nothing can come back off-schema.

Code, server and training recipe: https://github.com/bgokden/assay

## How it is built

- Backbone `{base}` with a LoRA adapter (r={targs["lora_r"]}, alpha={targs["lora_alpha"]},
  lr={targs["lr"]}, {targs["epochs"]} epoch, batch {targs["batch_size"]} x {targs.get("grad_accum", 1)}
  accumulation{", 4-bit base (QLoRA)" if quant else ""}); {weights_line} {load_line}
- The answer is read from the model's own next-token logits over option label tokens at a
  single decision position, so the base model's zero-shot competence is the starting point.
- Questions are isolated branches over a shared state (block attention mask, restarted
  positions): packed and separate requests agree exactly.
- Trained with cross-entropy against soft targets: human label distributions where the source
  has them, SORD-smoothed levels for ordinal questions, one-hot otherwise. Choice options are
  shuffled per example.
- An evidence head (linear on the decision token, `assay_head.safetensors`) predicts whether the
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

{by_source}

{latency}
{abstention}
{family_section()}## Usage

```python
from assay.model import AssayModel
from assay.schema import Question

model = AssayModel.from_pretrained("{repo}")
answers = model.answer(
    state="My card was charged twice for order A-104.",
    questions={{
        "refund": Question(type="bool", instructions="Does the customer ask for money back?"),
        "team": Question(type="choice", instructions="Which team should handle this?",
                         options={{"billing": "Charges and refunds", "technical": "Bugs"}}),
    }},
)
print(answers["team"].probabilities, answers["refund"].p_true, answers["refund"].evidence)
```

## Limitations

Text only, English training data. No arithmetic, counting, date comparison or multi-hop
reasoning in one pass; keep those in code. Accuracy drops with unrelated state. The evidence
head is trained on coarse swapped-passage negatives. Probabilities are calibrated in aggregate
on the evaluated distributions, which is not a guarantee about any single answer or about your
data; check calibration on your own labels before acting on thresholds.

## Training data

Fifty-five public classification, inference, reading-comprehension and preference datasets
rendered as typed questions with described options, plus a synthetic policy-application
generator (see `assay/data/tasks.py` in the repository for the rubrics). Each dataset keeps
its own licence; the per-dataset list is in
[docs/datasets.md](https://github.com/bgokden/assay/blob/main/docs/datasets.md). Several
sources carry non-commercial or research-only terms; check them before commercial use.

## Relationship to other work

Assay is an independent project. Jev and System One are names of TypeSafe AI's products and
are mentioned only to describe and compare; kev-suites is Jared Palmer's evaluation data.
Assay is not affiliated with or endorsed by either.
"""
    with open(out_path, "w") as f:
        f.write(card)


def update_card_and_thresholds(
    run: str, repo: str, base: str, merged: bool, dry_run: bool, keep_staging: bool = False
) -> None:
    staging = os.path.join(run, "hub-card")
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)
    write_model_card(run, repo, base, os.path.join(staging, "README.md"), merged=merged)
    for fn in ("conformal.json", "conformal-eval.json"):
        if os.path.exists(os.path.join(run, fn)):
            shutil.copy(os.path.join(run, fn), os.path.join(staging, fn))
    print(f"staged {staging}: {sorted(os.listdir(staging))}")
    if dry_run:
        return
    HfApi().upload_folder(folder_path=staging, repo_id=repo, repo_type="model")
    print(f"updated https://huggingface.co/{repo}")
    if not keep_staging:
        shutil.rmtree(staging)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="prepare the folder, do not upload")
    ap.add_argument(
        "--keep-staging", action="store_true", help="leave the staged copy in the run directory"
    )
    ap.add_argument(
        "--card-only",
        action="store_true",
        help="upload only the model card and the conformal files to an existing repository",
    )
    args = ap.parse_args()

    with open(os.path.join(args.run, CONFIG_FILE)) as f:
        config = json.load(f)
    base = config["base_model_id"]
    if args.card_only:
        update_card_and_thresholds(
            args.run,
            args.repo,
            base,
            merged=not args.no_merge,
            dry_run=args.dry_run,
            keep_staging=args.keep_staging,
        )
        return
    staging = os.path.join(args.run, "hub")
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)

    model = AssayModel.from_pretrained(args.run, dtype=torch.bfloat16, device="cpu")
    if not args.no_merge and hasattr(model.lm, "merge_and_unload"):
        merged = model.lm.merge_and_unload()
        merged.save_pretrained(staging, safe_serialization=True)
        model.tokenizer.save_pretrained(staging)
        hub_config = dict(config)
        hub_config["base_model_id"] = args.repo
        hub_config["adapter"] = None
        hub_config["merged_from"] = base
        hub_config["quantized_base"] = config.get("quantized_base")
    else:
        hub_config = dict(config)
    if config.get("adapter"):
        shutil.copytree(os.path.join(args.run, config["adapter"]), os.path.join(staging, "adapter"))
    shutil.copy(os.path.join(args.run, HEAD_FILE), os.path.join(staging, HEAD_FILE))
    with open(os.path.join(staging, CONFIG_FILE), "w") as f:
        json.dump(hub_config, f, indent=2)
    for fn in os.listdir(args.run):
        if (
            fn.startswith("eval-")
            and fn.endswith(".json")
            or fn in ("calibration.json", "train_args.json", "train_log.jsonl", "conformal.json")
        ):
            shutil.copy(os.path.join(args.run, fn), os.path.join(staging, fn))
    write_model_card(
        args.run, args.repo, base, os.path.join(staging, "README.md"), merged=not args.no_merge
    )
    print(f"staged {staging}: {sorted(os.listdir(staging))}")
    if args.dry_run:
        return
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=staging, repo_id=args.repo, repo_type="model")
    print(f"uploaded to https://huggingface.co/{args.repo}")
    if not args.keep_staging:
        shutil.rmtree(staging)  # a full copy of what the Hub now holds, rebuilt on demand


if __name__ == "__main__":
    main()
