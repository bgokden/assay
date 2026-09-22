"""Assemble one results table for the whole model family, so every card and the README quote
the same numbers.

    uv run python scripts/family_table.py > docs/models.md
"""

from __future__ import annotations

import json
import os

FAMILY = [
    ("assay-0.6b", "runs/assay-0.6b-v4", "Qwen3-0.6B-Base", "0.6B", "Apache-2.0"),
    ("assay-1.7b", "runs/assay-1.7b-v2", "Qwen3-1.7B-Base", "1.7B", "Apache-2.0"),
    ("assay-4b", "runs/assay-4b-v4", "Qwen3-4B-Base", "4B", "Apache-2.0"),
    ("assay-27b", "runs/assay-27b", "Qwen3.8-27B (4-bit)", "27B", "Apache-2.0"),
    ("assay-compiled-base", "runs/compiled-late-gte-base", "gte-modernbert-base", "149M", "Apache-2.0"),
]


def cells(run: str, name: str) -> str:
    path = os.path.join(run, f"eval-{name}-scaled.json")
    if not os.path.exists(path):
        return "-"
    with open(path) as f:
        o = json.load(f)["overall"]
    return f"{o['accuracy']:.3f} / {o['brier']:.3f} / {o['ece']:.3f}"


def abstention(run: str) -> str:
    path = os.path.join(run, "conformal-eval.json")
    if not os.path.exists(path):
        return "-"
    with open(path) as f:
        ev = json.load(f).get("holdout", {})
    parts = []
    for qtype in ("bool", "choice"):
        v = ev.get(qtype)
        if v and v["act_rate"] > 0:
            parts.append(f"{qtype} {v['act_rate']:.0%} at {v['act_error']:.1%}")
    return ", ".join(parts) or "-"


def latency(run: str) -> str:
    for name in ("latency.txt", "latency-cpu.txt"):
        path = os.path.join(run, name)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            rows = [line.split() for line in f if line.strip() and line.split()[0].isdigit()]
        if not rows:
            continue
        one = next((r[1] for r in rows if r[0] == "1"), "-")
        many = next((r[1] for r in rows if r[0] == "24"), "-")
        where = "CPU" if name.endswith("cpu.txt") else "GPU"
        return f"{one} / {many} ms ({where})"
    return "-"


def main() -> None:
    print("# The Assay model family\n")
    print("Every row is the same recipe on a different backbone, evaluated on the same splits.")
    print("Cells are accuracy / Brier / ECE after one temperature fitted on seen-task")
    print("calibration data. Unseen tasks are eleven datasets never trained on; the transfer")
    print("suite is `jaredpalmer/kev-suites` transfer-v4 dev, whose sources are excluded from")
    print("training. Abstention is the fitted conformal act rate and the error among answers")
    print("acted on, at alpha 0.1, measured on unseen tasks. Latency is one question / 24")
    print("packed questions over one state.\n")
    print("| model | backbone | size | seen (dev) | unseen (holdout) | transfer-v4 | abstention (unseen) | latency |")
    print("|---|---|---|---|---|---|---|---|")
    for name, run, base, size, _ in FAMILY:
        print(
            f"| [{name}](https://huggingface.co/Berk/{name}) | {base} | {size} | "
            f"{cells(run, 'dev')} | {cells(run, 'holdout')} | {cells(run, 'transfer-v4')} | "
            f"{abstention(run)} | {latency(run)} |"
        )
    print("\nAll weights are Apache-2.0. Training data: 55 public datasets rendered as typed")
    print("questions plus our own generators, published at")
    print("[Berk/assay-synthetic](https://huggingface.co/datasets/Berk/assay-synthetic).")


if __name__ == "__main__":
    main()
