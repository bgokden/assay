"""Print a markdown table of every run's evaluation results.

uv run python scripts/summarize_runs.py runs/base-1.7b runs/assay-1.7b runs/assay-4b
"""

from __future__ import annotations

import json
import os
import sys

SPLITS = [
    ("dev", "eval-dev.json"),
    ("dev scaled", "eval-dev-scaled.json"),
    ("holdout", "eval-holdout.json"),
    ("holdout scaled", "eval-holdout-scaled.json"),
    ("transfer-v4", "eval-transfer-v4.json"),
    ("transfer-v4 scaled", "eval-transfer-v4-scaled.json"),
]


def cell(run: str, fn: str) -> str:
    path = os.path.join(run, fn)
    if not os.path.exists(path):
        return "-"
    with open(path) as f:
        o = json.load(f)["overall"]
    return f"{o['accuracy']:.3f} / {o['brier']:.3f} / {o['ece']:.3f}"


def main(runs: list[str]) -> None:
    print("| run | " + " | ".join(name for name, _ in SPLITS) + " |")
    print("|---|" + "---|" * len(SPLITS))
    for run in runs:
        print(
            f"| {os.path.basename(run.rstrip('/'))} | "
            + " | ".join(cell(run, fn) for _, fn in SPLITS)
            + " |"
        )
    print("\ncells: accuracy / Brier / ECE")


if __name__ == "__main__":
    main(sys.argv[1:])
