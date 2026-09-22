"""One configuration from a dataset to a calibrated, abstaining, servable decision model.

A pipeline is a JSON file naming the data, the tier and the hyper-parameters. The stages run
in order as separate processes -- training frees its memory before evaluation starts -- and
each stage is skipped when the files it produces are already there, so an interrupted run
continues where it stopped. The result is a model directory the server and the publisher
both accept.

    python -m assay.pipeline --config examples/pipelines/support-decoder.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

from assay.conformal import CONFORMAL_FILE
from assay.tiers import CONFIG_FILES as CONFIG_BY_TIER
from assay.tiers import TIERS

REQUIRED = ("name", "tier", "base_model", "data", "out")
DATA_FILES = ("train.jsonl", "dev.jsonl", "calibration.jsonl")


@dataclass
class Stage:
    """One process of the pipeline. `produces` are paths relative to the run directory; when
    they all exist the stage has already run."""

    name: str
    argv: list[str]
    produces: list[str] = field(default_factory=list)


def flags(options: dict) -> list[str]:
    """Command-line flags for a mapping of options. Underscores become dashes, True becomes a
    bare flag, False and None drop out, and a list becomes repeated values; anything the
    trainers accept can therefore be set from the configuration without naming it here."""
    argv: list[str] = []
    for key, value in options.items():
        flag = "--" + str(key).replace("_", "-").lstrip("-")
        if value is False or value is None:
            continue
        if value is True:
            argv.append(flag)
        elif isinstance(value, (list, tuple)):
            argv += [flag] + [str(item) for item in value]
        else:
            argv += [flag, str(value)]
    return argv


def load_config(path: str) -> dict:
    with open(path) as f:
        config = json.load(f)
    validate(config)
    return config


def validate(config: dict) -> None:
    missing = [key for key in REQUIRED if not config.get(key)]
    if missing:
        raise ValueError(f"the pipeline needs {', '.join(missing)}")
    if config["tier"] not in TIERS:
        raise ValueError(f"tier must be one of {', '.join(TIERS)}, not {config['tier']!r}")
    absent = [f for f in DATA_FILES if not os.path.exists(os.path.join(config["data"], f))]
    if absent:
        raise ValueError(f"{config['data']} has no {', '.join(absent)}")
    for name, path in config.get("evaluate", {}).items():
        if not os.path.exists(path):
            raise ValueError(f"the {name} evaluation file {path} does not exist")


def evaluations(config: dict) -> dict[str, str]:
    """The splits to evaluate: what the configuration names, or dev and holdout from the data
    directory when it names none. Both tiers then report on the same splits."""
    named = config.get("evaluate")
    if named:
        return named
    found = {}
    for split in ("dev", "holdout"):
        path = os.path.join(config["data"], f"{split}.jsonl")
        if os.path.exists(path):
            found[split] = path
    return found


def stages(config: dict) -> list[Stage]:
    """The stages for this configuration. The decoder tier calibrates and evaluates in their
    own processes; the encoder and seq2seq trainer does both itself at the end of training."""
    out = config["out"]
    tier = config["tier"]
    evaluate = evaluations(config)
    python = [sys.executable, "-m"]
    if tier == "decoder":
        train = Stage(
            "train",
            python
            + [
                "assay.train",
                "--base",
                config["base_model"],
                "--data",
                config["data"],
                "--out",
                out,
            ]
            + flags(config.get("train", {})),
            [CONFIG_BY_TIER[tier], "assay_head.safetensors"],
        )
        steps = [
            train,
            Stage(
                "calibrate",
                python
                + [
                    "assay.calibrate",
                    "--model",
                    out,
                    "--data",
                    os.path.join(config["data"], "calibration.jsonl"),
                ]
                + flags(config.get("calibrate", {})),
                ["calibration.json"],
            ),
        ]
        for name, path in evaluate.items():
            # raw first, then scaled: the conformal fit reads the raw scores and applies the
            # fitted temperature itself, so `eval-<name>.json` must not already carry it
            for suffix, extra in (("", ["--temperature", "1.0"]), ("-scaled", [])):
                steps.append(
                    Stage(
                        f"evaluate-{name}{suffix}",
                        python
                        + [
                            "assay.evaluate",
                            "--model",
                            out,
                            "--data",
                            path,
                            "--out",
                            os.path.join(out, f"eval-{name}{suffix}.json"),
                        ]
                        + extra
                        + flags(config.get("evaluate_options", {})),
                        [f"eval-{name}{suffix}.json"],
                    )
                )
    else:
        arch = "seq2seq" if tier == "seq2seq" else config.get("architecture", "compiled")
        steps = [
            Stage(
                "train",
                python
                + [
                    "assay.train_compiled",
                    "--arch",
                    arch,
                    "--encoder",
                    config["base_model"],
                    "--data",
                    config["data"],
                    "--out",
                    out,
                    "--evaluate",
                ]
                + [f"{name}={path}" for name, path in evaluate.items()]
                + flags(config.get("train", {})),
                [CONFIG_BY_TIER[tier], "calibration.json"],
            )
        ]
    conformal = config.get("conformal", {})
    steps.append(
        Stage(
            "conformal",
            python
            + ["assay.conformal", "--model", out, "--evaluate"]
            + list(evaluate)
            + flags(conformal),
            [CONFORMAL_FILE],
        )
    )
    return steps


def done(stage: Stage, out: str) -> bool:
    return bool(stage.produces) and all(
        os.path.exists(os.path.join(out, name)) for name in stage.produces
    )


def run_stage(stage: Stage, out: str) -> float:
    """Run one stage, its output going to the terminal and to `<out>/<stage>.log`."""
    os.makedirs(out, exist_ok=True)
    log_path = os.path.join(out, f"{stage.name}.log")
    started = time.time()
    with open(log_path, "a") as log:
        log.write(f"\n=== {stage.name} {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log.write(" ".join(stage.argv) + "\n")
        log.flush()
        process = subprocess.Popen(
            stage.argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"stage {stage.name} failed with exit code {code}; see {log_path}")
    return time.time() - started


def summary(config: dict) -> dict:
    """The numbers a reader wants after a run: the fitted temperature, every evaluation, and
    the conformal thresholds, read back from the files the stages wrote."""
    out = config["out"]
    result: dict = {"name": config["name"], "tier": config["tier"], "model": out}
    calibration = os.path.join(out, "calibration.json")
    if os.path.exists(calibration):
        with open(calibration) as f:
            result["temperature"] = json.load(f).get("temperature")
    measured: dict = {}
    for name in evaluations(config):
        for candidate in (f"eval-{name}-scaled.json", f"eval-{name}.json"):
            path = os.path.join(out, candidate)
            if os.path.exists(path):
                with open(path) as f:
                    overall = json.load(f).get("overall", {})
                measured[name] = {
                    key: overall[key] for key in ("n", "accuracy", "brier", "ece") if key in overall
                }
                break
    if measured:
        result["evaluations"] = measured
    conformal = os.path.join(out, CONFORMAL_FILE)
    if os.path.exists(conformal):
        with open(conformal) as f:
            result["conformal"] = json.load(f).get("types")
    return result


def run(config: dict, force: bool = False, only: list[str] | None = None) -> dict:
    out = config["out"]
    os.makedirs(out, exist_ok=True)
    manifest: dict = {"config": config, "stages": []}
    for stage in stages(config):
        if only and stage.name not in only:
            continue
        if not force and done(stage, out):
            print(f"[{stage.name}] already done", flush=True)
            manifest["stages"].append({"name": stage.name, "status": "skipped"})
            continue
        print(f"[{stage.name}] {' '.join(stage.argv)}", flush=True)
        seconds = run_stage(stage, out)
        manifest["stages"].append(
            {"name": stage.name, "status": "ran", "seconds": round(seconds, 1)}
        )
    manifest["summary"] = summary(config)
    with open(os.path.join(out, "pipeline.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="pipeline JSON")
    ap.add_argument("--only", nargs="*", help="run just these stages")
    ap.add_argument("--force", action="store_true", help="run stages that are already done")
    ap.add_argument("--dry-run", action="store_true", help="print the stages and stop")
    args = ap.parse_args()
    config = load_config(args.config)
    if args.dry_run:
        for stage in stages(config):
            state = "done" if done(stage, config["out"]) else "to run"
            print(f"{stage.name} ({state}): {' '.join(stage.argv)}")
        return
    manifest = run(config, force=args.force, only=args.only)
    print(json.dumps(manifest["summary"], indent=2))
    print(
        f"\nserve it with: python -m assay.server --model {config['out']}"
        f"\npublish it with: python -m assay.publish --run {config['out']} --repo <user>/<name>"
    )


if __name__ == "__main__":
    main()
