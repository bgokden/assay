import json
import os
import sys

import pytest

from assay.pipeline import Stage, done, flags, load_config, run, run_stage, stages, validate


def write_data(tmp_path) -> str:
    data = tmp_path / "data"
    data.mkdir()
    for name in ("train.jsonl", "dev.jsonl", "calibration.jsonl", "holdout.jsonl"):
        (data / name).write_text("")
    return str(data)


def config_for(tmp_path, tier: str) -> dict:
    data = write_data(tmp_path)
    return {
        "name": "test",
        "tier": tier,
        "base_model": "some/base",
        "data": data,
        "out": str(tmp_path / "run"),
        "evaluate": {"holdout": os.path.join(data, "holdout.jsonl")},
        "train": {"epochs": 1, "lora_r": 8, "resume": True, "hard_targets": False},
        "conformal": {"alpha": 0.1},
    }


def test_flags_maps_options_to_the_trainer_flags():
    assert flags({"epochs": 2, "lora_r": 8}) == ["--epochs", "2", "--lora-r", "8"]
    assert flags({"resume": True, "hard_targets": False, "missing": None}) == ["--resume"]
    assert flags({"extra": ["a.jsonl", "b.jsonl"]}) == ["--extra", "a.jsonl", "b.jsonl"]


def test_decoder_stages_are_train_calibrate_evaluate_conformal(tmp_path):
    steps = stages(config_for(tmp_path, "decoder"))
    assert [s.name for s in steps] == [
        "train",
        "calibrate",
        "evaluate-holdout",
        "evaluate-holdout-scaled",
        "conformal",
    ]


def test_the_raw_evaluation_is_not_temperature_scaled(tmp_path):
    """The conformal fit rescales the raw scores itself, so eval-<name>.json must be raw or
    the temperature is applied twice."""
    by_name = {s.name: s for s in stages(config_for(tmp_path, "decoder"))}
    raw = " ".join(by_name["evaluate-holdout"].argv)
    scaled = " ".join(by_name["evaluate-holdout-scaled"].argv)
    assert "--temperature 1.0" in raw
    assert "--temperature" not in scaled
    assert raw.endswith("--batch-size 16") or "eval-holdout.json" in raw


def test_encoder_trains_calibrates_and_evaluates_in_one_stage(tmp_path):
    """The compiled trainer fits the temperature and evaluates itself, so the pipeline must
    not run those again."""
    config = config_for(tmp_path, "encoder")
    steps = stages(config)
    assert [s.name for s in steps] == ["train", "conformal"]
    train = " ".join(steps[0].argv)
    assert "--arch compiled" in train and "--evaluate holdout=" in train
    assert "--epochs 1" in train and "--lora-r 8" in train and "--resume" in train


def test_seq2seq_uses_the_seq2seq_architecture(tmp_path):
    train = stages(config_for(tmp_path, "seq2seq"))[0]
    assert "--arch" in train.argv and train.argv[train.argv.index("--arch") + 1] == "seq2seq"
    assert train.produces[0] == "assay_seq2seq_config.json"


def test_validate_rejects_a_bad_tier_and_missing_data(tmp_path):
    config = config_for(tmp_path, "decoder")
    validate(config)
    with pytest.raises(ValueError, match="tier must be one of"):
        validate({**config, "tier": "vision"})
    with pytest.raises(ValueError, match="no train.jsonl"):
        validate({**config, "data": str(tmp_path)})
    with pytest.raises(ValueError, match="needs name"):
        validate({**config, "name": ""})
    with pytest.raises(ValueError, match="does not exist"):
        validate({**config, "evaluate": {"holdout": str(tmp_path / "nowhere.jsonl")}})


def test_load_config_reads_a_file(tmp_path):
    config = config_for(tmp_path, "decoder")
    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps(config))
    assert load_config(str(path))["name"] == "test"


def test_a_stage_is_skipped_once_its_files_exist(tmp_path):
    out = str(tmp_path / "run")
    os.makedirs(out)
    stage = Stage("train", [sys.executable, "-c", "print('ran')"], ["assay_config.json"])
    assert not done(stage, out)
    open(os.path.join(out, "assay_config.json"), "w").close()
    assert done(stage, out)


def test_run_stage_logs_and_raises_on_failure(tmp_path):
    out = str(tmp_path / "run")
    stage = Stage("hello", [sys.executable, "-c", "print('from the stage')"])
    assert run_stage(stage, out) >= 0
    with open(os.path.join(out, "hello.log")) as log:
        assert "from the stage" in log.read()
    failing = Stage("bad", [sys.executable, "-c", "raise SystemExit(3)"])
    with pytest.raises(RuntimeError, match="exit code 3"):
        run_stage(failing, out)


def test_run_writes_a_manifest_and_a_summary(tmp_path, monkeypatch):
    config = config_for(tmp_path, "decoder")
    os.makedirs(config["out"], exist_ok=True)
    with open(os.path.join(config["out"], "calibration.json"), "w") as f:
        json.dump({"temperature": 1.23}, f)
    with open(os.path.join(config["out"], "eval-holdout.json"), "w") as f:
        json.dump({"overall": {"n": 4, "accuracy": 0.75, "brier": 0.3, "ece": 0.02}}, f)
    ran = []
    monkeypatch.setattr(
        "assay.pipeline.run_stage", lambda stage, out: ran.append(stage.name) or 0.0
    )
    manifest = run(config)
    # calibration and the raw evaluation were already there; the scaled one was not
    assert ran == ["train", "evaluate-holdout-scaled", "conformal"]
    assert manifest["summary"]["temperature"] == 1.23
    assert manifest["summary"]["evaluations"]["holdout"]["accuracy"] == 0.75
    with open(os.path.join(config["out"], "pipeline.json")) as f:
        assert json.load(f)["summary"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


def test_a_decoder_run_with_a_checkpoint_resumes(tmp_path):
    config = config_for(tmp_path, "decoder")
    config["train"].pop("resume")
    assert "--resume" not in " ".join(stages(config)[0].argv)
    os.makedirs(os.path.join(config["out"], "checkpoint"), exist_ok=True)
    assert "--resume" in " ".join(stages(config)[0].argv)
    config["train"]["resume"] = False  # asked for a fresh run, and that is honoured
    assert "--resume" not in " ".join(stages(config)[0].argv)
