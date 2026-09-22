import os

import pytest
import torch

from assay.checkpoint import state_bytes, write_checkpoint


def test_state_bytes_counts_nested_tensors():
    state = {
        "model": {"a": torch.zeros(10, dtype=torch.float32)},
        "optimizer": [torch.zeros(5, dtype=torch.float64), {"step": 3}],
    }
    assert state_bytes(state) == 10 * 4 + 5 * 8


def test_write_and_replace(tmp_path):
    path = str(tmp_path / "checkpoint.pt")
    assert write_checkpoint(path, {"model": torch.ones(4), "step": 1}, 1)
    assert torch.load(path, weights_only=False)["step"] == 1
    assert write_checkpoint(path, {"model": torch.ones(4), "step": 2}, 2)
    assert torch.load(path, weights_only=False)["step"] == 2
    assert not os.path.exists(path + ".tmp")


def test_skips_when_the_disk_cannot_hold_it(tmp_path, monkeypatch):
    """The first checkpoint of a run has nothing to compare against, so the size of the state
    itself must decide: this is what a full disk did to a seq2seq run."""
    path = str(tmp_path / "checkpoint.pt")
    usage = type("Usage", (), {"free": 1024})()
    monkeypatch.setattr("assay.checkpoint.shutil.disk_usage", lambda _: usage)
    assert not write_checkpoint(path, {"model": torch.ones(1024, 1024), "step": 1}, 1)
    assert not os.path.exists(path)


def test_a_failed_write_keeps_the_previous_checkpoint(tmp_path, monkeypatch):
    path = str(tmp_path / "checkpoint.pt")
    assert write_checkpoint(path, {"model": torch.ones(4), "step": 1}, 1)

    def fail(state, target):
        with open(target, "wb") as f:
            f.write(b"short")
        raise RuntimeError("basic_ios::clear: iostream error")

    monkeypatch.setattr("assay.checkpoint.torch.save", fail)
    assert not write_checkpoint(path, {"model": torch.ones(4), "step": 2}, 2)
    assert not os.path.exists(path + ".tmp")
    assert torch.load(path, weights_only=False)["step"] == 1


def test_a_short_write_is_discarded(tmp_path, monkeypatch):
    path = str(tmp_path / "checkpoint.pt")
    assert write_checkpoint(path, {"model": torch.ones(1000), "step": 1}, 1)
    monkeypatch.setattr(
        "assay.checkpoint.torch.save", lambda state, target: open(target, "wb").write(b"x" * 8)
    )
    assert not write_checkpoint(path, {"model": torch.ones(1000), "step": 2}, 2)
    assert torch.load(path, weights_only=False)["step"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
