"""Checkpoint writing on a disk that can fill up.

A full disk truncates a torch.save without always raising, and a short checkpoint is worse
than an old one: it cannot be loaded, so the run has nothing to resume from. Every write goes
through a temporary file, and is skipped outright when the free space cannot hold it.
"""

import os
import shutil

import torch

HEADROOM = 1.2
MAX_CONSECUTIVE_FAILURES = 3


def state_bytes(state) -> int:
    """The bytes the tensors inside a checkpoint occupy, nested containers included."""
    if torch.is_tensor(state):
        return state.numel() * state.element_size()
    if isinstance(state, dict):
        return sum(state_bytes(value) for value in state.values())
    if isinstance(state, (list, tuple)):
        return sum(state_bytes(value) for value in state)
    return 0


def write_checkpoint(path: str, state: dict, step: int) -> bool:
    """Write `state` to `path` atomically. Returns False, leaving any existing checkpoint
    untouched, when the disk cannot hold the write or the write fails."""
    directory = os.path.dirname(path) or "."
    previous = os.path.getsize(path) if os.path.exists(path) else 0
    needed = max(state_bytes(state), previous)
    free = shutil.disk_usage(directory).free
    if free < needed * HEADROOM:
        print(
            f"skipping checkpoint at step {step}: {free / 2**30:.1f} GiB free, "
            f"{needed * HEADROOM / 2**30:.1f} GiB needed",
            flush=True,
        )
        return False
    tmp = path + ".tmp"
    try:
        torch.save(state, tmp)
        written = os.path.getsize(tmp)
        if previous and written < previous * 0.5:
            raise RuntimeError(f"short checkpoint: {written} bytes against {previous} before")
        os.replace(tmp, path)
    except (OSError, RuntimeError) as error:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"checkpoint at step {step} was not written: {error}", flush=True)
        return False
    return True


def remove_checkpoint(path: str) -> None:
    """Drop the checkpoint of a finished run: the weights are saved and the evaluations are
    written, so the optimizer state is only taking up the disk the next run needs."""
    for name in (path, path + ".tmp"):
        if os.path.exists(name):
            os.remove(name)
            print(f"removed {name}", flush=True)
