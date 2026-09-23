"""Shared test setup.

The tests that need a GPU load a small model onto it. This machine also trains on that GPU,
so a run can find the card already full: without this the suite fails with an out-of-memory
error that looks like a bug in the code under test. Skipping with a clear reason is honest,
and the tests still run in full when the card is free.
"""

import os

import pytest
import torch

DEVICE = os.environ.get("ASSAY_TEST_DEVICE", "cuda")
NEEDED_GIB = float(os.environ.get("ASSAY_TEST_FREE_GIB", "6"))


def free_gib() -> float:
    free, _ = torch.cuda.mem_get_info()
    return free / 2**30


def pytest_collection_modifyitems(config, items):
    if DEVICE != "cuda" or not torch.cuda.is_available():
        return
    available = free_gib()
    if available >= NEEDED_GIB:
        return
    skip = pytest.mark.skip(
        reason=f"only {available:.1f} GiB free on the GPU, {NEEDED_GIB:.0f} needed "
        "(something else is using the card)"
    )
    for item in items:
        # every module that targets the GPU declares DEVICE at the top; fixtures are not a
        # reliable signal because some tests build their own model
        if getattr(item.module, "DEVICE", None) == "cuda":
            item.add_marker(skip)
