"""Which tier a saved model directory holds.

Every trainer writes one config file, and its name identifies the tier: the decoder reads a
base language model's next-token distribution, the encoder tier runs a sentence encoder with
a small reader, the encoder-decoder tier reads the answer from a seq2seq decoder. Loaders,
the publisher and the conformal fit ask here instead of each knowing the file names.
"""

from __future__ import annotations

import json
import os

TIERS = ("decoder", "encoder", "seq2seq")
CONFIG_FILES = {
    "decoder": "assay_config.json",
    "encoder": "assay_compiled_config.json",
    "seq2seq": "assay_seq2seq_config.json",
}
WEIGHTS_FILES = {
    "decoder": "assay_head.safetensors",
    "encoder": "assay_compiled.safetensors",
    "seq2seq": "assay_seq2seq.safetensors",
}


def tier_of(path: str) -> str:
    """The tier of a saved model directory, by the config file its trainer wrote."""
    for tier in TIERS:
        if os.path.exists(os.path.join(path, CONFIG_FILES[tier])):
            return tier
    raise FileNotFoundError(f"{path} holds none of {', '.join(CONFIG_FILES.values())}")


def config_of(path: str) -> dict:
    with open(os.path.join(path, CONFIG_FILES[tier_of(path)])) as f:
        return json.load(f)


def temperature_of(path: str) -> float:
    return float(config_of(path).get("temperature", 1.0))
