"""Keep saved artifacts loadable by the transformers version other people have.

transformers 5 renamed two things without keeping the old spellings, and transformers 4 is
still what SGLang, vLLM and llama.cpp's converter pin. One rename fails loudly and one fails
silently; the silent one is the reason this module exists rather than a line in the docs.

The tokenizer config's `additional_special_tokens` became `extra_special_tokens`, still
written as a list. transformers 4 reads that name as a mapping, so a tokenizer saved by 5
raises `AttributeError: 'list' object has no attribute 'keys'` on 4. Writing the list under
the old name loads on both. A tokenizer with no concrete class is also saved as
`TokenizersBackend`, which 4 rejects with `Tokenizer class TokenizersBackend does not exist`;
its transformers 4 name is `PreTrainedTokenizerFast`, which 5 still accepts.

The RoPE settings moved into a `rope_parameters` block and the top-level `rope_theta` and
`rope_scaling` stopped being written. transformers 4 reads the old names and, when they are
absent, falls back to the model class's default -- for Qwen3 that is 10000 against a true
1000000, a hundredfold error in the RoPE base. Nothing reports it. The model loads, serves and
answers; it just answers as a different model. Measured on assay-0.6b through SGLang, the
readout hidden state fell to a cosine of 0.9940 against the reference and probabilities moved
by up to 7.1e-2 -- far enough to cross a decision threshold, close enough to look like
rounding. Writing both spellings costs nothing and both generations then agree.
"""

from __future__ import annotations

import json
import os

TOKENIZER_CONFIG = "tokenizer_config.json"
MODEL_CONFIG = "config.json"
# transformers 5 names the generic fast tokenizer after its backend; 4 has never heard of it and
# refuses to build one. A model that keeps a concrete class name, as the Qwen3 tokenizers do, is
# unaffected -- which is why this surfaced only on the ModernBERT tier.
GENERIC_FAST_TOKENIZER_5 = "TokenizersBackend"
GENERIC_FAST_TOKENIZER_4 = "PreTrainedTokenizerFast"


def normalise_tokenizer_config(directory: str) -> bool:
    """Rewrite a saved tokenizer config so both transformers generations load it.

    Returns True when something was changed. Safe to call on a directory without one.
    """
    path = os.path.join(directory, TOKENIZER_CONFIG)
    if not os.path.exists(path):
        return False
    with open(path) as f:
        config = json.load(f)
    changed = False
    extra = config.get("extra_special_tokens")
    if isinstance(extra, list):
        config.pop("extra_special_tokens")
        existing = config.get("additional_special_tokens") or []
        config["additional_special_tokens"] = list(dict.fromkeys([*existing, *extra]))
        changed = True
    if config.get("tokenizer_class") == GENERIC_FAST_TOKENIZER_5:
        config["tokenizer_class"] = GENERIC_FAST_TOKENIZER_4
        changed = True
    if not changed:
        return False
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    return True


# transformers 5 keys a per-attention-type `rope_parameters` block by layer type; transformers 4
# gives each type its own field. ModernBERT is the one this repository publishes.
BY_ATTENTION_TYPE = {"full_attention": "global_rope_theta", "sliding_attention": "local_rope_theta"}


def _restore_rope(config: dict) -> bool:
    """Copy a `rope_parameters` block back out to the names transformers 4 reads."""
    changed = False
    parameters = config.get("rope_parameters")
    if isinstance(parameters, dict) and any(k in BY_ATTENTION_TYPE for k in parameters):
        for layer_type, field in BY_ATTENTION_TYPE.items():
            block = parameters.get(layer_type)
            if (
                isinstance(block, dict)
                and block.get("rope_theta") is not None
                and (config.get(field) is None)
            ):
                config[field] = block["rope_theta"]
                changed = True
        return changed
    if isinstance(parameters, dict):
        theta = parameters.get("rope_theta")
        if theta is not None and config.get("rope_theta") is None:
            config["rope_theta"] = theta
            changed = True
        scaling = {k: v for k, v in parameters.items() if k != "rope_theta"}
        if scaling.get("rope_type", "default") != "default" and config.get("rope_scaling") is None:
            config["rope_scaling"] = scaling
            changed = True
    # multimodal and hybrid checkpoints keep the backbone under a nested config
    for value in list(config.values()):
        if isinstance(value, dict) and "rope_parameters" in value:
            changed = _restore_rope(value) or changed
    return changed


def normalise_model_config(directory: str) -> bool:
    """Rewrite a saved model config so transformers 4 reads the RoPE settings 5 wrote.

    Returns True when something was changed. Safe to call on a directory without one.
    """
    path = os.path.join(directory, MODEL_CONFIG)
    if not os.path.exists(path):
        return False
    with open(path) as f:
        config = json.load(f)
    if not _restore_rope(config):
        return False
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    return True


def normalise_saved_model(directory: str) -> list[str]:
    """Apply every compatibility fix to a staged directory; returns what was rewritten.

    Publishing goes through this rather than the individual functions, so a new publish path
    cannot apply one fix and quietly miss the other.
    """
    rewritten = []
    if normalise_tokenizer_config(directory):
        rewritten.append(TOKENIZER_CONFIG)
    if normalise_model_config(directory):
        rewritten.append(MODEL_CONFIG)
    return rewritten
