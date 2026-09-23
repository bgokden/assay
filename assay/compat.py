"""Keep saved artifacts loadable by the transformers version other people have.

transformers 5 renamed the tokenizer config's `additional_special_tokens` to
`extra_special_tokens` and kept writing it as a list. transformers 4 reads
`extra_special_tokens` as a mapping, so a tokenizer saved by 5 raises
`AttributeError: 'list' object has no attribute 'keys'` on 4 -- which is still the version
most people have, and the one SGLang and llama.cpp's converter pin.

Writing the list under the old name loads on both: 4 understands
`additional_special_tokens`, and so does 5.
"""

from __future__ import annotations

import json
import os

TOKENIZER_CONFIG = "tokenizer_config.json"


def normalise_tokenizer_config(directory: str) -> bool:
    """Rewrite a saved tokenizer config so both transformers generations load it.

    Returns True when something was changed. Safe to call on a directory without one.
    """
    path = os.path.join(directory, TOKENIZER_CONFIG)
    if not os.path.exists(path):
        return False
    with open(path) as f:
        config = json.load(f)
    extra = config.get("extra_special_tokens")
    if not isinstance(extra, list):
        return False
    config.pop("extra_special_tokens")
    existing = config.get("additional_special_tokens") or []
    merged = list(dict.fromkeys([*existing, *extra]))
    config["additional_special_tokens"] = merged
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    return True
