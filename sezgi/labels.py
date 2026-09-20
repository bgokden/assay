"""Option label alphabet.

Every option is rendered with a short label ("A", "B", ..., "AA", ...) and the answer is read
from the model's next-token distribution over those labels. A label is usable only if the
tokenizer encodes " <label>" (leading space, as it appears after "Answer:") as exactly one token.
"""

from __future__ import annotations

import string

from transformers import PreTrainedTokenizerBase

YES = " yes"
NO = " no"


class LabelAlphabet:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, size: int = 255) -> None:
        self.labels: list[str] = []
        self.token_ids: list[int] = []
        candidates = list(string.ascii_uppercase)
        candidates += [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
        for label in candidates:
            ids = tokenizer.encode(" " + label, add_special_tokens=False)
            if len(ids) == 1:
                self.labels.append(label)
                self.token_ids.append(ids[0])
            if len(self.labels) >= size:
                break
        if len(self.labels) < size:
            raise ValueError(
                f"tokenizer only provides {len(self.labels)} single-token labels, need {size}"
            )
        yes = tokenizer.encode(YES, add_special_tokens=False)
        no = tokenizer.encode(NO, add_special_tokens=False)
        if len(yes) != 1 or len(no) != 1:
            raise ValueError("tokenizer must encode ' yes' and ' no' as single tokens")
        self.yes_id = yes[0]
        self.no_id = no[0]

    def __len__(self) -> int:
        return len(self.labels)
