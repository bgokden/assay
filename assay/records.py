"""Dataset record format shared by data building, training and evaluation.

One JSON object per line:

    {
      "state": <str | object | list>,
      "questions": {
        "<name>": {
          "type": "bool" | "choice" | "score",   ("noul" is read as "bool")
          "instructions": str,
          "options": {key: description | null}     (choice)
          "levels": [str, ...]                      (score)
          "yes": str | null, "no": str | null       (bool, optional)
          "label": key | true | false | int         (the hard label)
          "target": {key: probability}              (optional soft label over keys)
          "answerable": bool                        (optional, default true)
        }
      },
      "meta": {"source": str, ...}
    }

Records written by other projects in the "criteria" style (kev-suites, Jev examples) are
converted on load.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from typing import Any

from assay.schema import Question


@dataclasses.dataclass
class LabeledQuestion:
    name: str
    question: Question
    label_index: int
    target: list[float]  # distribution over question.keys
    answerable: bool
    source: str


@dataclasses.dataclass
class Record:
    state: Any
    questions: list[LabeledQuestion]
    meta: dict[str, Any]


def _label_to_index(question: Question, label: Any) -> int:
    keys = question.keys
    if question.type == "bool":
        if isinstance(label, str):
            label = label.strip().lower() in ("true", "yes", "1")
        return 0 if bool(label) else 1
    if question.type == "score":
        return int(label)
    return keys.index(str(label))


def _question_from_criteria(q: dict[str, Any]) -> Question:
    """Convert a `criteria`-style question (kev-suites / Jev) to our schema."""
    t = q["type"]
    criteria = q.get("criteria")
    if t == "choice":
        options = criteria if isinstance(criteria, dict) else {str(k): None for k in criteria}
        return Question(type="choice", instructions=q["instructions"], options=options)
    if t == "score":
        return Question(type="score", instructions=q["instructions"], levels=list(criteria))
    yes = no = None
    if isinstance(criteria, dict):
        yes = criteria.get("true", criteria.get("yes"))
        no = criteria.get("false", criteria.get("no"))
    return Question(type="bool", instructions=q["instructions"], yes=yes, no=no)


def parse_question(name: str, q: dict[str, Any], default_source: str) -> LabeledQuestion:
    if "criteria" in q:
        question = _question_from_criteria(q)
    else:
        question = Question.from_dict(q)
    label_index = _label_to_index(question, q["label"])
    keys = question.keys
    if q.get("target"):
        target = [float(q["target"].get(k, 0.0)) for k in keys]
        total = sum(target)
        if total <= 0:
            raise ValueError(f"empty target for question {name}")
        target = [t / total for t in target]
    else:
        target = [0.0] * len(keys)
        target[label_index] = 1.0
    return LabeledQuestion(
        name=name,
        question=question,
        label_index=label_index,
        target=target,
        answerable=bool(q.get("answerable", True)),
        source=str(q.get("src") or q.get("source") or default_source),
    )


def parse_record(d: dict[str, Any]) -> Record:
    meta = d.get("meta") or d.get("_meta") or {}
    default_source = str(meta.get("source", "unknown"))
    questions = [parse_question(n, q, default_source) for n, q in d["questions"].items()]
    return Record(state=d["state"], questions=questions, meta=meta)


def read_records(path: str, limit: int | None = None) -> Iterator[Record]:
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                yield parse_record(json.loads(line))


def question_to_dict(q: Question) -> dict[str, Any]:
    d: dict[str, Any] = {"type": q.type, "instructions": q.instructions}
    if q.type == "choice":
        d["options"] = q.options
    elif q.type == "score":
        d["levels"] = q.levels
    else:
        if q.yes is not None:
            d["yes"] = q.yes
        if q.no is not None:
            d["no"] = q.no
    return d


def write_record(
    f,
    state: Any,
    questions: dict[str, tuple[Question, Any, dict[str, float] | None, bool]],
    meta: dict[str, Any],
) -> None:
    """questions: name -> (question, label, target or None, answerable)."""
    out_q = {}
    for name, (q, label, target, answerable) in questions.items():
        d = question_to_dict(q)
        d["label"] = label
        if target is not None:
            d["target"] = target
        if not answerable:
            d["answerable"] = False
        out_q[name] = d
    f.write(json.dumps({"state": state, "questions": out_q, "meta": meta}, ensure_ascii=False))
    f.write("\n")
