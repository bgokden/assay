import io
import json

from assay.records import parse_record, write_record
from assay.schema import Question


def test_parse_criteria_style_record():
    d = {
        "state": {"passage": "p", "question": "q"},
        "questions": {
            "answer": {
                "type": "choice",
                "instructions": "Which?",
                "criteria": {"a": "one", "b": "two"},
                "label": "b",
                "src": "mmlu",
            },
            "flag": {
                "type": "bool",
                "instructions": "Is it?",
                "criteria": {"true": "yes desc", "false": "no desc"},
                "label": False,
            },
            "level": {"type": "score", "instructions": "How?", "criteria": ["lo", "mid", "hi"], "label": 2},
        },
        "_meta": {"source": "suite"},
    }
    r = parse_record(d)
    by_name = {q.name: q for q in r.questions}
    assert by_name["answer"].label_index == 1
    assert by_name["answer"].source == "mmlu"
    assert by_name["flag"].question.yes == "yes desc"
    assert by_name["flag"].label_index == 1
    assert by_name["flag"].source == "suite"
    assert by_name["level"].label_index == 2
    assert by_name["level"].target == [0.0, 0.0, 1.0]


def test_write_then_parse_roundtrip_with_soft_target():
    q = Question(type="choice", instructions="Pick", options={"x": "ex", "y": None})
    buf = io.StringIO()
    write_record(buf, "state", {"pick": (q, "x", {"x": 0.7, "y": 0.3}, False)}, {"source": "t"})
    line = buf.getvalue()
    d = json.loads(line)
    assert d["questions"]["pick"]["answerable"] is False
    r = parse_record(d)
    lq = r.questions[0]
    assert lq.label_index == 0
    assert lq.target == [0.7, 0.3]
    assert lq.answerable is False
    assert lq.question.options == {"x": "ex", "y": None}


def test_noul_alias_is_read_as_bool():
    q = Question(type="noul", instructions="Is it?")
    assert q.type == "bool"
    assert q.keys == ["yes", "no"]
    r = parse_record({"state": "s", "questions": {"q": {"type": "noul", "instructions": "Is it?", "label": True}}})
    assert r.questions[0].question.type == "bool"
    assert r.questions[0].label_index == 0
