import json

import pytest

from assay.apply import load_questions, read_states


def test_reads_bare_states_and_records(tmp_path):
    path = tmp_path / "states.jsonl"
    path.write_text(
        json.dumps("a plain string state")
        + "\n"
        + json.dumps({"channel": "email", "message": "an object state"})
        + "\n"
        + json.dumps({"state": "from a record", "meta": {"id": "set/0"}, "questions": {}})
        + "\n\n"  # blank lines are skipped
    )
    got = list(read_states(str(path)))
    assert [identifier for _, identifier in got] == [0, 1, "set/0"]
    assert got[1][0]["message"] == "an object state"
    assert got[2][0] == "from a record"


def test_limit_stops_early(tmp_path):
    path = tmp_path / "states.jsonl"
    path.write_text("\n".join(json.dumps(f"state {i}") for i in range(5)))
    assert len(list(read_states(str(path), limit=2))) == 2


def test_questions_are_validated(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps({"q": {"type": "noul", "instructions": "Short?"}}))
    questions = load_questions(str(path))
    assert questions["q"].type == "bool"  # the other tools' name for it

    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    with pytest.raises(ValueError, match="non-empty"):
        load_questions(str(empty))
