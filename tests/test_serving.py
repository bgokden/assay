import torch

from assay.schema import Question
from assay.serving import DirectRunner, runner_for

BOOL = Question(type="bool", instructions="Is it?")
CHOICE = Question(
    type="choice", instructions="Which?", options={"a": "first", "b": "second", "c": "third"}
)


class StubTokenizer:
    def __call__(self, text, truncation=False, max_length=None):
        return {"input_ids": list(range(len(str(text).split())))}


class StubModel:
    """Answers rows the way the small tiers do, and records the size of every forward pass."""

    temperature = 1.0

    def __init__(self) -> None:
        self.tokenizer = StubTokenizer()
        self.calls: list[int] = []

    def __call__(self, states, questions):
        self.calls.append(len(states))
        logits = torch.zeros(len(states), 4)
        for i, q in enumerate(questions):
            logits[i, : len(q.keys)] = torch.arange(len(q.keys), dtype=torch.float)
        return logits, torch.zeros(len(states))


def test_rows_are_regrouped_by_request():
    model = StubModel()
    runner = runner_for(model)
    assert isinstance(runner, DirectRunner)
    items = [runner.prepare("a state", [BOOL, CHOICE]), runner.prepare("another", [BOOL])]
    answers = runner.answer_batch(items, [[BOOL, CHOICE], [BOOL]])
    assert [len(a) for a in answers] == [2, 1]
    assert set(answers[0][0].probabilities) == {"yes", "no"}
    assert set(answers[0][1].probabilities) == {"a", "b", "c"}
    assert model.calls == [3]  # one pass for both requests


def test_a_large_request_is_split_into_bounded_passes():
    model = StubModel()
    runner = runner_for(model)
    runner.ROW_BUDGET = 4
    questions = [BOOL] * 10
    item = runner.prepare("a state", questions)
    answers = runner.answer_batch([item], [questions])
    assert len(answers[0]) == 10
    assert model.calls == [4, 4, 2]
    assert runner.passes([questions]) == 3


def test_the_size_of_a_request_grows_with_the_state_and_the_questions():
    runner = runner_for(StubModel())
    one = runner.size(runner.prepare("three word state", [BOOL]))
    many = runner.size(runner.prepare("three word state", [BOOL, BOOL, BOOL]))
    assert many > one
