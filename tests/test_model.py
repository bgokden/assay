import os

import pytest
import torch

from assay.encoding import encode, identity_order
from assay.model import AssayModel
from assay.schema import Question

BASE = os.environ.get("ASSAY_TEST_BASE", "Qwen/Qwen3-0.6B-Base")
DEVICE = os.environ.get("ASSAY_TEST_DEVICE", "cuda")

pytestmark = pytest.mark.skipif(
    DEVICE == "cuda" and not torch.cuda.is_available(), reason="needs a GPU"
)


@pytest.fixture(scope="module")
def model():
    return AssayModel.from_base(BASE, lora_r=None, dtype=torch.float32, device=DEVICE)


STATE = (
    "Hi, I've been trying to connect my Stripe account for 3 days and the integration keeps "
    "failing. I'm losing sales. Please help ASAP."
)
QUESTIONS = {
    "department": Question(
        type="choice",
        instructions="Which team should handle this?",
        options={
            "billing": "Payment or subscription issues",
            "technical": "Bugs or integration problems",
            "sales": "Pricing or account questions",
        },
    ),
    "frustration": Question(
        type="score",
        instructions="How frustrated does the customer appear?",
        levels=["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"],
    ),
    "is_urgent": Question(type="bool", instructions="Does the message convey urgency?"),
}


def test_packed_equals_separate(model):
    packed_all = model.answer(STATE, QUESTIONS)
    for name, q in QUESTIONS.items():
        single = model.answer(STATE, {name: q})[name]
        for key in q.keys:
            assert packed_all[name].probabilities[key] == pytest.approx(
                single.probabilities[key], abs=2e-4
            )


def test_zero_shot_is_sensible(model):
    answers = model.answer(STATE, QUESTIONS)
    assert answers["department"].argmax == "technical"
    assert answers["is_urgent"].p_true > 0.5
    assert 0.0 <= answers["frustration"].score <= 2.0
    for a in answers.values():
        assert abs(sum(a.probabilities.values()) - 1.0) < 1e-5
        assert 0.0 <= a.confidence <= 1.0


def test_option_order_does_not_change_keys(model):
    q = QUESTIONS["department"]
    packed_a = encode(model.tokenizer, model.alphabet, STATE, [q], orders=[identity_order(q)])
    packed_b = encode(model.tokenizer, model.alphabet, STATE, [q], orders=[[2, 0, 1]])
    a = model.answer_packed([packed_a], [[q]])[0][0]
    b = model.answer_packed([packed_b], [[q]])[0][0]
    assert list(a.probabilities) == list(b.probabilities) == ["billing", "technical", "sales"]
    assert a.argmax == b.argmax == "technical"


def test_save_and_load_roundtrip(model, tmp_path):
    model.save_pretrained(str(tmp_path))
    assert (tmp_path / "assay_head.safetensors").exists()
    assert (tmp_path / "assay_config.json").exists()


@pytest.fixture(scope="module")
def content_model():
    return AssayModel.from_base(
        BASE, lora_r=None, dtype=torch.float32, device=DEVICE, content_term=True
    )


def test_content_term_starts_neutral(model, content_model):
    """Zero-initialised content projection: answers equal the plain label readout."""
    plain = model.answer(STATE, QUESTIONS)
    with_term = content_model.answer(STATE, QUESTIONS)
    for name, q in QUESTIONS.items():
        for key in q.keys:
            assert with_term[name].probabilities[key] == pytest.approx(
                plain[name].probabilities[key], abs=1e-5
            )


def test_content_term_scores_each_option_line(content_model):
    torch.manual_seed(0)
    with torch.no_grad():
        content_model.content_proj.weight.normal_(std=0.05)
    q = QUESTIONS["department"]
    packed = encode(
        content_model.tokenizer, content_model.alphabet, STATE, [q, QUESTIONS["is_urgent"]]
    )
    from assay.encoding import collate

    batch = collate([packed], content_model.tokenizer.pad_token_id).to(content_model.device)
    hidden = torch.zeros(
        1, len(packed), content_model.evidence_head.in_features, device=content_model.device
    )
    for k, (a, b) in enumerate(packed.questions[0].option_spans):
        hidden[0, a:b, k] = 1.0  # option k's line is a one-hot direction
    decision = torch.zeros(2, hidden.shape[-1], device=content_model.device)
    scores = content_model._content_scores(hidden, decision, batch)
    assert scores.shape == (2, 3)
    assert torch.all(scores[1] == 0.0)  # a bool without descriptions has no option lines
    with torch.no_grad():
        content_model.content_proj.weight.zero_()


def test_content_term_roundtrip(content_model, tmp_path):
    with torch.no_grad():
        content_model.content_proj.weight.fill_(0.01)
    content_model.save_pretrained(str(tmp_path))
    loaded = AssayModel.from_pretrained(str(tmp_path), dtype=torch.float32, device=DEVICE)
    assert loaded.content_proj is not None
    assert torch.allclose(loaded.content_proj.weight.cpu(), content_model.content_proj.weight.cpu())
    before = content_model.answer(STATE, QUESTIONS)
    after = loaded.answer(STATE, QUESTIONS)
    for name, q in QUESTIONS.items():
        for key in q.keys:
            assert after[name].probabilities[key] == pytest.approx(
                before[name].probabilities[key], abs=1e-4
            )
    with torch.no_grad():
        content_model.content_proj.weight.zero_()


def test_load_model_honours_a_cpu_request(saved_model, monkeypatch):
    """The decoder tier used to drop `device`, so a caller asking for cpu got cuda -- and on a
    machine without a GPU, an unexplained "No CUDA GPUs are available" from inside a loader it
    had just told to use the cpu. That is the only tier where this was wrong, and it is the
    tier every published decoder model uses."""
    import torch

    from assay.evaluate import load_model

    captured = {}
    original = AssayModel.from_pretrained

    def spy(path, *args, **kwargs):
        captured["device"] = kwargs.get("device")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(AssayModel, "from_pretrained", staticmethod(spy))
    model = load_model(saved_model, dtype=torch.float32, device="cpu")
    assert captured["device"] == "cpu"
    assert next(model._backbone().parameters()).device.type == "cpu"


def test_adapter_at_the_repository_root_resolves_to_the_repository():
    """An adapter-only repository is a PEFT repository, and the Hub counts its downloads from
    `adapter_config.json` at the root exactly. Published under `adapter/` it reports no
    downloads at all, forever -- which is what assay-8b and assay-27b did until they were
    restructured. `adapter: "."` is how the config records a root adapter."""
    from assay.model import adapter_source

    assert adapter_source("/models/assay-8b", ".") == "/models/assay-8b"
    assert adapter_source("/models/assay-8b", "") == "/models/assay-8b"
    assert adapter_source("/models/run", "adapter") == "/models/run/adapter"
