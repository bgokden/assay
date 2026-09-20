# Assay

Calibrated typed decisions from one forward pass of a language model. No text is generated.

You send a **state** (text, an object, or a list) and a set of named **typed questions**.
Assay returns a probability distribution over the options of every question, a confidence,
and an **evidence** score that says whether the state contains what is needed to answer.
Every question is evaluated as an isolated branch over the shared state, in a single prefill,
so a request with twenty questions costs about the same as a request with one.


```python
from assay.model import AssayModel
from assay.schema import Question

model = AssayModel.from_pretrained("runs/assay-4b")

answers = model.answer(
    state="Hi, I've been trying to connect my Stripe account for 3 days and the integration "
          "keeps failing. I'm losing sales. Please help ASAP.",
    questions={
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
            levels=["Calm, just stating facts", "Frustrated but civil", "Very angry"],
        ),
        "is_urgent": Question(type="noul", instructions="Does the message convey urgency?"),
    },
)
answers["department"].argmax        # "technical"
answers["department"].probabilities # {"billing": 0.05, "technical": 0.93, "sales": 0.02}
answers["frustration"].score        # 1.2  (probability-weighted level)
answers["is_urgent"].noul           # 0.97
answers["is_urgent"].evidence       # 0.99 (the state supports answering this)
```

## Question types

| type | asks | answer fields |
|---|---|---|
| `noul` | is this statement true? | `noul` = P(yes) |
| `choice` | which one of these options? (2..255) | `choice`, `probabilities`, `confidence` |
| `score` | where on this ordered scale? (2..10 levels) | `score` (expected level), `level`, `probabilities`, `confidence` |

All answers also carry `evidence`. `confidence` is `(p_max - 1/K) / (1 - 1/K)`: 0 for a uniform
distribution, 1 for a one-hot. Option descriptions are free text; write them as situations
("Broken, but a workaround exists"), not adjectives.

## How it works

1. **Readout from the base model's own logits.** The prompt lists the options with letter
   labels and ends with `Answer:`. The next-token distribution at that position, restricted to
   the label tokens, is the answer. A base LM is already well calibrated in this format
   (Kadavath et al. 2022); fine-tuning nudges it rather than replacing it with a fresh head.
2. **Isolated branches over a shared state.** State tokens are encoded once. Each question is
   a block that attends to the state and to itself, never to sibling questions, with position
   ids restarted after the state. Packed and separate requests give identical probabilities.
3. **Trained for calibration.** LoRA (r=16, lr 5e-5, one epoch) with cross-entropy against
   *soft* targets: human label distributions where datasets have them (civil_comments,
   measuring-hate-speech, GoEmotions raters, DynaSent, SHP vote ratios, STS-B), SORD-smoothed
   levels for ordinal questions, one-hot otherwise. Log loss is a proper scoring rule; the
   targets are what make it calibrated. Choice options are shuffled per example so label
   letters carry no positional prior.
4. **Evidence head.** A linear head on the decision token predicts whether the state supports
   the question, trained on passage-swapped reading-comprehension pairs.
5. **One global temperature** fitted on a calibration split of the training tasks and applied
   unchanged to unseen tasks.

See `docs/research.md` for the literature and the community landscape this builds on.

## Results

Filled in from `runs/*/eval-*.json`; see the model cards on Hugging Face.

## Install and run

```bash
uv sync
uv run pytest                                             # unit tests (downloads Qwen3-0.6B)
uv run python -m assay.server --model runs/assay-4b       # POST /v1/decide on :8000
```

```bash
curl -s localhost:8000/v1/decide -H 'content-type: application/json' -d '{
  "state": "My card was charged twice for order A-104.",
  "questions": {
    "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "options": {"billing": "Charges and refunds", "technical": "Bugs and outages"}}
  }}'
```

## Reproduce

```bash
uv run python -m assay.data.build --out data/v1          # 66 public tasks -> jsonl
scripts/run_experiment.sh Qwen/Qwen3-4B-Base runs/assay-4b data/v1
```

`data/suites/kev-transfer-v4-dev.jsonl` is the public transfer suite from
[jaredpalmer/kev-suites](https://huggingface.co/datasets/jaredpalmer/kev-suites); none of its
sources are in Assay's training data.

## Limitations

Text only. No arithmetic, counting, date comparison or multi-hop reasoning: one forward pass
cannot do them, so keep those in code. Accuracy drops as unrelated state grows; filter first.
The evidence head is trained on swapped-passage negatives, which is a coarse notion of
"unsupported"; treat it as a first filter, not a proof.

## License

Apache-2.0.
