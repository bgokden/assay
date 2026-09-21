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

model = AssayModel.from_pretrained("Berk/assay-4b")

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
        "is_urgent": Question(type="bool", instructions="Does the message convey urgency?"),
    },
)
answers["department"].argmax        # "technical"
answers["department"].probabilities # {"billing": 0.05, "technical": 0.93, "sales": 0.02}
answers["frustration"].score        # 1.2  (probability-weighted level)
answers["is_urgent"].p_true           # 0.97
answers["is_urgent"].evidence       # 0.99 (the state supports answering this)
```

## Question types

| type | asks | answer fields |
|---|---|---|
| `bool` | is this statement true? | `p_true` = P(yes) |
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
   the question, trained on passage-swapped reading-comprehension pairs and on synthetic policy
   cases where a fact the rule needs is missing.
5. **One global temperature** fitted on a calibration split of the training tasks and applied
   unchanged to unseen tasks.

See `docs/research.md` for the literature and the community landscape this builds on, and
`docs/roadmap.md` for what comes next and why.

## Results

Models: [Berk/assay-27b](https://huggingface.co/Berk/assay-27b) (adapter + evidence head over
a 4-bit Qwen3.8-27B), [Berk/assay-4b](https://huggingface.co/Berk/assay-4b) and
[Berk/assay-1.7b](https://huggingface.co/Berk/assay-1.7b) (merged weights, adapter, evidence
head and model card in each repository). Cells are accuracy / Brier / ECE, single seed.

| model | seen tasks (dev, n=5513) | unseen tasks (holdout, n=2020) | kev transfer-v4 (n=764) |
|---|---|---|---|
| Qwen3-1.7B-Base, untrained | 0.542 / 0.551 / 0.077 | 0.623 / 0.456 / 0.070 | 0.588 / 0.495 / 0.116 |
| **assay-1.7b** | 0.740 / 0.355 / 0.035 | 0.752 / 0.334 / 0.024 | 0.670 / 0.436 / 0.115 |
| Qwen3-4B-Base, untrained | 0.640 / 0.452 / 0.032 | 0.740 / 0.356 / 0.042 | 0.707 / 0.383 / 0.051 |
| **assay-4b** (data v4) | 0.791 / 0.287 / 0.027 | 0.803 / 0.271 / 0.023 | 0.784 / 0.302 / 0.061 |
| Qwen3.8-27B (4-bit), untrained | - | 0.773 / 0.312 / 0.060 | 0.793 / 0.318 / 0.077 |
| **assay-27b** | 0.834 / 0.243 / 0.040 | 0.842 / 0.221 / 0.040 | **0.842 / 0.229 / 0.041** |

Trained rows are after temperature scaling (fitted on seen-task calibration data only); raw
numbers are in the model cards and `runs/*/eval-*.json`. "Unseen tasks" are eleven datasets
never trained on. The transfer suite is public and its sources are excluded from training; for
reference, its authors report Kev-8B at 0.774 / 0.339 and Jev at 0.857 / 0.211 on the same
items (their Brier definition may differ from ours).

What the training changed on the transfer suite for the 4B, untrained -> trained (data v4):
policy composition families 0.34-0.66 -> 0.75-0.88, deadline 0.30 -> 0.975, authorization
1.00 -> 1.00 (Brier 0.07 -> 0.00), MMLU 0.647 -> 0.690, QNLI 0.89 -> 0.90, tweet offensive
0.68 -> 0.70, SciQ 0.95 -> 0.95, emotion 0.57 -> 0.55, PAWS 0.76 -> 0.74. Dates and rules come
from two exact-label synthetic generators of our own (`assay.data.dates`, `assay.data.policy`);
40k extra examples soft-labelled by the 27B did not move unseen tasks and were dropped
(see `docs/roadmap.md`).

**Soft versus hard targets** (1.7B, same data and seed): soft targets lower raw ECE on unseen
tasks from 0.060 to 0.047 and confident errors from 7.7% to 6.5% on the transfer suite, but
after one fitted temperature the two are within noise (holdout Brier 0.334 vs 0.338). At this
scale the readout design and a single temperature do most of the calibration work; soft
targets are a modest, consistent extra.

On the transfer suite assay-27b is within 1.5 points of Jev's reported 0.857 and 0.018 Brier
of its 0.211, after 4.5 hours of QLoRA on one RTX 5090. Per family it is above Jev on dates
(0.95 vs 0.93), QNLI and emotion, at parity on authorization, and below on MMLU (0.78 vs 0.90),
PAWS and offensive language.

**Latency** (RTX 5090, plain transformers): 4B in bf16, 23 ms for one question, 57 ms for 24
questions packed over the same state (548 ms as separate requests). 27B in 4-bit, 110 ms for
one question; hybrid backbones are not packed yet, so 24 questions run as a batch in 750 ms.

## Install and run

```bash
uv sync
uv run pytest                                             # unit tests (downloads Qwen3-0.6B)
uv run python -m assay.server --model Berk/assay-4b       # POST /v1/decide on :8000
```

```bash
curl -s localhost:8000/v1/decide -H 'content-type: application/json' -d '{
  "state": "My card was charged twice for order A-104.",
  "questions": {
    "refund": {"type": "bool", "instructions": "Does the customer ask for money back?"},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "options": {"billing": "Charges and refunds", "technical": "Bugs and outages"}}
  }}'
```

## Reproduce

```bash
uv run python -m assay.data.build --out data/v2          # 66 public tasks + policy cases -> jsonl
uv run python -m assay.data.distill --teacher runs/assay-27b --train data/v2/train.jsonl \
  --out data/distill --generic 0                          # hard policy and date generators
cat data/v2/train.jsonl data/distill/policy_hard.jsonl data/distill/dates.jsonl > data/v4/train.jsonl
scripts/run_experiment.sh Qwen/Qwen3-4B-Base runs/assay-4b data/v4 --checkpoint-every 500
uv run python -m assay.publish --run runs/assay-4b --repo <user>/assay-4b
```

The 4B run takes 47 minutes on one RTX 5090 (LoRA r=16, lr 5e-5, batch 8, one epoch over
54,350 examples); the 1.7B takes 23 minutes; the 27B takes 4.5 hours with
`--quant 4bit --batch-size 4 --grad-accum 2 --eval-batch-size 4 --max-state-tokens 1024`
(evaluate it with `scripts/eval_all.sh runs/assay-27b data/v2 4 1024`).

`data/suites/kev-transfer-v4-dev.jsonl` is the public transfer suite from
[jaredpalmer/kev-suites](https://huggingface.co/datasets/jaredpalmer/kev-suites); none of its
sources are in Assay's training data.

## Limitations

Text only. No arithmetic, counting, date comparison or multi-hop reasoning: one forward pass
cannot do them, so keep those in code. Accuracy drops as unrelated state grows; filter first.
The evidence head is trained on swapped-passage and missing-fact negatives, which is a coarse
notion of "unsupported"; treat it as a first filter, not a proof. Probabilities are calibrated
in aggregate on the distributions above, which says nothing about any single answer or about
your data: check calibration on your own labels before acting on thresholds.

## Data licences

Training and evaluation sources with their declared licences are listed in
[docs/datasets.md](docs/datasets.md). Several carry non-commercial or research-only terms;
check them before commercial use of the weights.

## Relationship to other work

Assay is an independent project. Jev and System One are names of TypeSafe AI's products and
are mentioned only to describe and compare; kev-suites is Jared Palmer's evaluation data.
Assay is not affiliated with or endorsed by either.

## License

Apache-2.0.
