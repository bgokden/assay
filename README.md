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
6. **Abstention with a stated error rate.** `assay.conformal` fits two thresholds per question
   type on the same calibration split: a prediction-set threshold (split conformal, the set
   contains the label with probability at least 1 - alpha) and an act threshold on the top
   probability (one-sided binomial tests over a grid with a Bonferroni correction: among
   answers above it, the error rate is at most alpha at confidence 1 - delta). The server
   returns them as `act` and `set` on every answer when `conformal.json` is present. The
   evidence head says "the state does not say"; this layer says "the model does not know".

See `docs/research.md` for the literature and the community landscape this builds on, and
`docs/roadmap.md` for what comes next and why.

## Results

Models: [Berk/assay-27b](https://huggingface.co/Berk/assay-27b) (adapter + evidence head over
a 4-bit Qwen3.8-27B), [Berk/assay-4b](https://huggingface.co/Berk/assay-4b),
[Berk/assay-1.7b](https://huggingface.co/Berk/assay-1.7b) and
[Berk/assay-0.6b](https://huggingface.co/Berk/assay-0.6b) (merged weights, adapter, evidence
head and model card in each repository). Cells are accuracy / Brier / ECE, single seed.

| model | seen tasks (dev, n=5513) | unseen tasks (holdout, n=2020) | kev transfer-v4 (n=764) |
|---|---|---|---|
| Qwen3-0.6B-Base, untrained | - | 0.586 / 0.498 / 0.072 | 0.527 / 0.541 / 0.109 |
| **assay-0.6b** (data v4) | 0.705 / 0.391 / 0.030 | 0.704 / 0.397 / 0.037 | 0.636 / 0.499 / 0.124 |
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

**Soft versus hard targets** (1.7B, same data, two seeds each): soft targets lower raw ECE on
unseen tasks (0.047 and 0.057 against 0.060 and 0.068), confident errors on the transfer suite
(6.5% and 5.9% against 7.7% and 6.8%) and scaled holdout Brier (0.334 and 0.328 against 0.338
and 0.341), in the same direction for both seeds; accuracy and post-scaling ECE are within
noise. At this scale the readout design and a single temperature do most of the calibration
work; soft targets are a modest, consistent extra.

**Abstention** (`assay.conformal`, alpha 0.1, delta 0.05, fitted on the seen-task calibration
split): assay-4b acts on 88% of bool and 75% of choice questions there with 7.4% and 8.0%
error among the acted-on answers, as the construction guarantees. On the eleven unseen tasks
the same thresholds act on 75% / 74% with 10.2% / 8.0% error and the prediction sets cover
the label 86% / 93% of the time; on the transfer suite the error is 14%. Task shift costs a
few points over the stated rate, less for choice than for bool; the report prints both so
the number you quote is the one for your data. Score questions get prediction sets (about
two levels wide) but no act threshold at this alpha, because exact-level accuracy is the
wrong error notion for ordinal answers.

**Seed noise** (two seeds each): 1.7B runs reproduce to 0.3 points on every split; two 4B runs
differ by 0.5-0.7 points on seen and unseen tasks and 1.1 points on the 764-item transfer
suite (0.784 and 0.795). Differences under about 1.5 points on the transfer suite are noise.

On the transfer suite assay-27b is within 1.5 points of Jev's reported 0.857 and 0.018 Brier
of its 0.211, after 4.5 hours of QLoRA on one RTX 5090. Per family it is above Jev on dates
(0.95 vs 0.93), QNLI and emotion, at parity on authorization, and below on MMLU (0.78 vs 0.90),
PAWS and offensive language.

**Latency** (RTX 5090, plain transformers): 4B in bf16, 23 ms for one question, 57 ms for 24
questions packed over the same state (548 ms as separate requests). 27B in 4-bit, 110 ms for
one question; hybrid backbones are not packed yet, so 24 questions run as a batch in 750 ms.

## Encoder tier: decisions without a language model

`assay.compiled` is a second architecture with the same interface and no LM: an encoder
(gte-modernbert-base, 149M) turns the state into token embeddings once and compiles each
question once into query vectors (from the instruction) and option vectors plus option token
encodings (from the option text). A decision is cross-attention from the queries over the
state tokens, a bilinear score against each option vector, and a late-interaction term (each
option token's best cosine match among the state tokens, ColBERT-style). Options are scored by
their own content, so there is no label alphabet and answers are order-invariant by
construction; compiled questions can be cached across states.

| model | seen tasks (dev) | unseen tasks (holdout) | kev transfer-v4 |
|---|---|---|---|
| untrained cosine baseline | 0.417 / 0.665 / 0.116 | 0.541 / 0.563 / 0.067 | 0.514 / 0.585 / 0.075 |
| cross-encoder, same backbone (one pass per option) | 0.670 / 0.422 / 0.027 | 0.619 / 0.479 / 0.025 | 0.527 / 0.531 / 0.061 |
| **assay-compiled-base** (compiled + late interaction) | 0.668 / 0.442 / 0.037 | 0.606 / 0.494 / 0.061 | 0.542 / 0.572 / 0.095 |

It sits at the level of the untrained 1.7B decoder on unseen tasks: strong on single-text
classification (topic, sentiment, spam at 0.9+), near chance on knowledge (MMLU 0.25) and
multi-step reasoning. Its point is cost: on a CPU with 8 threads, encoding a state takes
30 ms, compiling six questions 90 ms once, and then deciding all six takes 3 ms
(`scripts/bench_compiled.py`). Published as
[Berk/assay-compiled-base](https://huggingface.co/Berk/assay-compiled-base).

That 0.606 is where the tier stopped, and the attempts to move it are worth reporting as
negative results: a deeper reader whose option tokens self-attend and cross-attend into the
state reached 0.576; 1.2M teacher-labelled examples cost 4 points mixed into the task data and
6 points as a two-stage pretrain; a 0.6B decoder's hidden states in place of the encoder gave
0.553. Meanwhile a cross-encoder with full joint attention on the same backbone reaches only
0.619, so reader capacity was never the constraint. The clearest measurement: the same 0.6B
weights score **0.704** on unseen tasks when the answer is read from their own next-token
distribution and **0.553** when they are a feature extractor under a learned reader. For this
family of tasks, reading the answer out of a language model beats learning a head on top of
one; `assay-0.6b` is the better small model, and the encoder tier is the CPU option. Details
and per-family numbers in `docs/roadmap.md`.

## Install and run

```bash
uv sync
uv run pytest                                             # unit tests (downloads Qwen3-0.6B)
uv run python -m assay.server --model Berk/assay-4b       # POST /v1/decide on :8000
```

`assay.backends.sglang` serves the decoder tier through an SGLang deployment instead of local
transformers: `/generate` returns `token_ids_logprob` and `return_hidden_states` in one
request, so the readout and the evidence head both come from one forward pass, and
RadixAttention reuses a state prefix across separate questions. Check a deployment with
`verify_against_local` before trusting it - SGLang's response layout is not pinned by a
published schema.

The server also accepts the System One style payload that other open decision models use
(`/v1/systemone`, questions typed `choice`/`noul`/`score` with options under `criteria`), so a
client written for that interface works unchanged; `POST /v1/decide_graph` walks a decision
tree in a single forward pass; `/health`, `/metrics` and `/v1/stats` are for operations.

```bash
curl -s localhost:8000/v1/decide -H 'content-type: application/json' -d '{
  "state": "My card was charged twice for order A-104.",
  "questions": {
    "refund": {"type": "bool", "instructions": "Does the customer ask for money back?"},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "options": {"billing": "Charges and refunds", "technical": "Bugs and outages"}}
  }}'
```

## Train on your own data

One JSON file describes the data, the tier and the hyper-parameters; the pipeline runs
training, temperature calibration, evaluation and conformal abstention as separate processes
and skips any stage whose output is already there.

```bash
uv run python examples/support_data.py                                  # a dataset in the record format
uv run python -m assay.pipeline --config examples/pipelines/support-encoder.json --dry-run
uv run python -m assay.pipeline --config examples/pipelines/support-encoder.json
uv run python -m assay.server --model runs/example-support-encoder      # the same server, your model
```

```json
{
  "name": "support-encoder",
  "tier": "encoder",
  "base_model": "Alibaba-NLP/gte-modernbert-base",
  "data": "examples/data/support",
  "out": "runs/example-support-encoder",
  "train": {"epochs": 2, "lr": 5e-5, "batch_size": 16},
  "conformal": {"alpha": 0.1, "delta": 0.05}
}
```

`tier` is `decoder`, `encoder` or `seq2seq`; whatever is under `train` becomes flags for that
tier's trainer, so every option a trainer has is available without the pipeline knowing about
it. The training data is one JSON object per line: a state and the typed questions over it
with their labels (`assay/records.py` documents the fields). [examples/](examples/) has the
runnable scripts for the model API, decision graphs and both server interfaces;
[docs/models.md](docs/models.md) is the table of every published model.

## Reproduce

```bash
uv run python -m assay.data.build --out data/v2          # 66 public tasks + policy cases -> jsonl
uv run python -m assay.data.distill --teacher runs/assay-27b --train data/v2/train.jsonl \
  --out data/distill --generic 0                          # hard policy and date generators
cat data/v2/train.jsonl data/distill/policy_hard.jsonl data/distill/dates.jsonl > data/v4/train.jsonl
scripts/run_experiment.sh Qwen/Qwen3-4B-Base runs/assay-4b data/v4 --checkpoint-every 500
uv run python -m assay.conformal --model runs/assay-4b --alpha 0.1  # act/set thresholds
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

## Released data

[Berk/assay-synthetic](https://huggingface.co/datasets/Berk/assay-synthetic) holds the parts
of the training mix that this repository generates: 50k train and 2k test items for each of
`policy` (rules composed with AND/OR/NOT/IF-ELSE over stated facts), `policy_hard` (nesting
depth 3, predicates over text, and cases where a needed fact is missing so the answer is
"cannot be determined") and `dates` (grace periods with dates in ISO, long, short, weekday
and relative forms), plus the 90-question rubric bank used for distillation. Labels are
computed in code, so they are exact and the test splits are a reliable measure of rule
application and date reasoning. Apache-2.0.

The public datasets Assay also trains on are not redistributed: they keep their own licences
(several non-commercial) and are listed below and in `docs/datasets.md`.

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
