# Roadmap

Written 2026-09-21 after the first release (assay-1.7b, assay-4b) and the backbone probes.
Each item names the evidence behind it, the expected gain, and what would make us drop it.

## Where we are

| model | seen tasks (dev) | unseen tasks (holdout) | public transfer suite |
|---|---|---|---|
| assay-1.7b | 0.740 / 0.355 / 0.035 | 0.752 / 0.334 / 0.024 | 0.670 / 0.436 / 0.115 |
| assay-4b (data v2) | 0.776 / 0.304 / 0.037 | 0.798 / 0.280 / 0.021 | 0.770 / 0.307 / 0.067 |
| assay-4b (data v4, published) | 0.791 / 0.287 / 0.027 | 0.803 / 0.271 / 0.023 | 0.784 / 0.302 / 0.061 |
| assay-27b (QLoRA on Qwen3.8-27B) | 0.834 / 0.243 / 0.040 | 0.842 / 0.221 / 0.040 | 0.842 / 0.229 / 0.041 |
| Jev (third-party run) | - | - | 0.857 / 0.211 / - |

Cells: accuracy / Brier / ECE after temperature scaling.

The gap to Jev on the transfer suite (8.7 points for the 4B) decomposes into knowledge
(MMLU, ~2.9 points), date arithmetic (~1.7), offensive-language style (~1.2), rule
composition (~0.8), SciQ (~0.8), paraphrase (~0.6). A third is model size, a fifth is a
capability our one-pass setup lacked, the rest is data mix.

Backbone ceilings measured with the untrained readout (transfer accuracy / Brier / ECE):
Qwen3-4B 0.707 / 0.383 / 0.051, Qwen3-8B 0.721 / 0.366 / 0.067, Qwen3-14B (8-bit) 0.737 /
0.358 / 0.052, Qwen3-30B-A3B 0.751 / 0.344 / 0.024, Qwen3.5-4B 0.704, Qwen3.5-9B 0.728,
Qwen3.8-27B (4-bit, instruct) 0.793 / 0.318 / 0.077. Dense scaling inside a series is slow;
the newest 27B is a different tier, already above every model we trained, and its weaknesses
(overconfidence, untrained rule composition) are what the recipe fixes.

## Plan, in order

### 1. Finish assay-27b (done 2026-09-21)

Result: transfer 0.842 / 0.229 / 0.041, holdout 0.842 / 0.221; MMLU 0.78, dates 0.95,
authorization 1.00. Published as adapter plus head at Berk/assay-27b. Remaining gap to Jev
is knowledge (MMLU) and two style families (PAWS, offensive), 1.5 points overall.

QLoRA on Qwen3.8-27B: 4-bit base, LoRA r=16, lr 5e-5, batch 4 x 2 accumulation, one epoch,
checkpoint every 500 steps with automatic resume. Fit its temperature. Publish adapter plus
evidence head (no merged upload; users load the 4-bit base).

Expected: 0.83-0.85 on the transfer suite, calibration in line with the smaller models.
Drop if: it does not beat assay-4b on the unseen-task holdout after scaling.

### 2. Distillation set from the 27B (run 2026-09-21; partly dropped)

Result on a 4B retrained with 40k teacher-labelled generic questions plus 5k hard policy cases
and 6k date cases (data v3): transfer 0.791 / 0.284 / 0.050 (from 0.770 / 0.307 / 0.067),
holdout 0.794 / 0.287 / 0.036 (from 0.798 / 0.280 / 0.021). The gain is entirely from the
exact-label synthetic families: deadline 0.60 -> 0.975, composition +3 to +12 points. The
teacher-labelled generic data moved nothing on unseen tasks and drifted knowledge and style
families slightly down. Conclusion: at 4B the generic tasks are capacity-limited; targeted
skill families with exact labels are the lever. Kept: `assay.data.policy` (hard mode) and
`assay.data.dates`. Dropped: generic teacher labels for decoder training (they may still
serve the compiled-function tier, which is data-limited by construction). A 4B retrained on
v2 plus the two generators only (data v4) gains on both splits: holdout 0.803 / 0.271 / 0.023,
transfer 0.784 / 0.302 / 0.061, deadline 0.975; it replaced the published assay-4b.

The 27B labels about 100k new inputs with its temperature-scaled probabilities: harder policy
cases (deeper rule nesting, more predicate types), date cases (varied formats, thresholds),
knowledge multiple choice, and unlabeled text rendered with our rubrics. Human soft labels
keep priority where they exist; teacher distributions fill the rest. Targets are the scaled
probabilities, never raw logits, so the teacher's overconfidence is not distilled.

Why first: every smaller tier is data-limited on exactly the families where the gap lives,
and the teacher is about to exist. Cost: a few hours of 27B inference.
Drop if: a 4B retrained on it does not gain on the holdout (then the small models are
capacity-limited, not data-limited).

### 3. Compiled-function tier (new architecture; first results 2026-09-22)

Built as `assay.compiled` (trainer `assay.train_compiled`, encoder `gte-modernbert-base`,
data v4 plus the 40k teacher-labelled generic set). Cells: accuracy / Brier / ECE after
scaling.

| encoder tier | seen (dev) | unseen (holdout) | transfer-v4 |
|---|---|---|---|
| zero-shot cosine, untrained | 0.417 / 0.665 / 0.116 | 0.541 / 0.563 / 0.067 | 0.514 / 0.585 / 0.075 |
| compiled, 1 epoch, lr 2e-5, options carry the instruction | 0.591 / 0.490 / 0.029 | 0.545 / 0.538 / 0.033 | 0.423 / 0.621 / 0.086 |
| cross-encoder (one pass per option) | 0.670 / 0.422 / 0.027 | 0.619 / 0.479 / 0.025 | 0.527 / 0.531 / 0.061 |

The first compiled run underfits (train loss 0.86 against 0.4-0.6 for the decoders, still
falling when the schedule ended) and had a design flaw: every option text began with the
instruction, so the two options of a bool question were near-identical vectors (bool tasks
were its weakest family). Single-text classification is strong on both encoders (spam,
sentiment, topic at 0.9+), anything that relates two spans (NLI, QA, knowledge) is near
chance for the compiled reader and mediocre for the cross-encoder. Both sit at the level of
the *untrained* 1.7B decoder on unseen tasks (0.623), at a fraction of its cost: on CPU with
8 threads the compiled model encodes a state in 30 ms, compiles six questions once in
135 ms, and then decides all six in 2 ms.

Queued: the compiled model with content-only options, 3 epochs at lr 5e-5 and 8 slots; a
"conditioned" middle tier (instruction and state in one encoder pass, options compiled and
scored by content: one pass per question rather than per option). ModernBERT-large (MLM
weights only, no retrieval fine-tuning) was tried and dropped: its loss stayed 70% above
gte-base's at the same step, and it ran out of memory on the largest batches.
Decision rule unchanged: the compiled tier stays only if it comes within 5 points of the
cross-encoder on the holdout.

The question is compiled once into parameters; the state is encoded once; a decision is a
small computation between the two. One paraphrase-class encoder produces token-level state
embeddings and, from the instructions and option descriptions, a set of query vectors and
one vector per option. Queries attend over the state tokens (late interaction), the pooled
result is scored against the option vectors, softmax; evidence comes from attention mass and
a coverage head. Trained on human soft labels plus the distillation set.

Why: microsecond decisions after one encoder pass per state, precompiled questions for fixed
pipelines, CPU deployment, and it is a design nobody in this space has. Expected: near the
decoders on classification-shaped tasks, well below them on knowledge and rules (0.55-0.65
on the transfer suite). Measured on holdout ECE and cost per decision like every tier.
Drop if: a cross-encoder with the same backbone beats it by more than 5 points on the holdout.

### 4. Cross-encoder tier, only if 3 loses badly

Entailment-style encoder: `[state + question] [SEP] [option description]`, one pass per
option, NLI-pretrained start, fine-tuned with the same targets. Cheap to build from the
same pieces; a middle tier is worth having only if the gap it closes is large.

### 5. Decoder v2 for the small models

Retrain the 4B on the distillation set with two changes: a content-scored option term (letter
logit plus a bilinear term on the option's own encoded representation, removing the label
alphabet limit and making answers permutation-equivariant by construction), and a second
seed so differences under two points stop being noise. Same for the 1.7B.

### 6. Abstention with a guarantee

A conformal layer over every tier: per-question-type thresholds fitted on the calibration
split so that "act" sets have a stated error rate. The evidence head says "the state does not
say"; this says "the model does not know". Costs nothing at inference.

### Supporting work

- Prefix-state serving for hybrid backbones (snapshot the state's recurrent and KV state,
  run question suffixes as a batch) so packed multi-question requests work on Qwen3.5/3.8.
- Latency and throughput benchmark per tier, including CPU for the compiled tier.
- Second seed for the soft-versus-hard target ablation.
- Known issue, resolved 2026-09-22: the sporadic native segfaults during long GPU runs were
  not the Triton or 4-bit kernels. Every one of them (14 in 24 hours, including four in 25
  minutes of plain-SDPA encoder training) faults at the same `libcuda.so` instruction and
  always on CPU core 6 of this 24-core machine, which points at that core. GPU jobs now run
  with `CPUAffinity=0-5,7-23` (systemd) or `taskset -c 0-5,7-23`; checkpoint plus resume
  stays on in every trainer.

## Evaluation rules that apply to all of the above

- Report accuracy, Brier, NLL, ECE and confident-error rate on seen tasks, unseen tasks and the
  public transfer suite; temperature fitted on seen-task calibration data only.
- The transfer suite's sources stay out of training. Its synthetic policy families are one
  skill family; do not tune to them.
- A change ships only if it holds on the unseen-task holdout, not just the transfer suite.
