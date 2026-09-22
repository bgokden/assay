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
| compiled, 3 epochs, lr 5e-5, 8 slots, content-only options | 0.642 / 0.455 / 0.038 | 0.561 / 0.517 / 0.045 | 0.423 / 0.635 / 0.090 |
| conditioned (instruction + state in one pass, compiled options), 2 epochs | 0.652 / 0.434 / 0.027 | 0.564 / 0.515 / 0.034 | 0.445 / 0.621 / 0.081 |
| compiled + late interaction (MaxSim option tokens x state tokens), 3 epochs | 0.668 / 0.442 / 0.037 | 0.606 / 0.494 / 0.061 | **0.542 / 0.572 / 0.095** |
| conditioned + late interaction, 2 epochs | 0.678 / 0.417 / 0.031 | 0.616 / 0.471 / 0.038 | 0.537 / 0.583 / 0.127 |

Verdict (2026-09-22 morning): the late-interaction term is what the compiled reader was
missing. With it the compiled tier is 1.3 points behind the cross-encoder on unseen tasks
and ahead of it on the transfer suite (SciQ 0.23 -> 0.76, deadline 0.30 -> 0.63: option text
that appears in the state is now found), inside the 5-point rule, while keeping the
encode-once decision path (3 ms for six decisions on a CPU after the state encode). The
conditioned variant matches the cross-encoder on unseen tasks at one encoder pass per
question instead of per option. Published: `Berk/assay-compiled-base` (compiled + late
interaction). All encoder tiers remain far below the decoders (unseen tasks 0.61 against
0.75-0.84); they are the on-device tier, not a replacement.

The first compiled run underfits (train loss 0.86 against 0.4-0.6 for the decoders, still
falling when the schedule ended) and had a design flaw: every option text began with the
instruction, so the two options of a bool question were near-identical vectors (bool tasks
were its weakest family). Single-text classification is strong on both encoders (spam,
sentiment, topic at 0.9+), anything that relates two spans (NLI, QA, knowledge) is near
chance for the compiled reader and mediocre for the cross-encoder. Both sit at the level of
the *untrained* 1.7B decoder on unseen tasks (0.623), at a fraction of its cost: on CPU with
8 threads the compiled model encodes a state in 30 ms, compiles six questions once in
135 ms, and then decides all six in 2 ms.

The three-epoch run with content-only options fixes the bool family (0.613 -> 0.669 on
unseen bool questions; scitail 0.71 -> 0.83) and lifts seen tasks by five points, but it
overfits (fitted temperature 1.93) and stays 5.8 points behind the cross-encoder on unseen
tasks; the transfer suite does not move because its knowledge families (MMLU, SciQ) need what
a 149M encoder does not have. Queued: a
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

### 4. Cross-encoder tier, only if 3 loses badly (built as the comparison; not published)

Entailment-style encoder: `[state + question] [SEP] [option description]`, one pass per
option, NLI-pretrained start, fine-tuned with the same targets. Cheap to build from the
same pieces; a middle tier is worth having only if the gap it closes is large.

### 5. Decoder v2 for the small models (done 2026-09-22)

Second seeds: assay-4b (data v4) seed 1 gives dev 0.786 / 0.288 / 0.022, holdout 0.796 /
0.278 / 0.024, transfer 0.795 / 0.291 / 0.049 against seed 0's 0.791 / 0.287 / 0.027,
0.803 / 0.271 / 0.023, 0.784 / 0.302 / 0.061: half a point to a point of seed noise, more on
the small transfer suite. The 1.7B soft/hard ablation at seed 1 reproduces seed 0 to 0.3
points and keeps the ordering (soft: lower raw ECE, fewer confident errors, lower Brier;
equal after scaling). Rule kept: under two points is noise.

Content-scored option term (`--content-term`: the option line's token span is mean-pooled
and scored bilinearly against the decision vector, zero-initialised so training starts from
the label readout): neutral. 1.7B: holdout 0.745 / 0.345 / 0.035, transfer 0.671 / 0.440
(plain seeds 0.752 / 0.334 and 0.752 / 0.328; 0.670 and 0.668); zeroing the learned term at
inference moves holdout accuracy by 0.5 points, so the model barely uses it. 4B: 0.788 /
0.287 dev, 0.800 / 0.278 holdout, 0.795 / 0.294 transfer, inside the two-seed band. The label
readout already carries the decision; the term stays as an option for questions with more
options than the label alphabet (552 single-token labels), off by default.

Retrain the 4B on the distillation set with two changes: a content-scored option term (letter
logit plus a bilinear term on the option's own encoded representation, removing the label
alphabet limit and making answers permutation-equivariant by construction), and a second
seed so differences under two points stop being noise. Same for the 1.7B.

### 6. Abstention with a guarantee (built 2026-09-22)

`assay.conformal`, fitted on the saved calibration predictions (`assay.calibrate` and
`assay.evaluate` now write per-item predictions next to their reports). Two thresholds per
question type: a split-conformal prediction-set threshold (coverage >= 1 - alpha) and an act
threshold on the top probability chosen by one-sided binomial tests over a grid with a
Bonferroni correction (acted-on error <= alpha at confidence 1 - delta). The server returns
`act` and `set` per answer when `conformal.json` is in the model directory.

At alpha 0.1, delta 0.05 on the 4B (seed 1): calibration split coverage 0.90, acted-on error
7.6% (bool) and 8.0% (choice) with act rates 90% and 73%, as the construction promises. On
unseen tasks the same thresholds give coverage 0.83 (bool) / 0.93 (choice) and acted-on error
10.8% / 7.1% at act rates 77% / 72%; on the transfer suite 14.6% / 11.3% at 90% / 74%. The
guarantee is for the calibration distribution and the shift to unseen task families costs a
few points of error, less for choice than for bool; that is the number to quote, and the
report prints it for every split. Score questions get no act threshold at alpha 0.1 because
exact-level accuracy is the wrong error for ordinal answers (adjacent levels count as wrong);
their prediction sets work (coverage 0.84-1.0, about two levels wide). The 1.7B acts less
often (65% / 53% on unseen tasks) at similar error, the 27B more often (89% / 84% at 11.2% /
7.4%); the encoder tiers act rarely (10-34%). The published 1.7B, 4B and 27B now carry
`conformal.json` and an abstention section in their model cards.

The evidence head says "the state does not say"; this says "the model does not know". Costs
nothing at inference.

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
  stays on in every trainer. Evidence, the test procedure and the fix options are in
  `docs/ops.md`.

## Evaluation rules that apply to all of the above

- Report accuracy, Brier, NLL, ECE and confident-error rate on seen tasks, unseen tasks and the
  public transfer suite; temperature fitted on seen-task calibration data only.
- The transfer suite's sources stay out of training. Its synthetic policy families are one
  skill family; do not tune to them.
- A change ships only if it holds on the unseen-task holdout, not just the transfer suite.
