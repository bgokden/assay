# Research notes: calibrated typed-decision models

Compiled 2026-09-20. Purpose: ground the design of our own model (working name: Assay)
in prior work and in what the community learned in the first days after Jev's release.

## 1. What Jev is (externally observable)

- TypeSafe "System One" model, early access 2026-09-15, closed weights, hosted API only.
- Input: state (string/object/array) + typed questions: noul (yes/no), choice (<=255 options),
  score (2..10 ordered levels). Output: probabilities per option, `confidence`, expected score.
- confidence = (p_max - 1/K) / (1 - 1/K)  (distance from uniform; verified against docs widget).
- Claimed training: "RLCD" (reinforcement learning for calibrated decisions). Unpublished.
- Latency 70-500 ms, prefill only, questions evaluated as isolated branches over a shared state.
- Architecture reconstruction from API probing (archerhume.com, "Jev's Architecture Unmasked"):
  shared state KV cache + isolated question suffixes; decision reads the full option list;
  adding an irrelevant option shifts log-odds between the others (so not fixed independent logits);
  ECE 0.031 on 1200 MMLU items; probably MoE ~10B active.

## 2. Open replications (all created 2026-09-17..20)

| project | backbone | readout | training | calibration reported |
|---|---|---|---|---|
| jaredpalmer/kev (933 stars) | Qwen3 0.6B/4B/8B Base + LoRA | pointer head: decision token vs option-end hidden states | CE on hard labels, ~3-20k records public + synthetic | Kev-8B Brier 0.339 on transfer-v4 dev; Jev 0.211; untrained 8B base 0.366 |
| wfzyx/von | 395M encoder, "option-marker joint attention" | encoder | CE + Brier, temperature 1.037 | 93.5% acc on jabr/classifier-benchmark (78 cases) |
| ikermoel/open-alternative-jev | any HF LLM, inference only | next-token logits at each question's position | none | RACE-H 92.9%, MMLU 84% with Qwen3.6-27B |
| razorback16/openjev | DiffusionGemma 26B-A4B via vLLM | diffusion LM probabilities | none | confidence = 1 - H(p)/ln K |
| jaswanthsanjay88/rev | kev clone + prefix KV cache | pointer | CE | - |
| bespokelabs Nimble-9B | Qwen3.5-9B LoRA | letter-token logits | 2.7k synthetic contrastive records | 90.1% vs Jev 93.2% on their set |
| kotobalabs open-jev-deberta-v3-large | DeBERTa cross-encoder | NLI style | - | - |

| Chawla, "Build your own Jev (100% local)" (2026-09-20) | any HF LLM via SGLang `/v1/score` | letter-token logits, restricted softmax | none (explicitly) | none; article notes p is not an accuracy |

That last one is a tutorial rather than a project, and it is the clearest statement of what
the inference path alone gives you: SGLang's `/v1/score` takes a prompt and a list of
single-token label ids and returns the restricted softmax over them, so the readout needs no
model changes at all. It gets the details right that matter (labels must be one token,
leading spaces and chat templates change token ids, an escape option is needed when the list
is not exhaustive) and is honest that training, calibration and an evaluation stack are
missing. The gap it leaves is the one Assay is about: an untrained readout on our own suite
scores 0.623 on unseen tasks at 1.7B and 0.740 at 4B, against 0.752 and 0.803 trained, and
its probabilities have no guarantee attached.

The article's suggested policy, "act when the top probability exceeds 0.80 and leads the
runner-up by 0.20", is a reasonable default and measurably model-dependent. The same rule on
unseen tasks, on raw probabilities:

| model | acts on | error among acted |
|---|---|---|
| assay-27b | 68% | 4.5% |
| assay-4b | 66% | 7.8% |
| assay-1.7b | 58% | 9.8% |
| compiled encoder tier | 42% | 23.1% |
| cross-encoder tier | 29% | 16.2% |

On a well-trained decoder it lands close to our fitted thresholds (assay-4b: 66% at 7.8%
against conformal's 67% at 9.0% for a 10% target), which is a point in its favour; on a
weaker model the same constants silently act on 42% of questions at 23% error. That is the
argument for fitting thresholds per model and per question type with a stated error rate
(`assay.conformal`) rather than carrying constants between models.

**Decision 1.0** (vLLM Semantic Router, announced 2026-09-22): six Apache-2.0 models, Kai-0.6B
and Lex-0.6B (encoders), Eos-0.8B (hybrid decoder), Sol-2B, Nox-4B, Lux-9B, served at
`/v1/systemone` with the same three primitives we use. Reported on their own 54-task, 3,766
decision suite: Lux-9B 76.94, Nox-4B 73.09, Sol-2B 66.32, Eos-0.8B 61.89, Kai-0.6B 53.52 -
not comparable to our numbers, which are on different splits. Two things matter for us. Their
tiering reproduces what we measured independently on 2026-09-22: the 0.6B encoder is the weak
tier (53.52) and a 0.8B decoder beats it (61.89), the same ordering and roughly the same gap as
our 0.553 for a 0.6B under a learned reader against 0.704 reading out of the same weights. And
their request shape is becoming a common interface, so `assay.server` now serves `/v1/systemone`
alongside its own endpoint: one parser accepts both shapes (`criteria` as well as `options`,
`noul` as well as `bool`). The response groups answers as their SDK reads them
(`choices[name].choice`, `nouls[name].noul`, `scores[name].score`); their response schema is
not published verbatim, so that grouping is read off their SDK accessors and may need adjusting.

Their Eos-0.8B architecture doc is worth reading against ours. The backbone is a 752M hybrid
Qwen3.5-0.8B (18 GatedDeltaNet layers, 6 full-attention, hidden 1024) and the decision head is
1.05M parameters reading hidden states at the candidate positions and at a final query
position, with a bilinear branch and an MLP branch and separate fp32 layer norms. The answer
is **not** read from vocabulary logits: their head produces the logits directly, which is the
pointer-head family rather than our readout. Calibration is one fitted temperature (1.0389),
as ours is. Their documentation describes no evidence or abstention signal.

Two things follow. Their head-over-candidate-positions design is close to the content-scored
option term we built and measured as neutral (`--content-term`), and it is the opposite of the
choice our own measurements support at small scale: the same Qwen3-0.6B weights give 0.704 on
unseen tasks read out of the vocabulary and 0.553 under a learned reader. Their scores and
ours are on different suites, so this is a difference in design, not a result: a fair
comparison needs one common suite, and their weights ship as `backbone/` plus
`decision_head.safetensors` with their own loading code, so running them in our harness would
measure their weights under our prompt format rather than their model. Not attempted for that
reason.

Evaluation resources: `jaredpalmer/kev-suites` (HF dataset; transfer-v4 dev = 764 items from
mmlu, emotion, sciq, tweet_offensive, qnli, paws + synthetic rule holdouts; Jev numbers
published on it), `jabr/classifier-benchmark` (78 cases, 8 tasks).

### Findings from kev's research log (runs on H100, three seeds each)

1. Backbone capacity is the biggest lever: 0.6B -> 4B = +14-19 pp transfer; 4B -> 8B = +1.5-2 pp.
2. LoRA learning rate 2e-4 erodes base capability; 5e-5 is +4.7 pp transfer and best Brier.
   2-epoch runs at 2e-4 collapsed.
3. The fresh pointer head LOSES base knowledge: fine-tuned MMLU 0.69 vs untrained 8B base 0.76;
   PAWS 0.76 vs base 0.84. Their proposed next lever: a hybrid readout that adds the base
   model's own option-token logits, or distilling the base's zero-shot distribution.
4. More public data raised in-distribution accuracy but lowered transfer; the mix matters more
   than volume. 1 epoch is better calibrated than 2.
5. Exact option isolation (permutation invariance by construction) cost accuracy at 4B.
6. None-of-the-above minimal pairs helped a lot (0.75 -> 0.85-0.93).
7. Ordinal "deadline" date arithmetic did not move with data; they call it a capability limit
   of one-pass readout at <= 8B.
8. Calibration was never the optimisation target; Brier stayed at 0.34 vs Jev 0.21 while
   accuracy closed to within 8 pp. THE CALIBRATION GAP IS THE OPEN PROBLEM.

## 3. Literature

### Reading probabilities off an LM without generating
- MMLU-style scoring (Hendrycks 2021): next-token logits over option letters. Base models
  already do this well; instruction tuning/RLHF makes it overconfident.
- cappr, outlines.choice, guidance.select: same idea productised.
- Kadavath et al. 2022, "Language Models (Mostly) Know What They Know": large models are
  well calibrated on MC/true-false in the right format; P(True) self-evaluation is calibrated
  and improves with scale. Supports starting from the base LM's own logits.

### Biases of the readout and inference-time fixes
- Zhao et al. 2021 "Calibrate Before Use" (contextual calibration): majority-label, recency and
  common-token bias; correct with a content-free input.
- Zheng et al. 2023 PriDe: selection bias in MCQ comes from option-ID token priors; estimate the
  prior by permuting and divide it out.
- Zhou et al. 2023 Batch Calibration: marginalise the score over a batch to estimate contextual
  bias; zero-shot, inference only.
- Robinson et al. 2022 multiple choice symbol binding: letter labels work better than free text
  for models that can bind symbols to options.
- Holtzman et al. 2021 surface form competition: probability mass split across paraphrases of
  the same answer; fixed label symbols avoid it.

### Calibration training
- Guo et al. 2017: modern nets are overconfident; temperature scaling on a held-out set fixes
  ECE cheaply but is a single global knob.
- Gneiting & Raftery 2007: proper scoring rules (log loss, Brier) are minimised only by the true
  distribution, so training on them with correct targets is the principled route to calibration.
- Muller et al. 2019: label smoothing improves calibration but flattens everything uniformly.
- Peterson et al. 2019 (CIFAR-10H), Uma et al. 2021 (survey), Crowd-Calibrator 2024, and the
  2026 "annotation saturation" and SMECE papers: training on human label DISTRIBUTIONS instead
  of majority votes gives the best calibration and the best ranking of ambiguous items.
  Standard ECE against majority labels punishes a model that correctly says 0.62 when humans
  split 62/25/13; use Brier/ECE against the distribution where available.
- Diaz & Marques 2019 SORD: for ordinal targets, use a soft target distribution that decays
  with distance from the true level (exp of a negative metric); improves both accuracy and the
  quality of the expected value compared with one-hot CE.
- Kapoor et al. 2024 Calibration-tuning, ConfTuner 2025 (tokenised Brier loss): fine-tuning
  with a proper scoring rule on outcome labels improves calibration across QA tasks without
  hurting accuracy.
- 2026 surveys (arXiv 2605.23909, 2606.03437): LLMs remain overconfident, verbalised
  confidence dissociates from logprobs, calibration under language variation is poor.

### Knowing what is not answerable
- SQuAD 2.0, QuAIL "not enough information", CosmosQA "none of the above": explicit
  unanswerable supervision.
- "Don't Hallucinate, Abstain" (2024), Calibration-tuning: models can learn to flag knowledge
  gaps; abstention is a separate signal from the answer distribution.
- Jev's own docs list "indirection" and missing information as failure modes and offer no
  signal for it.

### Fast zero-shot classifiers (encoder route)
- GLiClass (2025), GLiNER2, ModernBERT zero-shot: 10x faster than cross-encoders, single pass,
  but weaker on knowledge/reasoning-shaped questions; von took this route.
- Late interaction (ColBERT, Khattab & Zaharia 2020): keep token-level state embeddings and
  let a compact query interact with them, instead of one pooled vector per side. Our compiled
  tier is late interaction with a learned reader: the question is compiled into query slots
  and option vectors, the decision is cross-attention plus a bilinear score, so the state is
  encoded once and reused across questions and the answer is scored by option content.

### Abstention with a guarantee
- Split conformal prediction (Vovk et al. 2005; Papadopoulos 2002): with a held-out
  calibration set, the set {k : p_k >= 1 - q} contains the label with probability >= 1 - alpha
  on exchangeable inputs; least-ambiguous-set scores (Sadinle, Lei & Wasserman 2019) give the
  smallest sets when the model is calibrated.
- Learn-then-Test (Angelopoulos, Bates et al. 2021): risk control for a chosen threshold by
  testing each candidate with a valid p-value and correcting for multiplicity. We use the
  plain Bonferroni version over a grid of confidence thresholds with one-sided binomial tests,
  which needs no monotonicity assumption and is a few lines of code.
- The guarantees are about the calibration distribution; the reports show how they carry
  over to unseen tasks, which is the number an operator needs.

## 4. Design conclusions (what is ours)

1. Keep the base LM's own answer logits as the readout (vocabulary logits over option label
   tokens at a single decision position). Zero-shot competence is preserved at init; LoRA at low
   LR nudges rather than replaces it. This is the "hybrid/base-anchored readout" kev
   identified as the next lever but never built.
2. Optimise calibration directly: cross-entropy against SOFT targets. Human label distributions
   where they exist (ChaosNLI, GoEmotions raters, civil_comments fractions, SST continuous,
   DynaSent), SORD distributions for ordinal Score questions, one-hot otherwise. Log loss is a
   proper scoring rule; the targets are what make it calibrated.
3. Option order shuffling during training as the fix for label-token priors (instead of
   inference-time permutation, which multiplies latency).
4. An `evidence` output per question: P(the state contains what is needed to answer this).
   Trained on constructed unanswerable pairs (passage swaps, SQuAD2/QuAIL style). Separates
   "the state is ambiguous" from "the state does not say".
5. Isolated question branches over a shared state (block mask, restarted positions), one
   prefill, no decoding. Standard now; we adopt it and do not claim it.
6. Evaluate calibration properly: Brier, NLL, ECE, confident-error rate, on task families the
   model never saw, plus kev transfer-v4 for a public head-to-head with Kev and Jev.
7. Post-hoc thresholds with finite-sample guarantees (conformal sets, a tested act threshold)
   fitted on the same calibration split as the temperature, so "act or abstain" is a stated
   error rate rather than a hand-picked cutoff.
8. A second, generation-free architecture (`assay.compiled`): the question compiled into
   parameters, the state into token embeddings, a decision computed between the two. Same
   typed interface, same targets, same evaluation; a different point on the cost curve.

## 5. Sources

- https://docs.typesafe.ai/ (quickstart, confidence, AI primer)
- https://archerhume.com/posts/jevs-architecture-unmasked
- https://github.com/jaredpalmer/kev (PLAN.md), https://huggingface.co/datasets/jaredpalmer/kev-suites
- https://github.com/wfzyx/von, https://github.com/ikermoel/open-alternative-jev,
  https://github.com/razorback16/openjev, https://github.com/jaswanthsanjay88/rev
- https://github.com/bespokelabsai/nimble
- Chawla, "Build your own Jev (100% local)" (SGLang /v1/score walkthrough, 2026-09-20)
- arXiv 2207.05221 (Kadavath), 2102.09690 (Zhao), 2309.03882 (PriDe), 2309.17249 (Batch Cal.),
  1706.04599 (Guo), 1906.02629 (Muller), 2408.14141 (Crowd-Calibrator), 2605.29797,
  2603.14092 (SMECE), 2605.23909, 2606.03437, 2508.07662 (GLiClass), SORD (CVPR 2019),
  Kapoor 2024 (aclanthology 2024.uncertainlp-1.1), 2004.12832 (ColBERT), 2110.01052
  (Learn-then-Test), Sadinle et al. 2019 (JASA, least-ambiguous set-valued classifiers)
