# Examples

Four things you can run: ask a published model, walk a decision tree, call a server, and
train your own model from your own data with one configuration file.

Everything here uses the same record and request shapes, so a model trained by the pipeline
is served and queried exactly like the published ones.

## Ask a model

```bash
python examples/quickstart.py --model Berk/assay-0.6b
```

One support ticket, three question types, calibrated probabilities with a confidence and an
evidence score. Nothing is generated: each number is a distribution over the option labels.
The encoder tier runs the same script on a CPU:

```bash
python examples/quickstart.py --model Berk/assay-compiled-base --tier encoder
```

## Walk a decision tree

```bash
python examples/decision_graph.py --model Berk/assay-0.6b
```

A triage graph: route the ticket, then ask the question that branch needs. Every question in
the graph is answered in one forward pass because the branches are isolated from each other,
and a node that is not confident enough falls back to human review instead of guessing.

## Call a server

```bash
python -m assay.server --model Berk/assay-0.6b --port 8000
python examples/serve_client.py --url http://127.0.0.1:8000
```

The same deployment answers `/v1/decide` (native), `/v1/systemone` and `/v1/systemone/batch`
(the shape other open decision models take: `criteria` for options, `noul` for a boolean) and
`/v1/decide_graph`.

## Train on your own data

```bash
python examples/support_data.py                                   # writes examples/data/support
python -m assay.pipeline --config examples/pipelines/support-encoder.json --dry-run
python -m assay.pipeline --config examples/pipelines/support-encoder.json
```

`support_data.py` writes a small support-desk dataset in the record format: one state per
line with the three question types over it. Replace it with your own file and nothing else
changes.

The pipeline runs training, temperature calibration, evaluation and conformal abstention as
separate processes, skipping any stage whose output is already there, and writes
`pipeline.json` with the numbers. The whole run takes about six minutes on a CPU, and it
scores 1.00 with a Brier of 1e-7: the tickets come from a handful of templates, so the task
is separable and the numbers say nothing about a real dataset. Put your own data in and the
same output becomes informative. `support-decoder.json` does the same for the decoder tier,
which needs a GPU; the encoder tier trains on a CPU, slowly.

Afterwards:

```bash
python -m assay.server --model runs/example-support-encoder --port 8000
python -m assay.publish --run runs/example-support-encoder --repo <user>/<name>
```

## Data written for other decision models

Training files in the other shape load unchanged: options under `criteria`, a boolean typed
`noul`, levels as a list.

```json
{"state": "The dashboard is down for everyone.",
 "questions": {
   "route": {"type": "choice", "instructions": "Which team?",
             "criteria": {"billing": "Payments", "technical": "Faults"}, "label": "technical"},
   "outage": {"type": "noul", "instructions": "Is this an outage?", "label": true}}}
```

Point a pipeline configuration at a directory of those files and it trains, calibrates and
serves them like any other dataset.

## The configuration file

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

`tier` is `decoder` (a base LM read at the answer position), `encoder` (a sentence encoder
with a small reader, CPU-friendly) or `seq2seq` (an encoder-decoder). Whatever is under
`train` becomes flags for that tier's trainer, so any option the trainer has is available
without the pipeline knowing about it. `evaluate` names extra splits as `{"name": "path"}`;
without it the pipeline evaluates `dev.jsonl` and `holdout.jsonl` from the data directory.
`alpha` is the error rate the conformal thresholds target and `delta` the confidence in that
bound.
