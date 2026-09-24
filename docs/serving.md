# Serving Assay

One process holds one model and answers decisions over HTTP. There is no decode loop -- every
request is a single prefill -- so the server is simple to run and its capacity is easy to
reason about: throughput is questions per second, not tokens per second.

```bash
uv run python -m assay.server --model Berk/assay-4b --port 8000
curl -s localhost:8000/health
```

`--model` takes a Hub id, a run directory from `assay.pipeline`, or `base:<hf id>` for an
untrained readout. The tier is read from the saved configuration, so the decoder, encoder and
encoder-decoder models are all served the same way.

## A note on transformers versions

transformers 5 changed three things a saved model carries, and transformers 4 -- what SGLang,
vLLM and llama.cpp's converter pin, and what most people still run -- reads none of them. The
published models carry both spellings, and `assay.compat` applies the same fix to anything this
repository publishes. If you save a model yourself, run it through `assay.compat` before
serving it anywhere that is not transformers 5.

Two of the three fail loudly:

- `additional_special_tokens` became `extra_special_tokens`, still written as a list, and 4
  reads that name as a mapping: `AttributeError: 'list' object has no attribute 'keys'`.
- a tokenizer with no concrete class is saved as `TokenizersBackend`, and 4 answers
  `Tokenizer class TokenizersBackend does not exist`. Its name on 4 is `PreTrainedTokenizerFast`.

The third fails silently, and it is the one to know about. The RoPE settings moved into a
`rope_parameters` block and the top-level `rope_theta` stopped being written. transformers 4
reads the old name, does not find it, and falls back to the model class's default -- for Qwen3
that is 10000 against a true 1000000, a hundredfold error in the RoPE base. Nothing warns. The
server starts, the model answers, and the answers are wrong: measured on assay-0.6b through
SGLang, the readout hidden state fell to a cosine of 0.9940 against the reference and
probabilities moved by up to 7.1e-2 -- enough to cross a decision threshold, small enough to
look like rounding if you are not comparing against anything.

That is the case for `verify_against_local` being routine rather than ceremonial. It is also
why the evidence score is a weak check on its own: it is a sigmoid that usually sits near 1.0,
so it agreed to 2.7e-3 while the hidden state under it was visibly wrong.

## A page for trying it

`GET /` serves a page with no build step and no external assets: paste a state, add typed
questions, and see the probabilities, the confidence, the evidence score and, when the model
has conformal thresholds, whether each answer clears the act threshold. A second tab lists the
agents this server holds and walks one over a state, showing the path it took and the action it
would return.

```bash
uv run python -m assay.server --model Berk/assay-4b --agents examples/agents --port 8000
# then open http://127.0.0.1:8000/
```

The page calls the same endpoints a client calls, so what it shows is what a client gets. It
is open even when `--api-key` is set -- the page asks for the key and sends it as a bearer
token -- because the guard belongs on `/v1`, not on a static file.

`GET /chat` is the same model in a conversation: the standing conditions are given once, then
each turn runs an agent over the conversation so far and shows what it decided, the
probabilities it decided on, and how long it took. It is the clearest demonstration of what
this is for -- a decision lands in tens of milliseconds, and an escalation is a decision too.
The reply text is a template the outcome selects; the model generates none of it.

`scripts/record_chat.py` records that page as a video against a running server, for when a
decision arriving in 35 ms is easier to show than to describe. It needs playwright and ffmpeg,
which this project does not depend on, so run it with a python that has them.

## Choosing a runtime

Four ways to run the same model. They differ in what they can return, not in what they answer:
the temperature, the evidence head and the conformal thresholds all run client-side, so a model
gives the same answers wherever the weights sit.

| | when it is the right one | cost of a decision | checked against the reference |
|---|---|---|---|
| `assay.server` (transformers) | the default, and the only one that packs | 24 questions cost about what one costs | it is the reference |
| SGLang | you already run SGLang, or want RadixAttention across requests | 42 ms for three questions, warm | probabilities 1.2e-2, evidence 7.7e-5 |
| llama.cpp | no GPU | 57 ms for one question, 262 ms for three | probabilities 1.1e-4, evidence 2.4e-6 |
| vLLM | not available; see [roadmap.md](roadmap.md) | | |

Prefer `assay.server` unless you have a reason not to. It is the only one that packs questions
into a single forward pass, which is the whole economic argument for asking twenty questions
about a state instead of one: on the dense decoder tiers twenty-four questions cost about what
one costs. Neither SGLang nor llama.cpp can do that -- each question is a separate request --
so their per-question cost is flat, and a decision that asks many questions pays for each.

The reason to use SGLang anyway is that you are already running it. The reason to use llama.cpp
is that there is no GPU. Both are worth checking with `verify_against_local` first, and the
numbers above are what that check returned here; the SGLang figure is bf16 rounding rather than
a runtime difference, and it is worst on near-uniform distributions.

## Choosing a model

| you want | use | unseen-task accuracy | one question |
|---|---|---|---|
| the best answers | [assay-27b](https://huggingface.co/Berk/assay-27b) | 0.842 | 110 ms |
| the usual choice | [assay-4b](https://huggingface.co/Berk/assay-4b) | 0.803 | 23 ms |
| cheap and quick | [assay-1.7b](https://huggingface.co/Berk/assay-1.7b) | 0.752 | 18 ms |
| the small end | [assay-0.6b](https://huggingface.co/Berk/assay-0.6b) | 0.704 | 17 ms |
| no GPU at all | [assay-compiled-base](https://huggingface.co/Berk/assay-compiled-base) | 0.606 | 29 ms (CPU) |

Full numbers, including abstention rates, are in [models.md](models.md). Twenty-four questions
over one state cost about what one question costs on the dense decoder tiers (57 ms on the
4B), so ask everything you need in one request rather than splitting it. The 27B has a hybrid
backbone whose linear-attention layers cannot be packed that way; it shares one encoding of
the state instead (`assay.prefix`), which puts twenty-four questions at 533 ms rather than 888.

Its 4-bit weights are also worth knowing about: probabilities move by around 2e-2 depending on
how requests are batched, so treat a threshold sitting within a couple of points of a decision
boundary on that model as approximate.

## Options

```
--model            a Hub id, a run directory, or base:<hf id>            (required)
--host --port      where to listen                        (127.0.0.1:8000)
--device           where the small tiers load; the decoder tier places itself
--api-key          require this bearer token on /v1/  (default $ASSAY_API_KEY, otherwise open)
--agents           a directory of agent specifications to register at startup
--max-state-tokens truncate a state to this many tokens                     (4096)
--max-batch-size   requests per forward pass                                  (16)
--max-batch-tokens token budget per forward pass                           (16384)
--batch-wait-ms    how long a batch waits to fill                            (5.0)
--max-queue        queued requests before the server answers 503              (256)
--workers-per-device  uvicorn workers; one GPU serves one                       (1)
```

### Tuning the batch

Requests that arrive together are answered in one forward pass. `--batch-wait-ms` is the only
latency you add deliberately: a request waits up to that long for company. Raise it (20-50 ms)
when throughput matters more than a single request's latency, lower it to 0 when it does not.
`--max-batch-tokens` protects memory -- a batch stops growing when the packed tokens would
exceed it -- and `--max-batch-size` bounds how many requests share a pass.

When the queue is longer than `--max-queue` the server answers **503** with `Retry-After: 1`
instead of growing an unbounded backlog. Out of memory is also a 503, with `Retry-After: 5`.
Both are signals to add a replica, not to retry harder.

### Scaling

One process, one model, one GPU. Run one container per GPU behind any load balancer; there is
no shared state between requests, so round-robin is enough. Sessions, affinity and sticky
routing are not needed. For CPU serving of the small tiers, give each process a few cores and
run several.

## Health, metrics and statistics

- `GET /health` -- `{"status": "ok", "model": ..., "queue_depth": n, "conformal": true}`.
  Use it as both liveness and readiness; it stays open when an API key is configured.
- `GET /metrics` -- Prometheus text: `assay_requests_total`, `assay_batches_total`,
  `assay_rejected_total`, `assay_queue_depth`, `assay_batch_size_mean`, `assay_latency_ms`
  with p50/p95/p99 quantiles.
- `GET /v1/stats` -- the same numbers as JSON, for a quick look without a scraper.

A healthy deployment has `assay_batch_size_mean` above 1 under load (requests are sharing
passes) and `assay_rejected_total` flat (the queue is not saturating).

## Authentication

`--api-key` or `ASSAY_API_KEY` requires `Authorization: Bearer <key>` on every `/v1/` route.
`/health` and `/metrics` stay open so probes and scrapers keep working. The server does no
rate limiting, no tenancy and no audit logging: put it behind your own gateway if you need
those.

## systemd

```ini
[Unit]
Description=Assay decision server
After=network-online.target

[Service]
WorkingDirectory=/srv/assay
Environment=ASSAY_API_KEY=change-me
ExecStart=/usr/bin/env uv run python -m assay.server \
    --model Berk/assay-4b --host 0.0.0.0 --port 8000 --agents /srv/assay/agents
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

## Docker

```dockerfile
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y python3 git && apt-get clean
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /srv/assay
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY assay ./assay
ENV HF_HOME=/models
EXPOSE 8000
CMD ["uv", "run", "python", "-m", "assay.server", "--model", "Berk/assay-4b", "--host", "0.0.0.0"]
```

Mount a volume at `/models` so the weights are downloaded once, and pass `--gpus all`.

### Without a GPU

The decoder tier runs on a CPU. It is slower -- `assay-0.6b` answers three questions in about
half a second on two cores, against 22 ms on a GPU -- but it needs no CUDA, and the image is
about a tenth of the size, because the dependencies that make the project heavy
(`bitsandbytes`, `flash-linear-attention`, `datasets`, `peft`, `accelerate`) are there for
quantized bases, hybrid backbones and training. A serving process touches none of them, so
install `assay` with `--no-deps` over a minimal CPU set:

```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && apt-get clean
RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface OMP_NUM_THREADS=2
WORKDIR /home/user/app
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --user "transformers>=4.51" "numpy>=2.0" "fastapi>=0.115" \
      "uvicorn>=0.30" "pydantic>=2.0" "huggingface-hub>=0.30" safetensors
RUN pip install --no-cache-dir --user --no-deps "assay @ git+https://github.com/bgokden/assay@main"
COPY --chown=user agents ./agents
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Berk/assay-0.6b')"
EXPOSE 7860
CMD ["python", "-m", "assay.server", "--model", "Berk/assay-0.6b", "--agents", "agents", \
     "--host", "0.0.0.0", "--port", "7860", "--device", "cpu"]
```

Baking the weights in with `snapshot_download` at build time means a cold container answers
immediately instead of pulling 1.2 GB on somebody's first request. Port 7860 and this layout
are what a Hugging Face Docker Space expects, so the same file runs there unchanged --
although Docker Spaces need a PRO account; only static Spaces are free.

## Serving through SGLang

`assay.backends.sglang` runs the decoder tier on an SGLang deployment instead of local
transformers: one `/generate` call returns both the option-label logprobs and the hidden state,
so the readout and the evidence head come from the same forward pass, and RadixAttention reuses
a state prefix across questions.

```bash
python -m sglang.launch_server --model-path Berk/assay-4b --port 30000 \
    --enable-return-hidden-states
```

```python
from assay.backends.sglang import SGLangClient, verify_against_local

client = SGLangClient("http://127.0.0.1:30000", "Berk/assay-4b")
client.answer(state, questions)
verify_against_local("http://127.0.0.1:30000", "Berk/assay-4b", state, questions)
```

`--enable-return-hidden-states` is required at launch and is not in the documented request
fields: without it the server rejects a request that asks for them and the evidence head has no
input. Pass `evidence=False` to the client to answer from the readout alone.

Checked against local transformers on assay-0.6b (SGLang 0.5.9): probabilities agree to 1.2e-2
and evidence to 7.7e-5. The probability figure is bf16 rounding rather than a runtime
difference -- our own bf16 path differs from the same fp32 reference by 1.4e-2 -- and it is
worst on a near-uniform distribution, where a small logit shift moves the most probability.

Two things to know. The prompt is sent as token ids, not text: rendering it back to a string
lets the server retokenize, which merges the newline pair at the state boundary into a single
token and moves probabilities by up to 6.1e-2. And the number of hidden-state rows returned is
not the number of prompt tokens -- RadixAttention recomputes only the tail of a cached prefix,
so the same prompt came back as 54 rows cold and 1 row warm. The readout is the last row of the
last sequence either way.

## Serving through llama.cpp

The decoder tier runs on a laptop this way, CPU only. Convert the merged weights once, then
start the server with the embedding flags -- the evidence head needs the hidden state at the
readout position, which is what last-token pooling returns:

```bash
python -m assay.publish --run runs/assay-0.6b --repo local/assay --dry-run   # merged weights
python convert_hf_to_gguf.py runs/assay-0.6b/hub --outfile assay-0.6b.gguf --outtype f16
llama-server -m assay-0.6b.gguf --port 8081 -c 4096 \
    --embeddings --pooling last --embd-normalize -1
```

```python
from assay.backends.llamacpp import LlamaCppClient, verify_against_local

client = LlamaCppClient("http://127.0.0.1:8081", "Berk/assay-0.6b")
client.answer(state, questions)
verify_against_local("http://127.0.0.1:8081", "Berk/assay-0.6b", state, questions)
```

`/completion` with `n_probs` gives the option-label distribution and `/embeddings` gives the
evidence head its input; the fitted temperature and the conformal thresholds are applied
client-side, exactly as with transformers. Checked on assay-0.6b (f16 GGUF, CPU) against local
transformers: probabilities within 2.8e-4, evidence within 2.6e-4, hidden state cosine
similarity 1.000000.

Two things to know. llama.cpp cannot pack questions into one sequence, so each question is a
request; they share the state prefix and `cache_prompt` reuses it, so the state is encoded once
for the first question. And `verify_against_local` is not optional ceremony: a GGUF is a
converted, usually quantised copy of the weights, and conversion is where a deployment goes
quietly wrong. Without `--embeddings --pooling last --embd-normalize -1` there is no evidence
score; pass `evidence=False` to answer from the readout alone.

### What it costs

assay-0.6b as an f16 GGUF, CPU only, `llama-server -t 8` with its default four slots:

| | |
|---|---|
| one question | 57 ms |
| three questions, one state | 262 ms (87 ms each) |
| three questions, `evidence=False` | 187 ms (62 ms each) |
| peak throughput | 15.6 questions/s, at two concurrent requests |
| agreement with transformers | probabilities 1.1e-4, evidence 2.4e-6 |

Two things those numbers say. The evidence head costs about 25 ms per question, because it is
a second request -- `/embeddings` -- rather than a second output of the first; turn it off if
you only need the distribution. And concurrency past two requests makes things worse, not
better: the threads are already saturated, so further requests contend rather than pipeline.
Add processes across cores, not concurrency within one.

## API

Every route below is under `/v1` and takes JSON. Answers carry `probabilities`, `confidence`
(0 for a uniform distribution, 1 for a one-hot) and `evidence` (whether the state contains what
is needed to answer). When the model directory holds `conformal.json`, they also carry `act`
(the top answer clears the fitted error rate) and `set` (the options that cannot be ruled out).

### POST /v1/decide

```json
{"state": "My card was charged twice for order A-104.",
 "questions": {
   "refund": {"type": "bool", "instructions": "Does the customer ask for money back?"},
   "team": {"type": "choice", "instructions": "Which team should handle this?",
            "options": {"billing": "Charges and refunds", "technical": "Bugs and outages"}},
   "anger": {"type": "score", "instructions": "How angry is the customer?",
             "levels": ["calm", "annoyed", "furious"]}}}
```

```json
{"model": "assay-4b",
 "answers": {
   "refund": {"type": "bool", "p_true": 0.97, "probabilities": {"yes": 0.97, "no": 0.03},
              "confidence": 0.94, "evidence": 0.99, "act": true, "set": ["yes"]},
   "team": {"type": "choice", "choice": "billing", "probabilities": {}},
   "anger": {"type": "score", "score": 1.2, "level": 1, "legend": {"0": "calm"}}},
 "usage": {"input_tokens": 61}, "latency_ms": 24.8}
```

Up to 255 questions per request. A malformed question is a 422.

### POST /v1/systemone and /v1/systemone/batch

The request shape other open decision models use: options under `criteria`, booleans typed
`noul`. The response groups answers by type -- `choices[name].choice`, `nouls[name].noul`,
`scores[name].score` -- so a client written for that interface works unchanged. The batch form
takes `{"requests": [...]}`, at most 128 requests and 512 decisions in one submission, and
answers them in as few passes as the budget allows.

### POST /v1/decide_graph

```json
{"state": {"message": "everything is down"},
 "graph": {"start": "triage",
           "nodes": {"triage": {"question": {"type": "bool", "instructions": "An outage?"},
                                "edges": {"yes": "page"},
                                "min_probability": 0.45, "fallback": "human"},
                     "page": {"outcome": "page_oncall"},
                     "human": {"outcome": "human_review"}}}}
```

Answers every question in the graph in one forward pass and walks it, returning `outcome`,
`path`, `path_probability` and every answer. Cycles and undefined targets are a 422. See
[agents.md](agents.md).

### Agents

- `POST /v1/agents` registers an agent specification for this process.
- `GET /v1/agents`, `GET /v1/agents/<name>` list and describe them.
- `POST /v1/agents/<name>/run` takes `{"state": ...}` and returns the outcome, the `action` it
  stands for and that action's `arguments`.
- `DELETE /v1/agents/<name>` removes one.

The server never performs an action; it says what should happen and the caller does it. Agents
that must survive a restart belong in files loaded with `--agents`. See [agents.md](agents.md).

### GET /v1/models

The model this process serves and the backbone it was trained from.

## Scoring a file instead of serving

For bulk work there is no reason to go through HTTP:

```bash
uv run python -m assay.apply --model Berk/assay-4b --questions questions.json \
    --states tickets.jsonl --out answers.jsonl
```

`questions.json` is the question set in the request shape, `tickets.jsonl` is one state per
line (a bare value, or a record with a `state` field), and each output line carries the state's
identifier and its answers.
