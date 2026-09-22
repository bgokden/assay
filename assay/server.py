"""HTTP server: POST /v1/decide with a state and typed questions, get distributions back.

    uv run python -m assay.server --model runs/assay-4b --port 8000

Request:
    {"state": <str | object | list>,
     "questions": {"name": {"type": "bool" | "choice" | "score", "instructions": "...",
                            "options": {...} | "levels": [...] | "yes": "...", "no": "..."}}}
Response:
    {"model": "...", "answers": {"name": {...}}, "usage": {"input_tokens": n}, "latency_ms": t}

POST /v1/systemone accepts the System One style payload other open decision models use
(questions typed choice/noul/score with options under `criteria`) and groups the answers by
type, so clients written for that interface work unchanged.

POST /v1/decide_graph walks a decision tree: a start node, question nodes whose edges are
keyed by the chosen option, and outcome nodes. Every question in the graph is answered in one
forward pass (branches are isolated over the shared state), so a whole tree costs what one
question costs; the walk is then pure logic. See assay.graph.

Requests are batched: several arriving together are answered in one forward pass, which is
where the throughput is (24 questions cost 56 ms batched against 544 ms one at a time).
GET /health reports readiness, GET /metrics exposes Prometheus counters.

When the model directory holds conformal.json (see assay.conformal), every answer also
carries "act" (the top answer is confident enough for the fitted error rate) and "set"
(the options that cannot be ruled out at the fitted coverage), and a graph node can require
that threshold before it routes.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from assay.batching import Batcher, QueueFull
from assay.conformal import CONFORMAL_FILE, decorate
from assay.encoding import encode, identity_order
from assay.graph import Graph, walk
from assay.model import AssayModel
from assay.schema import Answer, Question

MAX_QUESTIONS = 255


class DecideRequest(BaseModel):
    state: Any
    questions: dict[str, dict[str, Any]] = Field(min_length=1, max_length=MAX_QUESTIONS)


class GraphRequest(BaseModel):
    state: Any
    graph: dict[str, Any]


class SystemOneRequest(BaseModel):
    """The System One style payload used by other open decision models: a state, named
    questions typed choice/noul/score, and options under `criteria`."""

    state: Any
    questions: dict[str, dict[str, Any]] = Field(min_length=1, max_length=MAX_QUESTIONS)
    model: str | None = None


def load_conformal(model: str) -> dict[str, Any] | None:
    """Thresholds from a model directory or a Hub repository, when it has them."""
    if model.startswith("base:"):
        return None
    if not os.path.isdir(model):
        from huggingface_hub import snapshot_download

        model = snapshot_download(model)
    path = os.path.join(model, CONFORMAL_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def create_app(
    model: AssayModel,
    model_name: str,
    max_state_tokens: int = 4096,
    conformal: dict[str, Any] | None = None,
    batcher: Batcher | None = None,
) -> FastAPI:
    batcher = batcher or Batcher(model)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await batcher.stop()  # drain in-flight batches on shutdown

    app = FastAPI(title="assay", version="0.5.0", lifespan=lifespan)
    app.state.batcher = batcher

    def parse(questions: dict[str, dict[str, Any]]) -> dict[str, Question]:
        try:
            return {n: Question.from_dict(q) for n, q in questions.items()}
        except (KeyError, ValueError, TypeError) as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    async def run(state: Any, questions: dict[str, Question]) -> tuple[dict[str, Any], int]:
        """Encode, submit to the batcher, and serialise the answers with any thresholds."""
        names = list(questions)
        qs = [questions[n] for n in names]
        packed = encode(
            model.tokenizer,
            model.alphabet,
            state,
            qs,
            orders=[identity_order(q) for q in qs],
            max_state_tokens=max_state_tokens,
        )
        try:
            answers = await batcher.submit(packed, qs)
        except QueueFull as e:
            raise HTTPException(status_code=503, detail=str(e), headers={"Retry-After": "1"}) from e
        except torch.OutOfMemoryError as e:
            raise HTTPException(
                status_code=503, detail="out of memory", headers={"Retry-After": "5"}
            ) from e
        out = {n: a.to_dict(questions[n]) for n, a in zip(names, answers)}
        if conformal is not None:
            for n in names:
                decorate(out[n], questions[n], conformal)
        return out, len(packed)

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"models": [{"id": model_name, "base": model.base_model_id}]}

    @app.get("/health")
    def health() -> dict[str, Any]:
        depth = batcher.queue.qsize()
        return {
            "status": "ok" if depth < batcher.max_queue else "saturated",
            "model": model_name,
            "queue_depth": depth,
            "conformal": conformal is not None,
        }

    @app.get("/metrics")
    def metrics() -> Response:
        text = batcher.metrics.prometheus(batcher.queue.qsize())
        return Response(content=text, media_type="text/plain; version=0.0.4")

    @app.get("/v1/stats")
    def stats() -> dict[str, Any]:
        return batcher.metrics.snapshot(batcher.queue.qsize())

    @app.post("/v1/decide")
    async def decide(req: DecideRequest) -> dict[str, Any]:
        t0 = time.perf_counter()
        questions = parse(req.questions)
        out, tokens = await run(req.state, questions)
        elapsed = (time.perf_counter() - t0) * 1000.0
        batcher.metrics.observe_request(elapsed)
        return {
            "model": model_name,
            "answers": out,
            "usage": {"input_tokens": tokens},
            "latency_ms": round(elapsed, 1),
        }

    @app.post("/v1/systemone")
    async def systemone(req: SystemOneRequest) -> dict[str, Any]:
        """Compatible with the System One interface other open decision models expose, so the
        same client can call this server. The request shape is theirs (`criteria`, `noul`);
        the response groups answers by type the way their SDK reads them
        (`choices[name].choice`, `nouls[name].noul`, `scores[name].score`) and adds our own
        fields: confidence, evidence, and act/set when conformal thresholds are fitted."""
        t0 = time.perf_counter()
        questions = parse(req.questions)
        out, tokens = await run(req.state, questions)
        grouped: dict[str, dict[str, Any]] = {"choices": {}, "nouls": {}, "scores": {}}
        for name, question in questions.items():
            answer = out[name]
            if question.type == "choice":
                grouped["choices"][name] = {"choice": answer["choice"], **answer}
            elif question.type == "bool":
                grouped["nouls"][name] = {"noul": answer["p_true"], **answer}
            else:
                grouped["scores"][name] = {"score": answer["score"], **answer}
        elapsed = (time.perf_counter() - t0) * 1000.0
        batcher.metrics.observe_request(elapsed)
        return {
            "model": model_name,
            **grouped,
            "usage": {"input_tokens": tokens},
            "latency_ms": round(elapsed, 1),
        }

    @app.post("/v1/decide_graph")
    async def decide_graph(req: GraphRequest) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            graph = Graph.from_dict(req.graph)
        except (KeyError, ValueError, TypeError) as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        questions = graph.questions()
        if not questions:
            raise HTTPException(status_code=422, detail="graph has no questions")
        if len(questions) > MAX_QUESTIONS:
            raise HTTPException(
                status_code=422, detail=f"graph has more than {MAX_QUESTIONS} questions"
            )
        out, tokens = await run(req.state, questions)
        names = list(questions)
        qs = [questions[n] for n in names]
        answers = {n: model_answer(out[n], q) for n, q in zip(names, qs)}
        try:
            result = walk(graph, answers, conformal)
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        elapsed = (time.perf_counter() - t0) * 1000.0
        batcher.metrics.observe_request(elapsed)
        return {
            "model": model_name,
            **result,
            "answers": out,
            "usage": {"input_tokens": tokens, "questions": len(questions), "forward_passes": 1},
            "latency_ms": round(elapsed, 1),
        }

    return app


def model_answer(serialised: dict[str, Any], question: Question) -> Answer:
    """Rebuild an Answer from its serialised form so the graph can route on it."""
    probs = {k: float(serialised["probabilities"][k]) for k in question.keys}
    return Answer(
        type=question.type,
        probabilities=probs,
        confidence=float(serialised["confidence"]),
        evidence=float(serialised["evidence"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="saved model directory or base:<hf id>")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-state-tokens", type=int, default=4096)
    ap.add_argument("--max-batch-size", type=int, default=16, help="requests per forward pass")
    ap.add_argument(
        "--max-batch-tokens", type=int, default=16384, help="token budget per forward pass"
    )
    ap.add_argument(
        "--batch-wait-ms", type=float, default=5.0, help="how long to wait for a batch to fill"
    )
    ap.add_argument(
        "--max-queue", type=int, default=256, help="queued requests before answering 503"
    )
    ap.add_argument(
        "--workers-per-device", type=int, default=1, help="uvicorn workers; one GPU serves one"
    )
    args = ap.parse_args()
    from assay.evaluate import load_model

    model = load_model(args.model)
    model.eval()
    name = os.path.basename(args.model.rstrip("/")) or args.model
    batcher = Batcher(
        model,
        max_batch_size=args.max_batch_size,
        max_batch_tokens=args.max_batch_tokens,
        max_wait_ms=args.batch_wait_ms,
        max_queue=args.max_queue,
    )
    app = create_app(
        model, name, args.max_state_tokens, conformal=load_conformal(args.model), batcher=batcher
    )
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers_per_device)


if __name__ == "__main__":
    main()
