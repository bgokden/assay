"""HTTP server: POST /v1/decide with a state and typed questions, get distributions back.

    uv run python -m assay.server --model runs/assay-4b --port 8000

Request:
    {"state": <str | object | list>,
     "questions": {"name": {"type": "bool" | "choice" | "score", "instructions": "...",
                            "options": {...} | "levels": [...] | "yes": "...", "no": "..."}}}
Response:
    {"model": "...", "answers": {"name": {...}}, "usage": {"input_tokens": n}, "latency_ms": t}

When the model directory holds conformal.json (see assay.conformal), every answer also
carries "act" (the top answer is confident enough for the fitted error rate) and "set"
(the options that cannot be ruled out at the fitted coverage).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from assay.conformal import CONFORMAL_FILE, decorate
from assay.encoding import encode, identity_order
from assay.model import AssayModel
from assay.schema import Question


class DecideRequest(BaseModel):
    state: Any
    questions: dict[str, dict[str, Any]] = Field(min_length=1)


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
) -> FastAPI:
    app = FastAPI(title="assay", version="0.4.0")

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"models": [{"id": model_name, "base": model.base_model_id}]}

    @app.post("/v1/decide")
    def decide(req: DecideRequest) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            questions = {n: Question.from_dict(q) for n, q in req.questions.items()}
        except (KeyError, ValueError, TypeError) as e:
            raise HTTPException(status_code=422, detail=str(e))
        names = list(questions)
        qs = [questions[n] for n in names]
        packed = encode(
            model.tokenizer,
            model.alphabet,
            req.state,
            qs,
            orders=[identity_order(q) for q in qs],
            max_state_tokens=max_state_tokens,
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            answers = model.answer_packed([packed], [qs])[0]
        elapsed = (time.perf_counter() - t0) * 1000.0
        out = {n: a.to_dict(questions[n]) for n, a in zip(names, answers)}
        if conformal is not None:
            for n in names:
                decorate(out[n], questions[n], conformal)
        return {
            "model": model_name,
            "answers": out,
            "usage": {"input_tokens": len(packed)},
            "latency_ms": round(elapsed, 1),
        }

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="saved model directory or base:<hf id>")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-state-tokens", type=int, default=4096)
    args = ap.parse_args()
    from assay.evaluate import load_model

    model = load_model(args.model)
    model.eval()
    name = os.path.basename(args.model.rstrip("/")) or args.model
    app = create_app(model, name, args.max_state_tokens, conformal=load_conformal(args.model))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
