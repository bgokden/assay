"""HTTP server: POST /v1/decide with a state and typed questions, get distributions back.

    uv run python -m assay.server --model runs/assay-4b --port 8000

Request:
    {"state": <str | object | list>,
     "questions": {"name": {"type": "bool" | "choice" | "score", "instructions": "...",
                            "options": {...} | "levels": [...] | "yes": "...", "no": "..."}}}
Response:
    {"model": "...", "answers": {"name": {...}}, "usage": {"input_tokens": n}, "latency_ms": t}
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from assay.encoding import encode, identity_order
from assay.model import AssayModel
from assay.schema import Question


class DecideRequest(BaseModel):
    state: Any
    questions: dict[str, dict[str, Any]] = Field(min_length=1)


def create_app(model: AssayModel, model_name: str, max_state_tokens: int = 4096) -> FastAPI:
    app = FastAPI(title="assay", version="0.2.0")

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
        return {
            "model": model_name,
            "answers": {n: a.to_dict(questions[n]) for n, a in zip(names, answers)},
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
    uvicorn.run(create_app(model, name, args.max_state_tokens), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
