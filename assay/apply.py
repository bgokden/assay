"""Answer questions, or run an agent, over a file of states.

    python -m assay.apply --model runs/assay-0.6b --questions questions.json \\
        --states states.jsonl --out answers.jsonl

    python -m assay.apply --model runs/assay-0.6b --agent examples/agents/support_triage.json \\
        --states tickets.jsonl --out routed.jsonl

`questions.json` holds the question set in the request shape (`{"name": {"type": ..., ...}}`),
`--states` is one state per line -- a bare JSON value, or an object with a `state` field, which
lets an existing dataset file be scored as it is -- and each output line carries the state's
identifier, if it had one, and the answers. Any tier works, and when the model directory holds
conformal thresholds each answer also carries `act` and `set`.

With `--agent` each line is routed instead: the outcome, the action it stands for and the path
the walk took. When the input records carry an `expected` outcome (top level or under `meta`),
the run ends with how often the agent reached it, counted separately for the cases it routed
and the cases a guard sent to a fallback -- which is the number that tells you whether a
threshold is set where it should be.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterator
from typing import Any

import torch

from assay.agent import Agent
from assay.conformal import decorate, load_conformal
from assay.evaluate import load_model
from assay.schema import Question
from assay.serving import runner_for


def read_states(path: str, limit: int | None = None) -> Iterator[tuple[Any, Any]]:
    """Yields (state, id) pairs. A line that is an object with a `state` field is read as a
    record, so training and evaluation files can be scored without being rewritten."""
    for state, identifier, _ in read_records_for_apply(path, limit):
        yield state, identifier


def read_records_for_apply(path: str, limit: int | None = None) -> Iterator[tuple[Any, Any, Any]]:
    """(state, id, expected outcome). The expected outcome is optional and only used to score
    an agent against labelled cases."""
    with open(path) as f:
        kept = 0
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if limit is not None and kept >= limit:
                return
            kept += 1
            value = json.loads(line)
            if isinstance(value, dict) and "state" in value:
                meta = value.get("meta") or {}
                yield value["state"], meta.get("id", i), value.get("expected", meta.get("expected"))
            else:
                yield value, i, None


def load_questions(path: str) -> dict[str, Question]:
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path} must hold a non-empty object of named questions")
    return {name: Question.from_dict(q) for name, q in raw.items()}


def apply_model(
    model,
    questions: dict[str, Question],
    states: Iterator[tuple[Any, Any]],
    batch_size: int = 8,
    max_state_tokens: int = 4096,
    conformal: dict | None = None,
) -> Iterator[dict[str, Any]]:
    """One record per state, batched the way the server batches concurrent requests."""
    runner = runner_for(model, max_state_tokens)
    names = list(questions)
    qs = [questions[n] for n in names]
    batch: list[tuple[Any, Any]] = []
    for state, identifier in states:
        batch.append((state, identifier))
        if len(batch) == batch_size:
            yield from _answer(runner, batch, names, qs, conformal)
            batch = []
    if batch:
        yield from _answer(runner, batch, names, qs, conformal)


def _answer(runner, batch, names, qs, conformal) -> Iterator[dict[str, Any]]:
    items = [runner.prepare(state, qs) for state, _ in batch]
    with torch.inference_mode():
        results = runner.answer_batch(items, [qs] * len(batch))
    for (state, identifier), answers in zip(batch, results):
        out = {n: a.to_dict(q) for n, a, q in zip(names, answers, qs)}
        if conformal is not None:
            for n, q in zip(names, qs):
                decorate(out[n], q, conformal)
        yield {"id": identifier, "state": state, "answers": out}


def route_states(
    model,
    agent: Agent,
    records: Iterator[tuple[Any, Any, Any]],
    batch_size: int = 8,
    max_state_tokens: int = 4096,
    conformal: dict | None = None,
    keep: list[tuple[dict[str, Any], Any]] | None = None,
) -> Iterator[dict[str, Any]]:
    """One routed decision per state, batched: the agent's whole graph is one pass per state
    and several states share a pass. `keep` collects (answers, expected) pairs so thresholds
    can be swept afterwards without asking the model again."""
    runner = runner_for(model, max_state_tokens)
    questions = agent.questions()
    names = list(questions)
    qs = [questions[n] for n in names]
    batch: list[tuple[Any, Any, Any]] = []

    def flush() -> Iterator[dict[str, Any]]:
        items = [runner.prepare(state, qs) for state, _, _ in batch]
        with torch.inference_mode():
            results = runner.answer_batch(items, [qs] * len(batch))
        for (state, identifier, expected), answers in zip(batch, results):
            by_node = dict(zip(names, answers))
            if keep is not None:
                keep.append((by_node, expected))
            decision = agent.route(by_node, conformal)
            record = {"id": identifier, "state": state, **decision.to_dict()}
            if expected is not None:
                record["expected"] = expected
                record["correct"] = decision.outcome == expected
            yield record

    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            yield from flush()
            batch = []
    if batch:
        yield from flush()


def summarise_routing(records: list[dict[str, Any]]) -> dict[str, Any]:
    """How the agent did, split by whether a guard stopped it. The routed cases are the ones
    the agent acted on by itself; the guarded ones are what it handed over."""
    labelled = [r for r in records if "correct" in r]
    if not labelled:
        return {"routed": len(records)}
    guarded = [r for r in labelled if any("guard" in step for step in r["path"])]
    routed = [r for r in labelled if r not in guarded]

    def rate(rows: list[dict[str, Any]]) -> float | None:
        return round(sum(r["correct"] for r in rows) / len(rows), 4) if rows else None

    return {
        "cases": len(labelled),
        "accuracy": rate(labelled),
        "routed": len(routed),
        "routed_accuracy": rate(routed),
        "handed_over": len(guarded),
        "handed_over_accuracy": rate(guarded),
        "hand_over_rate": round(len(guarded) / len(labelled), 4),
    }


SWEEP_GRID = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def sweep_threshold(
    agent: Agent,
    cases: list[tuple[dict[str, Any], Any]],
    node: str,
    grid: list[float] | None = None,
    conformal: dict | None = None,
) -> list[dict[str, Any]]:
    """What a guard would have done, at each threshold, on answers already computed.

    Routing is pure, so the whole sweep costs nothing but a loop: the model ran once. Each row
    is what you would have got by setting `min_probability` on that node to that value.
    """
    if node not in agent.graph.nodes:
        raise ValueError(f"agent {agent.name!r} has no node called {node!r}")
    original = agent.graph.nodes[node].min_probability
    rows = []
    try:
        for threshold in grid or SWEEP_GRID:
            agent.graph.nodes[node].min_probability = threshold or None
            routed = []
            for answers, expected in cases:
                decision = agent.route(answers, conformal)
                routed.append(
                    {
                        "path": decision.path,
                        "correct": decision.outcome == expected,
                        "outcome": decision.outcome,
                    }
                )
            summary = summarise_routing([r for r in routed if "correct" in r])
            rows.append({"min_probability": threshold, **summary})
    finally:
        agent.graph.nodes[node].min_probability = original
    return rows


def format_sweep(node: str, rows: list[dict[str, Any]]) -> str:
    """The trade-off table: what each threshold buys and what it costs in handovers."""
    lines = [
        f"min_probability on {node!r}:",
        "  threshold  cases  accuracy  routed  routed_acc  handed_over  hand_over_rate",
    ]
    for r in rows:
        routed_accuracy = r.get("routed_accuracy")
        lines.append(
            f"  {r['min_probability']:>9.2f}  {r.get('cases', 0):>5}  "
            f"{r.get('accuracy') or 0:>8.3f}  {r.get('routed', 0):>6}  "
            f"{routed_accuracy if routed_accuracy is not None else float('nan'):>10.3f}  "
            f"{r.get('handed_over', 0):>11}  {r.get('hand_over_rate') or 0:>14.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="a run directory, a Hub id, or base:<hf id>")
    ap.add_argument("--questions", help="JSON object of named questions")
    ap.add_argument("--agent", help="an agent specification to route each state through")
    ap.add_argument(
        "--sweep",
        help="after routing, show what each min_probability on this node would have done",
    )
    ap.add_argument("--states", required=True, help="one state per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8, help="states per forward pass")
    ap.add_argument("--max-state-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, help="score only the first N lines")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--progress-every", type=int, default=1000)
    args = ap.parse_args()

    if bool(args.questions) == bool(args.agent):
        ap.error("give either --questions or --agent")
    agent = Agent.from_file(args.agent) if args.agent else None
    questions = agent.questions() if agent else load_questions(args.questions)
    model = load_model(args.model, device=args.device)
    model.eval()
    conformal = load_conformal(args.model) if os.path.isdir(args.model) else None
    kept: list[tuple[dict[str, Any], Any]] | None = [] if args.sweep else None
    if agent is not None:
        stream = route_states(
            model,
            agent,
            read_records_for_apply(args.states, args.limit),
            batch_size=args.batch_size,
            max_state_tokens=args.max_state_tokens,
            conformal=conformal,
            keep=kept,
        )
    else:
        stream = apply_model(
            model,
            questions,
            read_states(args.states, args.limit),
            batch_size=args.batch_size,
            max_state_tokens=args.max_state_tokens,
            conformal=conformal,
        )
    t0 = time.time()
    written = 0
    routed: list[dict[str, Any]] = []
    with open(args.out, "w") as out:
        for record in stream:
            out.write(json.dumps(record) + "\n")
            written += 1
            if agent is not None:
                routed.append(record)
            if args.progress_every and written % args.progress_every == 0:
                rate = written / max(1e-9, time.time() - t0)
                print(f"{written} states, {rate:.1f}/s", flush=True)
    what = f"agent {agent.name!r}" if agent else f"{len(questions)} questions each"
    print(f"wrote {written} states with {what} to {args.out}")
    if agent is not None:
        print(json.dumps(summarise_routing(routed), indent=2))
    if agent is not None and kept:
        labelled = [(answers, expected) for answers, expected in kept if expected is not None]
        if not labelled:
            print("nothing to sweep: no record carried an expected outcome")
        else:
            print()
            print(
                format_sweep(
                    args.sweep, sweep_threshold(agent, labelled, args.sweep, conformal=conformal)
                )
            )


if __name__ == "__main__":
    main()
