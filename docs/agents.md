# Agents

An agent here is not a loop that talks to itself. It is a decision graph plus the actions its
outcomes stand for: typed questions over a state, routed by calibrated probabilities, ending
at something you do. Deciding is **one forward pass** -- every question in the graph is
answered at once, because the question branches cannot see each other -- and the walk is then
pure logic. A tree with twelve questions costs what one question costs.

The split that matters: **the agent decides, the caller acts.** `route` is a pure function
from answers to an outcome and an action; nothing in this library sends an email, issues a
refund or pages anyone. That is what makes an agent testable, and what lets the same
specification run in a script, in a worker and behind an HTTP server without changing.

## A specification

```json
{
  "name": "support-triage",
  "description": "Route a support ticket, then say what should happen to it",
  "max_steps": 2,
  "graph": {
    "start": "triage",
    "nodes": {
      "triage": {
        "question": {
          "type": "choice",
          "instructions": "Which team should handle this ticket?",
          "options": {"billing": "Charges and refunds", "technical": "Faults and outages"}
        },
        "edges": {"billing": "refund", "technical": "outage"},
        "min_probability": 0.45,
        "min_evidence": 0.2,
        "fallback": "human"
      },
      "refund": {
        "question": {"type": "bool", "instructions": "Is the customer asking for money back?"},
        "edges": {"yes": "pay", "no": "reply"}
      },
      "outage": {
        "question": {"type": "bool", "instructions": "Is a service outage described?"},
        "edges": {"yes": "page"},
        "default": "reply"
      },
      "pay": {"outcome": "issue_refund"},
      "page": {"outcome": "page_oncall"},
      "reply": {"outcome": "reply"},
      "human": {"outcome": "human_review"}
    }
  },
  "actions": {
    "issue_refund": {"action": "refund", "arguments": {"queue": "billing", "limit": 500}},
    "page_oncall": {"action": "page", "arguments": {"rota": "infrastructure"}},
    "human_review": {"action": "escalate", "arguments": {"reason": "the model was not sure"}}
  }
}
```

A runnable copy is `examples/agents/support_triage.json`, and `examples/run_agent.py` runs it
with handlers.

### Nodes

A node either asks a question or ends the walk:

| field | meaning |
|---|---|
| `question` | a typed question: `bool` (`noul`), `choice` with `options`, `score` with `levels` |
| `edges` | option key to the next node, e.g. `{"yes": "pay", "no": "reply"}`; score keys are level indices as strings |
| `default` | taken when the chosen option has no edge |
| `outcome` | ends the walk with this value; a node has this or a question, never both |

### Guards: "not sure" is a route

| field | meaning |
|---|---|
| `min_probability` | the top option must reach this probability |
| `min_evidence` | the state must support answering at all |
| `require_act` | the answer must clear the model's fitted conformal act threshold |
| `fallback` | where the case goes when a guard is not met |

This is the part worth designing carefully. The model is calibrated, so `min_probability` is a
real error budget rather than a vibe: at 0.45 on a three-way choice you are routing only when
the model is roughly twice as sure as chance. `require_act` goes further and uses the
thresholds fitted by `assay.conformal`, which come with a bound on the error rate among the
cases you do act on -- the server passes them in automatically when the model directory has
them. A node with a guard and no `fallback` stops the walk instead, and the reply says why.

### Actions

`actions` maps an **outcome** to an action name and its arguments. Outcomes with no entry are
returned as themselves, which is what a read-only classifier wants. An action for an outcome
the graph can never reach is rejected when the specification is loaded, so a typo is an error
at startup rather than a branch that silently does nothing.

`max_steps` (1 to 8) allows a loop: if a handler returns a new state, the agent decides again
on it. Use it when an action genuinely changes what is known -- fetching an order, asking the
customer one question -- not as a way to let the model think longer, which it does not do.

## In Python

```python
from assay import load_model
from assay.agent import Agent
from assay.conformal import load_conformal

agent = Agent.from_file("examples/agents/support_triage.json")
model = load_model("Berk/assay-4b")
conformal = load_conformal("Berk/assay-4b")

def refund(state, queue, limit):
    ...  # your code
    return {"refunded": True}

run = agent.run(model, {"message": "I was charged twice"}, {"refund": refund}, conformal)
run.outcome            # "issue_refund"
run.last.action        # Action(name="refund", arguments={"queue": "billing", "limit": 500})
run.last.path          # every node visited, with probability, confidence, evidence
run.last.path_probability
run.result             # what your handler returned
```

A handler is called as `handler(state, **arguments)`. Returning a mapping with a `state` key
continues the loop with that state; anything else ends the run and becomes `result`. Without a
handler for the action, `run` stops and hands you the planned action to carry out yourself.

To decide without acting at all:

```python
decision, answers = agent.decide(model, state, conformal)
decision.outcome, decision.action, decision.complete, decision.stopped
```

## Over HTTP

Register agents at startup from a directory, or create them at runtime:

```bash
uv run python -m assay.server --model Berk/assay-4b --agents examples/agents --port 8000

curl -s localhost:8000/v1/agents
curl -s localhost:8000/v1/agents/support-triage/run -H 'content-type: application/json' \
  -d '{"state": {"channel": "email", "message": "The dashboard is down for everyone."}}'
```

```json
{"model": "assay-4b", "agent": "support-triage", "outcome": "page_oncall",
 "action": "page", "arguments": {"rota": "infrastructure"},
 "path": [{"node": "triage", "chosen": "technical", "probability": 0.95, "confidence": 0.93,
           "evidence": 1.0},
          {"node": "outage", "chosen": "yes", "probability": 0.97, "confidence": 0.94,
           "evidence": 1.0}],
 "path_probability": 0.9215, "complete": true,
 "answers": {"triage": {}, "outage": {}},
 "usage": {"input_tokens": 74, "questions": 5, "forward_passes": 1}, "latency_ms": 31.2}
```

The server performs no actions. It tells you what should happen; your worker does it, and
calls again with the new state when the agent has further steps. `POST /v1/agents` registers a
specification for the life of the process -- convenient while building one -- and anything
that must survive a restart belongs in a file under `--agents`.

## Measuring one

An agent that routes confidently and wrongly is worse than no agent, so measure it on cases
whose outcome you know. Put the expected outcome on each record and route the file:

```bash
uv run python -m assay.apply --model Berk/assay-4b --agent examples/agents/support_triage.json \
    --states labelled_tickets.jsonl --out routed.jsonl
```

```json
{"state": {"message": "..."}, "expected": "issue_refund", "meta": {"id": "ticket/1"}}
```

Each output line carries the outcome, the action, the path and `correct`, and the run ends
with a summary:

```json
{"cases": 500, "accuracy": 0.86,
 "routed": 412, "routed_accuracy": 0.93,
 "handed_over": 88, "handed_over_accuracy": 0.55, "hand_over_rate": 0.176}
```

Read it as three numbers, not one. `routed_accuracy` is how often the agent was right when it
acted on its own; `hand_over_rate` is what that cost you in human work; `handed_over_accuracy`
is what the agent would have scored on the cases it declined -- if that is close to
`routed_accuracy`, your guard is too tight and you are paying people to confirm what the model
already knew. Keep the file as a regression test for the next model.

To pick the threshold, sweep it. Routing is pure, so every threshold is a re-route of answers
the model already gave: the sweep costs one pass over the file, not one per value.

```bash
uv run python -m assay.apply --model Berk/assay-4b --agent examples/agents/support_triage.json \
    --states labelled_tickets.jsonl --out routed.jsonl --sweep triage
```

```
min_probability on 'triage':
  threshold  cases  accuracy  routed  routed_acc  handed_over  hand_over_rate
       0.00    500     0.831     500       0.831            0           0.000
       0.50    500     0.847     476       0.863           24           0.048
       0.70    500     0.862     431       0.901           69           0.138
       0.90    500     0.844     338       0.944          162           0.324
```

Accuracy here counts a handover as wrong unless the case was labelled for the fallback, so the
column bends: guarding too little acts on cases it should not, guarding too much hands over
cases it would have got right. Pick the row where `routed_acc` is high enough for the action
to be safe and `hand_over_rate` is work you can actually absorb.

## Designing one that works

**Ask for what the state can support.** The `evidence` score exists because a question whose
answer is not in the state gets a confident-looking guess otherwise. Guard the first node with
`min_evidence` and route the rest to a person.

**Keep branches shallow and questions independent.** Every question is answered against the
same state without seeing the others' answers, so a question that only makes sense given an
earlier answer ("how large is the refund they asked for?") is still answered for every ticket.
That costs nothing, but it means the question must make sense on its own.

**Do not ask the model to compute.** One forward pass will not do arithmetic, compare dates or
chain three facts. Ask it to classify and judge; keep the arithmetic in the handler.

**Put the error budget in the graph, not in the prompt.** A guard and a fallback are how you
say "I would rather pay a human than be wrong here", and the calibration is what makes that
trade explicit. Measure it: run `assay.apply` over labelled cases and look at what the
abstained set contains before you tune a threshold.
