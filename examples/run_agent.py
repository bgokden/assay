"""Run an agent over a ticket: one forward pass decides, your handlers act.

    python examples/run_agent.py --model Berk/assay-0.6b
    python examples/run_agent.py --model Berk/assay-0.6b --ticket "The dashboard is down."

The agent is `examples/agents/support_triage.json`: a decision graph whose outcomes name
actions. The handlers below are the only code that touches the outside world, and the one for
`escalate` shows what the guards are for -- when the model is not confident enough the case
goes to a person instead of down a branch.
"""

import argparse
import json

from assay import load_model
from assay.agent import Agent
from assay.conformal import load_conformal

TICKETS = {
    "refund": "I was charged twice for order A-104 last week. Please refund the second charge.",
    "outage": "The dashboard has been down for the whole team since this morning.",
    "access": "I cannot sign in since the password reset email never arrived.",
}


def refund(state, queue, limit):
    print(f"  -> would open a refund in the {queue} queue (limit {limit})")
    return {"refunded": True}


def page(state, rota):
    print(f"  -> would page the {rota} rota")
    return {"paged": True}


def reset_password(state):
    print("  -> would send a password reset")
    return {"reset": True}


def draft_reply(state):
    print("  -> would draft a reply for a human to send")
    return {"drafted": True}


def escalate(state, reason):
    print(f"  -> would hand this to a person: {reason}")
    return {"escalated": True}


HANDLERS = {
    "refund": refund,
    "page": page,
    "reset_password": reset_password,
    "draft_reply": draft_reply,
    "escalate": escalate,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Berk/assay-0.6b")
    ap.add_argument("--agent", default="examples/agents/support_triage.json")
    ap.add_argument("--ticket", help="one ticket; without it, three examples run")
    ap.add_argument("--device", default="cuda", help="where the small tiers load")
    args = ap.parse_args()

    agent = Agent.from_file(args.agent)
    model = load_model(args.model, device=args.device)
    conformal = load_conformal(args.model)
    tickets = {"yours": args.ticket} if args.ticket else TICKETS

    for label, message in tickets.items():
        print(f"\n{label}: {message}")
        run = agent.run(model, {"channel": "email", "message": message}, HANDLERS, conformal)
        decision = run.last
        print(f"  outcome: {decision.outcome} (path probability {decision.path_probability:.3f})")
        for step in decision.path:
            guard = f"  [{step['guard']}]" if "guard" in step else ""
            print(f"    {step['node']}: {step['chosen']} p={step['probability']:.2f}{guard}")
        print(f"  result: {json.dumps(run.result)}")

    print(
        f"\nEach ticket cost one forward pass over {len(agent.questions())} questions; "
        "the branches cannot see each other, so the whole tree is answered at once."
    )


if __name__ == "__main__":
    main()
