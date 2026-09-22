"""Walk a decision tree over one state in a single forward pass.

Every question in the graph is answered at once -- the branches cannot see each other, so the
walk only chooses which answers to follow -- and a node that is not confident enough falls
back instead of guessing.

    python examples/decision_graph.py --model Berk/assay-0.6b
"""

import argparse
import json

from assay import load_model
from assay.graph import Graph, walk

TICKET = {
    "channel": "email",
    "message": (
        "The dashboard has been down for the whole team since this morning and we are losing "
        "orders. Nobody can log in either."
    ),
}

GRAPH = {
    "start": "triage",
    "nodes": {
        "triage": {
            "question": {
                "type": "choice",
                "instructions": "Which team should handle this ticket?",
                "options": {
                    "billing": "Charges, invoices and refunds",
                    "technical": "Faults, errors and outages",
                    "account": "Sign-in, passwords and profile changes",
                },
            },
            "edges": {"billing": "refund", "technical": "outage", "account": "identity"},
            "min_probability": 0.45,
            "min_evidence": 0.2,
            "fallback": "human",
        },
        "refund": {
            "question": {"type": "bool", "instructions": "Is the customer asking for money back?"},
            "edges": {"yes": "issue_refund", "no": "reply"},
        },
        "outage": {
            "question": {
                "type": "bool",
                "instructions": "Does this describe an outage affecting several people?",
            },
            "edges": {"yes": "page_oncall"},
            "default": "reply",
        },
        "identity": {
            "question": {
                "type": "bool",
                "instructions": "Does the customer need help signing in?",
            },
            "edges": {"yes": "reset_access", "no": "reply"},
        },
        "issue_refund": {"outcome": "issue_refund"},
        "page_oncall": {"outcome": "page_oncall"},
        "reset_access": {"outcome": "reset_access"},
        "reply": {"outcome": "reply"},
        "human": {"outcome": "human_review"},
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Berk/assay-0.6b")
    ap.add_argument("--device", default="cuda", help="where the small tiers load")
    args = ap.parse_args()

    graph = Graph.from_dict(GRAPH)
    model = load_model(args.model, device=args.device)
    answers = model.answer(state=TICKET, questions=graph.questions())
    result = walk(graph, answers)

    print(f"outcome: {result['outcome']} (path probability {result['path_probability']:.3f})")
    for step in result["path"]:
        guard = f"  [{step['guard']}]" if "guard" in step else ""
        print(f"  {step['node']}: {step['chosen']} p={step['probability']:.3f}{guard}")
    print(f"\n{len(graph.questions())} questions, one forward pass")
    print(json.dumps({name: a.confidence for name, a in answers.items()}, indent=2))


if __name__ == "__main__":
    main()
