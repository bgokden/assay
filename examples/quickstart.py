"""Ask one state the three question types and read the calibrated answers.

    python examples/quickstart.py --model Berk/assay-0.6b

Any tier works: `load_model` reads the tier from the saved configuration. The small tiers
run on a CPU with --device cpu.
"""

import argparse

from assay import load_model
from assay.schema import Question

TICKET = {
    "channel": "email",
    "message": (
        "I was charged twice for order A-104 last week and the second charge is still there. "
        "We cannot close the books until this is fixed."
    ),
}

QUESTIONS = {
    "route": Question(
        type="choice",
        instructions="Which team should handle this ticket?",
        options={
            "billing": "Charges, invoices and refunds",
            "technical": "Faults, errors and outages",
            "account": "Sign-in, passwords and profile changes",
        },
    ),
    "refund_requested": Question(
        type="bool", instructions="Is the customer asking for money back?"
    ),
    "urgency": Question(
        type="score",
        instructions="How urgent is this ticket?",
        levels=["Can wait", "Needs attention today", "Blocking work right now"],
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Berk/assay-0.6b")
    ap.add_argument("--device", default="cuda", help="where the small tiers load")
    args = ap.parse_args()

    model = load_model(args.model, device=args.device)
    answers = model.answer(state=TICKET, questions=QUESTIONS)

    route = answers["route"]
    print(f"route:            {route.argmax}  {route.probabilities}")
    print(f"  confidence {route.confidence:.2f}, evidence {route.evidence:.2f}")
    refund = answers["refund_requested"]
    print(f"refund_requested: p_true {refund.p_true:.3f}, confidence {refund.confidence:.2f}")
    urgency = answers["urgency"]
    levels = QUESTIONS["urgency"].levels
    print(f"urgency:          {urgency.score:.2f} of 2 ({levels[int(urgency.argmax)]})")
    print("\nNo text was generated: every number above is a distribution over the option labels.")


if __name__ == "__main__":
    main()
