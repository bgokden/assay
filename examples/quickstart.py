"""Ask one state the three question types and read the calibrated answers.

    python examples/quickstart.py --model Berk/assay-0.6b

Any tier works: a Hub id or a run directory for the decoder models, or an encoder-tier
directory with --tier encoder, which runs on a CPU.
"""

import argparse

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


def load(model: str, tier: str, device: str):
    if tier == "encoder":
        from assay.compiled import load_any

        return load_any(model, device=device)
    from assay.model import AssayModel

    return AssayModel.from_pretrained(model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Berk/assay-0.6b")
    ap.add_argument("--tier", choices=["decoder", "encoder"], default="decoder")
    ap.add_argument("--device", default="cpu", help="only used by the encoder tier")
    args = ap.parse_args()

    model = load(args.model, args.tier, args.device)
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
