"""Build a small support-desk dataset in the Assay record format.

The point of the example is the shape of the data, not the data itself: every line is one
state with the three question types over it, which is all a tier needs to train, calibrate
and abstain. Run it before the pipeline:

    python examples/support_data.py --out examples/data/support
"""

import argparse
import json
import os
import random

TEAMS = {
    "billing": "Charges, invoices and refunds",
    "technical": "Faults, errors and outages",
    "account": "Sign-in, passwords and profile changes",
}

TEMPLATES = {
    "billing": [
        "I was charged twice for order {order} and would like the second charge back.",
        "My invoice for {month} shows {amount}, which is more than the plan I signed up for.",
        "Please cancel the subscription and refund the last payment of {amount}.",
        "The discount code was not applied to order {order}; can you correct the total?",
    ],
    "technical": [
        "The dashboard has been returning a 500 error since {month} for everyone on the team.",
        "Exports fail halfway through with 'connection reset' on order {order}.",
        "The mobile app crashes as soon as I open the reports tab.",
        "Webhooks stopped arriving this morning and the retry queue is empty.",
    ],
    "account": [
        "I cannot sign in after the password reset email for {email} never arrived.",
        "Please remove {email} from the workspace, that colleague has left.",
        "Two-factor codes are rejected even though the clock on my phone is right.",
        "I need to change the billing contact on the account to {email}.",
    ],
}

REFUND_WORDS = ("refund", "back", "cancel the subscription", "correct the total")
URGENCY = ["Can wait", "Needs attention today", "Blocking work right now"]
PRESSURE = {
    0: ["", " No rush.", " Whenever you get a chance."],
    1: [" Please look at this today.", " We would like an answer this afternoon."],
    2: [" This is blocking the whole team.", " We cannot work until it is fixed."],
}

QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "options": TEAMS,
    },
    "refund_requested": {
        "type": "bool",
        "instructions": "Is the customer asking for money back?",
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this ticket?",
        "levels": URGENCY,
    },
}


def record(rng: random.Random, index: int, split: str) -> dict:
    team = rng.choice(list(TEMPLATES))
    text = rng.choice(TEMPLATES[team]).format(
        order=f"A-{rng.randint(100, 999)}",
        month=rng.choice(["January", "February", "March", "April"]),
        amount=f"{rng.randint(10, 400)} EUR",
        email=f"{rng.choice(['alex', 'sam', 'robin', 'kim'])}@example.com",
    )
    urgency = rng.randint(0, 2)
    text += rng.choice(PRESSURE[urgency])
    refund = any(word in text for word in REFUND_WORDS)
    questions = json.loads(json.dumps(QUESTIONS))  # a fresh copy per line
    questions["route"]["label"] = team
    questions["refund_requested"]["label"] = refund
    questions["urgency"]["label"] = urgency
    return {
        "state": {"channel": "email", "message": text},
        "questions": questions,
        "meta": {"source": "support_example", "split": split, "id": f"support/{split}/{index}"},
    }


def write(path: str, rng: random.Random, n: int, split: str) -> None:
    with open(path, "w") as f:
        f.writelines(json.dumps(record(rng, i, split)) + "\n" for i in range(n))
    print(f"wrote {n} records to {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="examples/data/support")
    ap.add_argument("--train", type=int, default=1200)
    ap.add_argument("--eval", type=int, default=200, help="size of dev, calibration and holdout")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    write(os.path.join(args.out, "train.jsonl"), rng, args.train, "train")
    for split in ("dev", "calibration", "holdout"):
        write(os.path.join(args.out, f"{split}.jsonl"), rng, args.eval, split)


if __name__ == "__main__":
    main()
