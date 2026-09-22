"""Synthetic date-reasoning task: a deadline rule in words, two dates in varied formats, a
typed verdict computed exactly in code.

Covers due dates with grace periods, return and warranty windows, and subscription renewals.
Dates appear as ISO, long, short and weekday-prefixed forms; relative phrasings ("three days
after the due date") appear in a share of cases. Ambiguous numeric forms (07/04) are avoided.
"""

from __future__ import annotations

import datetime
import random

from assay.data.registry import Example
from assay.schema import Question

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def fmt_date(d: datetime.date, rng: random.Random) -> str:
    style = rng.randrange(5)
    if style == 0:
        return d.isoformat()
    if style == 1:
        return d.strftime("%B %-d, %Y")
    if style == 2:
        return d.strftime("%-d %B %Y")
    if style == 3:
        return d.strftime("%b %-d, %Y")
    return f"{WEEKDAYS[d.weekday()]}, {d.strftime('%-d %B %Y')}"


SCENARIOS = [
    {
        "policy": "Reports received on or before the due date are on time. Reports received within {grace} days after the due date are late but accepted. Later reports are refused.",
        "anchor": "due date", "event": "report", "arrival": "was received",
        "levels": ["On time", "Late but accepted", "Refused"],
        "score_q": "How late is the report?", "bool_q": "Was the report received on time?",
    },
    {
        "policy": "Items may be returned within {grace} days of delivery for a full refund. Returns after that are refused.",
        "anchor": "delivery date", "event": "return request", "arrival": "was made",
        "levels": ["Within the return window", "Outside the return window"],
        "score_q": "Is the return request inside the window?", "bool_q": "Is the return request within the return window?",
        "two_level": True,
    },
    {
        "policy": "Warranty claims are covered when filed within {grace} days of the purchase date, and are declined afterwards.",
        "anchor": "purchase date", "event": "claim", "arrival": "was filed",
        "levels": ["Covered by warranty", "Declined, outside the warranty period"],
        "score_q": "Is the claim covered?", "bool_q": "Is the warranty claim covered?",
        "two_level": True,
    },
    {
        "policy": "Payments received by the due date incur no fee. Payments received within {grace} days after the due date incur a late fee. Later payments cause the account to be suspended.",
        "anchor": "due date", "event": "payment", "arrival": "was received",
        "levels": ["No fee", "Late fee", "Account suspended"],
        "score_q": "What is the consequence for this payment?", "bool_q": "Was the payment received by the due date?",
    },
]

DISTRACTORS = [
    "The {event} was submitted through the online portal.",
    "The reference number is {ref}.",
    "The customer's account manager is {name}.",
    "The {event} was marked as priority {prio}.",
]
NAMES = ["Priya", "Noah", "Mira", "Tomas", "Kofi", "Elena", "Yusuf", "Hana"]


def generate(n: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    out: list[Example] = []
    base = datetime.date(2024, 1, 1)
    while len(out) < n:
        sc = rng.choice(SCENARIOS)
        grace = rng.choice([3, 5, 7, 10, 14, 21, 30, 45, 60, 90])
        anchor = base + datetime.timedelta(days=rng.randrange(0, 1000))
        # offsets clustered around the two boundaries so the model cannot guess from magnitude
        offset = rng.choice(
            [rng.randint(-40, -1), rng.randint(-3, 0), 0, rng.randint(1, grace), grace, grace + 1, rng.randint(grace + 1, grace + 40)]
        )
        arrival = anchor + datetime.timedelta(days=offset)
        if offset <= 0:
            level = 0
        elif offset <= grace:
            level = 1
        else:
            level = 2
        two_level = sc.get("two_level", False)
        if two_level:
            level = 0 if offset <= grace else 1
        relative = rng.random() < 0.25
        if relative and offset != 0:
            direction = "after" if offset > 0 else "before"
            arrival_text = f"{abs(offset)} days {direction} the {sc['anchor']}"
        else:
            arrival_text = fmt_date(arrival, rng)
        facts = [
            f"The {sc['anchor']} was {fmt_date(anchor, rng)}.",
            f"The {sc['event']} {sc['arrival']} on {arrival_text}." if not relative or offset == 0 else f"The {sc['event']} {sc['arrival']} {arrival_text}.",
        ]
        for _ in range(rng.choice([1, 2])):
            facts.append(rng.choice(DISTRACTORS).format(event=sc["event"], ref=rng.randrange(1000, 9999), name=rng.choice(NAMES), prio=rng.choice(["low", "normal", "high"])))
        rng.shuffle(facts)
        state = {"policy": sc["policy"].format(grace=grace), "case": " ".join(facts)}
        kind = rng.choices(["score", "bool", "choice"], weights=[0.5, 0.3, 0.2])[0]
        if kind == "score" and not two_level:
            q = Question(type="score", instructions=sc["score_q"], levels=list(sc["levels"]))
            out.append(Example(state=state, question=q, label=level, name="verdict"))
        elif kind == "bool" or two_level and kind == "score":
            q = Question(type="bool", instructions=sc["bool_q"])
            on_time = (offset <= 0) if not two_level else (level == 0)
            out.append(Example(state=state, question=q, label=on_time, name="verdict"))
        else:
            keys = [lvl.lower().replace(",", "").replace(" ", "_") for lvl in sc["levels"]]
            q = Question(type="choice", instructions="Apply the policy to the case.", options={k: lvl for k, lvl in zip(keys, sc["levels"])})
            out.append(Example(state=state, question=q, label=keys[level], name="verdict"))
    return out
