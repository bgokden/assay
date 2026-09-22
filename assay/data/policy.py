"""Synthetic policy-application task: a policy in words, a case of facts, a typed verdict.

Predicates cover numeric comparisons, categorical membership, boolean flags and identity of
two named parties; rules compose them with AND, OR, NOT and IF-THEN-ELSE. Labels are computed
by evaluating the rule. Some cases omit a fact the rule needs: the choice version then labels
them "cannot be determined" and the bool version marks them unanswerable, which trains the
evidence head on a real "the state does not say" signal rather than only on passage swaps.
"""

from __future__ import annotations

import dataclasses
import functools
import random
from typing import Any

from assay.data.registry import Example
from assay.schema import Question

NUMERIC_FACTS = [
    ("order total", 0, 200),
    ("account balance", 0, 500),
    ("shipment weight in kg", 0, 60),
    ("request amount", 0, 1000),
    ("risk score", 0, 100),
    ("days since signup", 0, 400),
    ("seats requested", 1, 40),
    ("number of prior incidents", 0, 12),
]
CATEGORICAL_FACTS = {
    "customer tier": ["bronze", "silver", "gold", "platinum"],
    "region": ["EU", "US", "APAC", "LATAM"],
    "requester role": ["admin", "manager", "staff", "contractor"],
    "payment method": ["card", "invoice", "bank transfer", "cash"],
    "ticket category": ["billing", "technical", "legal", "sales"],
}
BOOLEAN_FACTS = [
    "identity verified",
    "manager approval attached",
    "invoice attached",
    "account in good standing",
    "two-factor authentication enabled",
    "signed agreement on file",
]
PARTY_PAIRS = [("requester", "approver"), ("submitter", "reviewer"), ("payer", "beneficiary")]
NAMES = [
    "Priya",
    "Noah",
    "Mira",
    "Tomas",
    "Kofi",
    "Elena",
    "Yusuf",
    "Hana",
    "Leo",
    "Sofia",
    "Berk",
    "Ana",
]
DISTRACTORS = [
    ("ticket id", lambda rng: str(rng.randrange(1000, 9999))),
    ("routing reference", lambda rng: str(rng.randrange(100, 999))),
    ("submission channel", lambda rng: rng.choice(["web portal", "email", "phone", "mobile app"])),
    ("case colour code", lambda rng: rng.choice(["blue", "green", "amber"])),
]
SUBJECTS = ["request", "refund", "order", "application", "access request", "claim", "transfer"]


@dataclasses.dataclass
class Pred:
    kind: str
    fact: str
    text: str
    fn: Any  # facts dict -> bool


TEXT_FIELDS = {
    "ticket subject": [
        "refund",
        "invoice",
        "outage",
        "password",
        "upgrade",
        "cancel",
        "shipping",
        "duplicate",
        "urgent",
        "renewal",
    ],
    "internal note": [
        "verified",
        "escalated",
        "fraud",
        "vip",
        "chargeback",
        "resolved",
        "pending",
        "legal",
    ],
}


def make_predicate(rng: random.Random, facts: dict[str, Any], hard: bool = False) -> Pred:
    kinds = ["numeric", "numeric", "categorical", "boolean", "identity"]
    if hard:
        kinds = kinds + ["contains", "contains"]
    kind = rng.choice(kinds)
    if kind == "contains":
        name = rng.choice(list(TEXT_FIELDS))
        words = rng.sample(TEXT_FIELDS[name], rng.choice([2, 3, 4]))
        facts[name] = " ".join(words)
        target = rng.choice(TEXT_FIELDS[name])
        if rng.random() < 0.5:
            return Pred(
                kind,
                name,
                f'the {name} contains the word "{target}"',
                lambda f, n=name, t=target: t in f[n].split(),
            )
        return Pred(
            kind,
            name,
            f'the {name} does not contain the word "{target}"',
            lambda f, n=name, t=target: t not in f[n].split(),
        )
    if kind == "numeric":
        name, lo, hi = rng.choice(NUMERIC_FACTS)
        facts[name] = rng.randint(lo, hi)
        op = rng.choice(["gt", "ge", "lt", "le", "between"])
        if op == "between":
            a = rng.randint(lo, hi - 1)
            b = rng.randint(a + 1, hi)
            text = rng.choice(
                [
                    f"the {name} is between {a} and {b}, including both ends",
                    f"the {name} is at least {a} and at most {b}",
                ]
            )
            return Pred(kind, name, text, lambda f, a=a, b=b, n=name: a <= f[n] <= b)
        t = rng.randint(lo + 1, hi - 1)
        texts = {
            "gt": [
                f"the {name} is greater than {t}",
                f"the {name} exceeds {t}",
                f"the {name} is above {t}",
            ],
            "ge": [f"the {name} is at least {t}", f"the {name} is {t} or more"],
            "lt": [
                f"the {name} is less than {t}",
                f"the {name} is below {t}",
                f"the {name} is under {t}",
            ],
            "le": [
                f"the {name} is at most {t}",
                f"the {name} is {t} or less",
                f"the {name} does not exceed {t}",
            ],
        }
        fns = {
            "gt": lambda f, t=t, n=name: f[n] > t,
            "ge": lambda f, t=t, n=name: f[n] >= t,
            "lt": lambda f, t=t, n=name: f[n] < t,
            "le": lambda f, t=t, n=name: f[n] <= t,
        }
        return Pred(kind, name, rng.choice(texts[op]), fns[op])
    if kind == "categorical":
        name = rng.choice(list(CATEGORICAL_FACTS))
        values = CATEGORICAL_FACTS[name]
        facts[name] = rng.choice(values)
        if rng.random() < 0.5:
            v = rng.choice(values)
            return Pred(kind, name, f"the {name} is {v}", lambda f, v=v, n=name: f[n] == v)
        subset = rng.sample(values, 2)
        return Pred(
            kind,
            name,
            f"the {name} is {subset[0]} or {subset[1]}",
            lambda f, s=subset, n=name: f[n] in s,
        )
    if kind == "boolean":
        name = rng.choice(BOOLEAN_FACTS)
        facts[name] = rng.choice([True, False])
        if rng.random() < 0.5:
            return Pred(kind, name, f"{name} is yes", lambda f, n=name: f[n] is True)
        return Pred(kind, name, f"{name} is no", lambda f, n=name: f[n] is False)
    a, b = rng.choice(PARTY_PAIRS)
    same = rng.random() < 0.4
    person_a = rng.choice(NAMES)
    person_b = person_a if same else rng.choice([n for n in NAMES if n != person_a])
    facts[a] = person_a
    facts[b] = person_b
    key = f"{a}/{b}"
    if rng.random() < 0.5:
        return Pred(
            kind, key, f"the {a} is the same person as the {b}", lambda f, a=a, b=b: f[a] == f[b]
        )
    return Pred(
        kind, key, f"the {a} and the {b} are different people", lambda f, a=a, b=b: f[a] != f[b]
    )


@dataclasses.dataclass
class Rule:
    text: str
    fn: Any
    preds: list[Pred]


def make_rule(
    rng: random.Random, facts: dict[str, Any], depth: int = 0, hard: bool = False
) -> Rule:
    max_depth = 3 if hard else 2
    if hard:
        shape = rng.choice(
            ["and", "or", "not", "ifelse", "and", "or"]
            if depth == 0
            else ["pred", "and", "or", "not", "ifelse"]
        )
    else:
        shape = rng.choice(
            ["pred", "and", "or", "not", "ifelse"]
            if depth == 0
            else ["pred", "pred", "and", "or", "not"]
        )
    if shape == "pred" or depth >= max_depth:
        p = make_predicate(rng, facts, hard)
        return Rule(p.text, p.fn, [p])
    if shape in ("and", "or"):
        n = rng.choice([2, 2, 3, 4] if hard else [2, 2, 3])
        parts = [make_rule(rng, facts, depth + 1, hard) for _ in range(n)]
        preds = [p for r in parts for p in r.preds]
        if shape == "and":
            text = rng.choice(
                [
                    "all of the following hold: ",
                    "both of the following hold: "
                    if n == 2
                    else "every one of the following holds: ",
                ]
            ) + "; ".join(f"({r.text})" for r in parts)
            return Rule(text, lambda f, parts=parts: all(r.fn(f) for r in parts), preds)
        text = rng.choice(
            ["at least one of the following holds: ", "any of the following holds: "]
        ) + "; ".join(f"({r.text})" for r in parts)
        return Rule(text, lambda f, parts=parts: any(r.fn(f) for r in parts), preds)
    if shape == "not":
        inner = make_rule(rng, facts, depth + 1, hard)
        return Rule(
            f"it is not the case that ({inner.text})",
            lambda f, inner=inner: not inner.fn(f),
            inner.preds,
        )
    cond = make_rule(rng, facts, depth + 1, hard)
    then = make_rule(rng, facts, depth + 1, hard)
    other = make_rule(rng, facts, depth + 1, hard)
    text = f"if ({cond.text}) then the requirement is ({then.text}); otherwise the requirement is ({other.text})"
    return Rule(
        text,
        lambda f, cond=cond, then=then, other=other: then.fn(f) if cond.fn(f) else other.fn(f),
        cond.preds + then.preds + other.preds,
    )


PAST = {
    "approve": "approved",
    "accept": "accepted",
    "allow": "allowed",
    "deny": "denied",
    "reject": "rejected",
    "refuse": "refused",
}


def render_policy(rng: random.Random, subject: str, rule: Rule) -> str:
    verb = rng.choice(["approve", "accept", "allow"])
    deny = {"approve": "deny", "accept": "reject", "allow": "refuse"}[verb]
    style = 0 if rule.text.startswith("if (") else rng.randrange(3)
    if style == 0:
        return f"{verb.capitalize()} the {subject} exactly when {rule.text}. Otherwise {deny} it."
    if style == 1:
        return (
            f"A {subject} is {PAST[verb]} only if {rule.text}; any other {subject} is {PAST[deny]}."
        )
    return f"Policy: {deny} the {subject} unless {rule.text}."


def render_fact(name: str, value: Any) -> str:
    if isinstance(value, bool):
        return f"{name.capitalize()}: {'yes' if value else 'no'}."
    if name in TEXT_FIELDS:
        return f'The {name} reads: "{value}".'
    if "/" in name:
        raise ValueError("identity keys are rendered per party")
    return f"The {name} is {value}."


def render_case(rng: random.Random, facts: dict[str, Any], omit: str | None) -> str:
    lines = []
    for name, value in facts.items():
        if name == omit:
            continue
        lines.append(render_fact(name, value))
    for name, gen in rng.sample(DISTRACTORS, rng.choice([1, 2])):
        lines.append(f"The {name} is {gen(rng)}.")
    rng.shuffle(lines)
    return " ".join(lines)


def needed_fact_names(pred: Pred) -> list[str]:
    return pred.fact.split("/") if pred.kind == "identity" else [pred.fact]


def outcome_depends_on(rule: Rule, facts: dict[str, Any], name: str) -> bool:
    """True if changing the omitted fact can flip the verdict (so it is truly required)."""
    base = rule.fn(facts)
    candidates: list[Any] = []
    value = facts[name]
    if isinstance(value, bool):
        candidates = [not value]
    elif isinstance(value, int):
        candidates = [v for v in range(0, 1001, 7)]
    elif name in CATEGORICAL_FACTS:
        candidates = CATEGORICAL_FACTS[name]
    elif name in TEXT_FIELDS:
        candidates = [
            " ".join(TEXT_FIELDS[name][i : i + 3]) for i in range(len(TEXT_FIELDS[name]) - 2)
        ]
    else:
        candidates = NAMES
    for c in candidates:
        trial = dict(facts)
        trial[name] = c
        if rule.fn(trial) != base:
            return True
    return False


def generate(n: int, seed: int, hard: bool = False) -> list[Example]:
    rng = random.Random(seed)
    out: list[Example] = []
    while len(out) < n:
        facts: dict[str, Any] = {}
        rule = make_rule(rng, facts, hard=hard)
        subject = rng.choice(SUBJECTS)
        policy = render_policy(rng, subject, rule)
        verdict = rule.fn(facts)
        omit = None
        if rng.random() < 0.2:
            candidates = [
                nm
                for p in rule.preds
                for nm in needed_fact_names(p)
                if outcome_depends_on(rule, facts, nm)
            ]
            if candidates:
                omit = rng.choice(candidates)
        state = {"policy": policy, "case": render_case(rng, facts, omit)}
        kind = rng.choices(["bool", "choice", "choice3"], weights=[0.4, 0.3, 0.3])[0]
        if kind == "bool":
            q = Question(
                type="bool",
                instructions=rng.choice(
                    [
                        f"Under the policy, is the {subject} approved?",
                        f"Does the case satisfy the policy for this {subject}?",
                    ]
                ),
            )
            out.append(
                Example(
                    state=state,
                    question=q,
                    label=bool(verdict),
                    answerable=omit is None,
                    name="verdict",
                )
            )
        elif kind == "choice":
            if omit is not None:
                continue
            q = Question(
                type="choice",
                instructions=rng.choice(
                    ["Apply the policy to the case.", f"What is the verdict on this {subject}?"]
                ),
                options={
                    "approve": f"The policy permits this {subject}",
                    "reject": f"The policy does not permit this {subject}",
                },
            )
            out.append(
                Example(
                    state=state,
                    question=q,
                    label="approve" if verdict else "reject",
                    name="verdict",
                )
            )
        else:
            q = Question(
                type="choice",
                instructions=rng.choice(
                    ["Apply the policy to the case.", f"What is the verdict on this {subject}?"]
                ),
                options={
                    "approve": f"The policy permits this {subject}",
                    "reject": f"The policy does not permit this {subject}",
                    "cannot_be_determined": "A fact the policy depends on is missing from the case",
                },
            )
            label = (
                "cannot_be_determined" if omit is not None else ("approve" if verdict else "reject")
            )
            out.append(Example(state=state, question=q, label=label, name="verdict"))
    return out


generate_hard = functools.partial(generate, hard=True)
