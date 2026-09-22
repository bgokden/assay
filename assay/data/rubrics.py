"""A bank of generic questions for distillation: rubrics that make sense over most texts, in
the three answer types, across support, reviews, news, dialogue, moderation, documents and
policy-shaped states. Teacher models label (state, question) pairs drawn from this bank."""

from __future__ import annotations

from assay.schema import Question

BOOL = [
    ("Does the text express a complaint?", None, None),
    ("Does the author ask for something to be done?", None, None),
    ("Is the text written in the first person?", None, None),
    ("Does the text mention a price, a payment or an amount of money?", None, None),
    ("Does the text contain a question?", None, None),
    ("Is the tone polite?", None, None),
    ("Does the text mention a specific date, day or time?", None, None),
    ("Does the text describe a problem that is still unresolved?", None, None),
    ("Does the text recommend something to the reader?", None, None),
    ("Does the text mention a person by name?", None, None),
    ("Is the text about a product or a service?", None, None),
    ("Does the text express gratitude?", None, None),
    ("Does the text contain sarcasm?", None, None),
    ("Would this text be appropriate to show to a child?", None, None),
    ("Does the text compare two or more things?", None, None),
    (
        "Does the author express a clear opinion?",
        "The author takes a side or states a judgement",
        "The text reports or describes without judging",
    ),
    ("Is the author angry?", None, None),
    ("Does the text contain a threat or an intent to harm?", None, None),
    (
        "Does the text contain personal data such as an address, phone number, account or card number?",
        None,
        None,
    ),
    ("Does the text mention a company or a brand?", None, None),
    ("Does the text mention a place, city or country?", None, None),
    ("Is the text mostly about the past?", None, None),
    ("Does the text describe something that has already been tried?", None, None),
    ("Does the author want a refund, replacement or compensation?", None, None),
    (
        "Is the text a reply to someone else?",
        "It reacts to or quotes a previous message",
        "It starts a new topic on its own",
    ),
    ("Does the text contain an apology?", None, None),
    ("Does the text state a deadline or a time limit?", None, None),
    ("Does the text contain a number?", None, None),
    ("Does the text describe a positive experience?", None, None),
    ("Is the text written by someone with expertise in the subject?", None, None),
    ("Does the text contain instructions or steps to follow?", None, None),
    ("Does the text mention health, illness or medicine?", None, None),
    ("Does the text mention money owed, a bill or a debt?", None, None),
    ("Does the text mention a legal matter such as a lawsuit, contract or regulation?", None, None),
    ("Is the text about technology, software or devices?", None, None),
    ("Is the text about food, restaurants or cooking?", None, None),
    ("Is the text about travel, transport or a trip?", None, None),
    ("Is the text about sports?", None, None),
    ("Is the text about politics or government?", None, None),
    ("Does the text mention a family member or a relationship?", None, None),
    ("Does the text describe an event that happened to the author?", None, None),
    ("Does the author sound uncertain or hedge their claims?", None, None),
    ("Does the text contain profanity or an insult?", None, None),
    ("Does the text state a fact that could be checked?", None, None),
    ("Does the text disagree with someone or something?", None, None),
    ("Does the text mention a quantity larger than one of something?", None, None),
    ("Does the text describe a cause and its effect?", None, None),
    ("Does the text request contact from a person, such as a call back or a meeting?", None, None),
    ("Is the situation described urgent?", None, None),
    ("Does the text mention that something was lost, broken or damaged?", None, None),
]

CHOICE = [
    (
        "What is the main purpose of the text?",
        {
            "inform": "Shares facts or news",
            "complain": "Expresses dissatisfaction",
            "request": "Asks for an action or help",
            "promote": "Advertises or persuades",
            "ask": "Asks a question to learn something",
            "entertain": "Tells a story or amuses",
        },
    ),
    (
        "Who is the most likely author?",
        {
            "customer": "A customer or user of a service",
            "representative": "Someone speaking for a company",
            "journalist": "A reporter or editor",
            "individual": "A private person writing informally",
            "expert": "A professional writing in their field",
        },
    ),
    (
        "Which emotion is most present?",
        {
            "neutral": None,
            "joy": None,
            "anger": None,
            "sadness": None,
            "fear": None,
            "surprise": None,
        },
    ),
    (
        "What kind of text is this?",
        {
            "review": "An evaluation of a product, place or work",
            "news": "Reporting on events",
            "message": "A note written to a specific recipient",
            "question": "Mainly a question seeking an answer",
            "description": "Describes something factually",
            "other": None,
        },
    ),
    (
        "If this arrived at a company's inbox, which desk should get it?",
        {
            "billing": "Charges, invoices, refunds",
            "technical": "Bugs, outages, how-to",
            "account": "Login, access, profile",
            "sales": "Pricing, upgrades, quotes",
            "feedback": "Opinions and reviews with no request",
            "not_support": "Unrelated to the company",
        },
    ),
    (
        "What is the best next action for a reader responsible for this text?",
        {
            "reply": "Write back",
            "escalate": "Hand to a specialist or manager",
            "archive": "No action needed",
            "forward": "Send to a different team",
        },
    ),
    (
        "Who is the text addressed to?",
        {
            "company": "A business or its staff",
            "public": "A general audience",
            "person": "One specific person",
            "unclear": None,
        },
    ),
    (
        "What is the author's overall stance toward the main subject?",
        {
            "favorable": None,
            "unfavorable": None,
            "mixed": "Both good and bad points",
            "none": "No stance is taken",
        },
    ),
    (
        "Which domain does the text belong to?",
        {
            "commerce": "Shopping, orders, products",
            "finance": "Banking, payments, investing",
            "health": "Medicine, wellbeing, care",
            "technology": "Software, devices, internet",
            "entertainment": "Film, music, games, books",
            "society": "Politics, news, public life",
            "personal": "Private life and relationships",
            "other": None,
        },
    ),
    (
        "What does the author want most?",
        {
            "fix": "A problem solved",
            "money": "A refund, discount or payment",
            "information": "An answer or explanation",
            "acknowledgement": "To be heard or agreed with",
            "nothing": "No want is expressed",
        },
    ),
    (
        "How should a moderator treat this text?",
        {
            "allow": "Nothing objectionable",
            "warn": "Borderline, note it",
            "remove": "Violates common rules on abuse, hate or harm",
        },
    ),
    (
        "What is the time frame the text is mostly about?",
        {
            "past": None,
            "present": None,
            "future": None,
            "timeless": "General statements without a time",
        },
    ),
    (
        "Which best describes the length and detail of the text?",
        {
            "brief": "One or two short points",
            "moderate": "A few details",
            "detailed": "Many specifics, names or steps",
        },
    ),
    (
        "What is the register of the text?",
        {
            "casual": "Slang, fragments, emoji",
            "conversational": "Plain everyday language",
            "professional": "Business or formal tone",
            "technical": "Specialised vocabulary",
        },
    ),
    (
        "What sentiment does the text convey about a product or service, if any?",
        {
            "positive": None,
            "negative": None,
            "neutral": None,
            "no_product": "No product or service is discussed",
        },
    ),
    (
        "Which of these is the text closest to?",
        {
            "story": "A narrative of events",
            "argument": "Reasons for a position",
            "instruction": "How to do something",
            "dialogue": "An exchange between people",
            "list": "Items or facts without narrative",
        },
    ),
    (
        "What is the likely relationship between the author and the reader?",
        {
            "customer_business": None,
            "colleagues": None,
            "friends_or_family": None,
            "strangers": None,
            "author_to_public": None,
        },
    ),
    (
        "If the text describes a problem, whose fault does the author think it is?",
        {
            "the_company": None,
            "the_author": None,
            "someone_else": None,
            "no_one": None,
            "no_problem": "No problem is described",
        },
    ),
    (
        "What level of expertise does the text assume from its reader?",
        {
            "none": "Anyone can follow",
            "some": "Basic familiarity with the topic",
            "expert": "Specialist knowledge",
        },
    ),
    (
        "Which response would serve the author best?",
        {
            "apology_and_fix": None,
            "explanation": None,
            "refund_or_credit": None,
            "thanks": None,
            "none_needed": None,
        },
    ),
]

SCORE = [
    (
        "How formal is the language?",
        ["Casual, slang or fragments", "Neutral everyday language", "Formal or professional"],
    ),
    ("How urgent is the text?", ["Not urgent", "Should be handled soon", "Needs immediate action"]),
    (
        "How positive is the overall tone?",
        [
            "Very negative",
            "Somewhat negative",
            "Neutral or mixed",
            "Somewhat positive",
            "Very positive",
        ],
    ),
    (
        "How confident does the author sound?",
        ["Uncertain or hedging", "Moderately confident", "Very confident"],
    ),
    (
        "How specific is the text?",
        [
            "Vague generalities",
            "Some concrete details",
            "Very specific with names, numbers or steps",
        ],
    ),
    (
        "How much does the text rely on the reader knowing prior context?",
        ["Self-contained", "Some context assumed", "Hard to follow without context"],
    ),
    (
        "How likely is the author to take further action if ignored?",
        ["Unlikely", "Possible", "Very likely"],
    ),
    ("How angry is the author?", ["Calm", "Annoyed", "Angry", "Furious"]),
    ("How clear is the text?", ["Confusing", "Mostly clear", "Very clear"]),
    ("How objective is the text?", ["Purely opinion", "Mixed", "Purely factual"]),
    (
        "How serious is the problem described, if any?",
        ["No problem", "Minor inconvenience", "Significant problem", "Severe or dangerous"],
    ),
    (
        "How satisfied is the author?",
        ["Very dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very satisfied"],
    ),
    (
        "How much effort would a good reply take?",
        ["A sentence", "A paragraph", "Research or coordination with others"],
    ),
    ("How trustworthy does the text seem?", ["Likely misleading or spam", "Uncertain", "Credible"]),
    (
        "How offensive is the text?",
        ["Not at all", "Mildly rude", "Offensive", "Hateful or threatening"],
    ),
    (
        "How complex is the language?",
        ["Simple words and short sentences", "Average", "Long sentences and rare words"],
    ),
    ("How emotional is the text?", ["Flat", "Some feeling", "Highly emotional"]),
    ("How persuasive is the text meant to be?", ["Not at all", "Somewhat", "Strongly persuasive"]),
    (
        "How well does the text explain its reasons?",
        ["No reasons given", "Some reasons", "Thorough reasoning"],
    ),
    (
        "How likely is the text to need a human rather than an automated reply?",
        ["Automated reply is fine", "Probably a human", "Certainly a human"],
    ),
]


def question_bank() -> list[Question]:
    bank: list[Question] = []
    for ins, yes, no in BOOL:
        bank.append(Question(type="bool", instructions=ins, yes=yes, no=no))
    for ins, options in CHOICE:
        bank.append(Question(type="choice", instructions=ins, options=options))
    for ins, levels in SCORE:
        bank.append(Question(type="score", instructions=ins, levels=levels))
    return bank
