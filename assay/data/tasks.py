"""Task definitions: public datasets rendered as typed questions with described options.

Sources used by the public kev transfer suite (mmlu, dair-ai emotion, sciq, tweet_eval
offensive, qnli, paws) are deliberately absent so that suite stays a fair transfer test.
Tasks marked holdout=True are never trained on; they measure calibration on unseen tasks.
"""

from __future__ import annotations

import ast
import collections
from typing import Any

from assay.data.policy import generate as generate_policy
from assay.data.registry import (
    Builder,
    Example,
    TaskSpec,
    bool_builder,
    choice_builder,
    clip_text,
    pick,
    score_builder,
)
from assay.schema import Question

PARQUET = "refs/convert/parquet"


def names_options(ds, column: str, describe: dict[str, str] | None = None) -> dict[str, str | None]:
    names = ds.features[column].names
    return {n: (describe or {}).get(n) for n in names}


def label_name(ds, column: str):
    names = ds.features[column].names
    return lambda row: names[row[column]] if row[column] is not None and row[column] >= 0 else None


def letters(n: int) -> list[str]:
    return [chr(ord("a") + i) for i in range(n)]


# ----------------------------------------------------------------------------------------
# Topic classification
# ----------------------------------------------------------------------------------------

AG_NEWS = TaskSpec(
    name="ag_news",
    hf_id="fancyzhx/ag_news",
    config=None,
    split="train",
    eval_split="test",
    build=choice_builder(
        [
            "Which section of a news site does this article belong to?",
            "What is the topic of this news article?",
            "Classify the article into one news category.",
        ],
        {
            "world": "International news, politics, conflicts and diplomacy",
            "sports": "Sports events, athletes, teams and results",
            "business": "Companies, markets, economy and finance",
            "science_technology": "Science, technology, software and research",
        },
        lambda r: clip_text(r["text"], 1200),
        lambda r: ["world", "sports", "business", "science_technology"][r["label"]],
    ),
)

DBPEDIA_NAMES = [
    "company",
    "educational_institution",
    "artist",
    "athlete",
    "office_holder",
    "mean_of_transportation",
    "building",
    "natural_place",
    "village",
    "animal",
    "plant",
    "album",
    "film",
    "written_work",
]
DBPEDIA = TaskSpec(
    name="dbpedia",
    hf_id="fancyzhx/dbpedia_14",
    config=None,
    split="train",
    eval_split="test",
    build=choice_builder(
        [
            "What kind of entity does this encyclopedia article describe?",
            "Which category best describes the subject of the text?",
        ],
        {
            "company": "A business or corporation",
            "educational_institution": "A school, college or university",
            "artist": "A musician, painter, writer or other artist",
            "athlete": "A sportsperson",
            "office_holder": "A politician or official holding public office",
            "mean_of_transportation": "A vehicle, ship, aircraft or train",
            "building": "A building or structure",
            "natural_place": "A river, mountain, lake or other natural feature",
            "village": "A village or small settlement",
            "animal": "An animal species",
            "plant": "A plant species",
            "album": "A music album",
            "film": "A movie",
            "written_work": "A book, novel, journal or other written work",
        },
        lambda r: {"title": r["title"], "text": clip_text(r["content"], 1200)},
        lambda r: DBPEDIA_NAMES[r["label"]],
    ),
)

BBC_NEWS = TaskSpec(
    name="bbc_news",
    hf_id="SetFit/bbc-news",
    config=None,
    split="train",
    eval_split="test",
    holdout=True,
    build=choice_builder(
        [
            "Which BBC News section does this article belong to?",
            "What is the topic of the article?",
        ],
        {
            "business": "Economy, companies and markets",
            "entertainment": "Film, music, television and celebrities",
            "politics": "Government, parliament, elections and policy",
            "sport": "Sports news and results",
            "tech": "Technology, gadgets, internet and computing",
        },
        lambda r: clip_text(r["text"], 1500),
        lambda r: r["label_text"],
    ),
)

TREC = TaskSpec(
    name="trec",
    hf_id="CogComp/trec",
    config=None,
    split="train",
    eval_split="test",
    revision=PARQUET,
    build=choice_builder(
        [
            "What type of answer is this question asking for?",
            "Classify the question by the kind of thing it asks about.",
        ],
        {
            "abbreviation": "Asks for an abbreviation or what an abbreviation stands for",
            "entity": "Asks for a thing: an animal, colour, product, event, substance, and so on",
            "description": "Asks for a definition, explanation, reason or description",
            "human": "Asks for a person or a group of people",
            "location": "Asks for a place: city, country, mountain, and so on",
            "number": "Asks for a numeric value: a count, date, distance, money, and so on",
        },
        lambda r: r["text"],
        lambda r: ["abbreviation", "entity", "description", "human", "location", "number"][
            r["coarse_label"]
        ],
    ),
)


def _ledgar(ds) -> Builder:
    options = names_options(ds, "label")
    return choice_builder(
        [
            "Which type of contract provision is this clause?",
            "Under which heading would this contract clause appear?",
        ],
        options,
        lambda r: clip_text(r["text"], 1500),
        label_name(ds, "label"),
    )


LEDGAR = TaskSpec(
    name="ledgar",
    hf_id="coastalcph/lex_glue",
    config="ledgar",
    split="train",
    eval_split="test",
    build=None,
    build_factory=_ledgar,
    max_train=1500,
)

SCOTUS_DESC = {
    "Criminal Procedure": "Rights of the accused, criminal trials, sentencing",
    "Civil Rights": "Discrimination, voting rights, equal protection",
    "First Amendment": "Speech, press, religion, assembly",
    "Due Process": "Procedural or substantive due process",
    "Privacy": "Privacy, abortion, contraception",
    "Attorneys": "Attorneys' fees, bar admission, legal practice",
    "Unions": "Labour unions and collective bargaining",
    "Economic Activity": "Business regulation, antitrust, contracts, bankruptcy",
    "Judicial Power": "Jurisdiction, standing, judicial review",
    "Federalism": "Relations between federal and state governments",
    "Interstate Relations": "Disputes between states",
    "Federal Taxation": "Federal tax law",
    "Miscellaneous": "Other issues",
}


def _scotus(ds) -> Builder:
    options = names_options(ds, "label", SCOTUS_DESC)
    return choice_builder(
        ["Which issue area does this Supreme Court opinion concern?"],
        options,
        lambda r: clip_text(r["text"], 2500),
        label_name(ds, "label"),
    )


SCOTUS = TaskSpec(
    name="scotus",
    hf_id="coastalcph/lex_glue",
    config="scotus",
    split="train",
    eval_split="test",
    build=None,
    build_factory=_scotus,
    max_train=800,
)

# ----------------------------------------------------------------------------------------
# Intent classification (many options)
# ----------------------------------------------------------------------------------------


def _banking77(ds) -> Builder:
    options = names_options(ds, "label")
    return choice_builder(
        [
            "What is the customer's intent in this banking support message?",
            "Which intent label fits this message to a bank's support chat?",
        ],
        options,
        lambda r: r["text"],
        label_name(ds, "label"),
    )


BANKING77 = TaskSpec(
    name="banking77",
    hf_id="legacy-datasets/banking77",
    config=None,
    split="train",
    eval_split="test",
    build=None,
    build_factory=_banking77,
    max_train=1500,
)


def _clinc(ds) -> Builder:
    options = names_options(ds, "intent")
    options["oos"] = "Out of scope: none of the other intents apply"
    return choice_builder(
        [
            "Which intent does the user's request express?",
            "What does the user want the assistant to do?",
        ],
        options,
        lambda r: r["text"],
        label_name(ds, "intent"),
    )


CLINC = TaskSpec(
    name="clinc150",
    hf_id="clinc/clinc_oos",
    config="plus",
    split="train",
    eval_split="test",
    build=None,
    build_factory=_clinc,
    max_train=1500,
)


# ----------------------------------------------------------------------------------------
# Sentiment: binary, fine-grained (ordinal), soft-labelled
# ----------------------------------------------------------------------------------------

SENTIMENT_YES = "The text expresses a positive opinion or feeling"
SENTIMENT_NO = "The text expresses a negative opinion or feeling"

IMDB = TaskSpec(
    name="imdb",
    hf_id="stanfordnlp/imdb",
    config=None,
    split="train",
    eval_split="test",
    build=bool_builder(
        ["Is this movie review positive?", "Does the reviewer like the movie?"],
        lambda r: clip_text(r["text"], 1800),
        lambda r: r["label"] == 1,
        yes=SENTIMENT_YES,
        no=SENTIMENT_NO,
    ),
)

ROTTEN = TaskSpec(
    name="rotten_tomatoes",
    hf_id="cornell-movie-review-data/rotten_tomatoes",
    config=None,
    split="train",
    eval_split="test",
    build=bool_builder(
        ["Is this review snippet positive?", "Is the critic's sentence favourable to the film?"],
        lambda r: r["text"],
        lambda r: r["label"] == 1,
    ),
    max_train=800,
)

SST2 = TaskSpec(
    name="sst2",
    hf_id="stanfordnlp/sst2",
    config=None,
    split="train",
    eval_split="validation",
    build=bool_builder(
        ["Is the sentiment of this sentence positive?"],
        lambda r: r["sentence"],
        lambda r: r["label"] == 1,
    ),
    max_train=800,
)

AMAZON_POLARITY = TaskSpec(
    name="amazon_polarity",
    hf_id="fancyzhx/amazon_polarity",
    config=None,
    split="train",
    eval_split="test",
    build=bool_builder(
        ["Is this product review positive?", "Would the reviewer recommend the product?"],
        lambda r: {"title": r["title"], "review": clip_text(r["content"], 1200)},
        lambda r: r["label"] == 1,
    ),
    max_train=800,
)

STAR_LEVELS = [
    "1 star: very negative, the customer is dissatisfied",
    "2 stars: negative, more problems than praise",
    "3 stars: mixed or neutral",
    "4 stars: positive with minor reservations",
    "5 stars: very positive, fully satisfied",
]

YELP = TaskSpec(
    name="yelp",
    hf_id="Yelp/yelp_review_full",
    config=None,
    split="train",
    eval_split="test",
    build=score_builder(
        ["How many stars did the reviewer most likely give?", "Rate the sentiment of this review."],
        STAR_LEVELS,
        lambda r: clip_text(r["text"], 1500),
        lambda r: r["label"],
    ),
    max_train=1200,
)

AMAZON_STARS = TaskSpec(
    name="amazon_stars",
    hf_id="SetFit/amazon_reviews_multi_en",
    config=None,
    split="train",
    eval_split="test",
    revision=PARQUET,
    build=score_builder(
        ["How many stars did the reviewer most likely give?"],
        STAR_LEVELS,
        lambda r: clip_text(r["text"], 1200),
        lambda r: int(r["label"]),
    ),
    max_train=1000,
)

APP_REVIEWS = TaskSpec(
    name="app_reviews",
    hf_id="sealuzh/app_reviews",
    config=None,
    split="train",
    holdout=True,
    build=score_builder(
        ["How many stars did this app review most likely give?"],
        STAR_LEVELS,
        lambda r: {"app": r["package_name"], "review": clip_text(r["review"], 1000)},
        lambda r: int(r["star"]) - 1,
    ),
)

SST5 = TaskSpec(
    name="sst5",
    hf_id="SetFit/sst5",
    config=None,
    split="train",
    eval_split="test",
    revision=PARQUET,
    build=score_builder(
        ["How positive is this movie review sentence?"],
        [
            "Very negative",
            "Somewhat negative",
            "Neutral or mixed",
            "Somewhat positive",
            "Very positive",
        ],
        lambda r: r["text"],
        lambda r: int(r["label"]),
    ),
    max_train=1000,
)


def _dynasent_counts(row: dict[str, Any]) -> list[float] | None:
    dist = row.get("label_distribution")
    if isinstance(dist, str):
        dist = ast.literal_eval(dist)
    if not dist:
        return None
    return [
        float(len(dist.get("negative", []))),
        float(len(dist.get("neutral", []))),
        float(len(dist.get("positive", []))),
    ]


DYNASENT = TaskSpec(
    name="dynasent",
    hf_id="dynabench/dynasent",
    config=None,
    data_dir="dynabench.dynasent.r1.all",
    split="train",
    eval_split="validation",
    revision=PARQUET,
    build=score_builder(
        [
            "What is the sentiment of this sentence?",
            "Is this sentence negative, neutral or positive?",
        ],
        ["Negative", "Neutral or mixed", "Positive"],
        lambda r: r["sentence"],
        lambda r: {"negative": 0, "neutral": 1, "positive": 2}.get(r["gold_label"]),
        counts_fn=_dynasent_counts,
    ),
    max_train=1000,
)


# ----------------------------------------------------------------------------------------
# Natural language inference, fact verification, paraphrase
# ----------------------------------------------------------------------------------------

NLI_OPTIONS = {
    "entailment": "The hypothesis must be true if the premise is true",
    "neutral": "The hypothesis might be true; the premise does not settle it",
    "contradiction": "The hypothesis cannot be true if the premise is true",
}
NLI_INSTRUCTIONS = [
    "Given the premise, what is the status of the hypothesis?",
    "Does the premise entail, contradict or say nothing about the hypothesis?",
]


def _nli_label(row: dict[str, Any]) -> str | None:
    label = row["label"]
    if label is None or label < 0:
        return None
    return ["entailment", "neutral", "contradiction"][label]


MNLI = TaskSpec(
    name="mnli",
    hf_id="nyu-mll/multi_nli",
    config=None,
    split="train",
    eval_split="validation_matched",
    build=choice_builder(
        NLI_INSTRUCTIONS,
        NLI_OPTIONS,
        lambda r: {"premise": r["premise"], "hypothesis": r["hypothesis"]},
        _nli_label,
    ),
    max_train=1200,
)

SNLI = TaskSpec(
    name="snli",
    hf_id="stanfordnlp/snli",
    config=None,
    split="train",
    eval_split="validation",
    build=choice_builder(
        NLI_INSTRUCTIONS,
        NLI_OPTIONS,
        lambda r: {"premise": r["premise"], "hypothesis": r["hypothesis"]},
        _nli_label,
    ),
    max_train=800,
)

ANLI = TaskSpec(
    name="anli",
    hf_id="facebook/anli",
    config=None,
    split="train_r3",
    eval_split="dev_r3",
    build=choice_builder(
        NLI_INSTRUCTIONS,
        NLI_OPTIONS,
        lambda r: {"premise": clip_text(r["premise"], 1500), "hypothesis": r["hypothesis"]},
        _nli_label,
    ),
    max_train=800,
)

CB = TaskSpec(
    name="cb",
    hf_id="aps/super_glue",
    config="cb",
    split="train",
    eval_split="validation",
    build=choice_builder(
        NLI_INSTRUCTIONS,
        NLI_OPTIONS,
        lambda r: {"premise": r["premise"], "hypothesis": r["hypothesis"]},
        lambda r: (
            ["entailment", "contradiction", "neutral"][r["label"]] if r["label"] >= 0 else None
        ),
    ),
    max_train=250,
    max_eval=56,
)

RTE = TaskSpec(
    name="rte",
    hf_id="nyu-mll/glue",
    config="rte",
    split="train",
    eval_split="validation",
    build=bool_builder(
        [
            "Does the text entail the hypothesis?",
            "If the text is true, must the hypothesis be true?",
        ],
        lambda r: {"text": r["sentence1"], "hypothesis": r["sentence2"]},
        lambda r: r["label"] == 0 if r["label"] >= 0 else None,
        yes="The hypothesis follows from the text",
        no="The hypothesis does not follow from the text",
    ),
    max_train=800,
)

SCITAIL = TaskSpec(
    name="scitail",
    hf_id="allenai/scitail",
    config="snli_format",
    split="train",
    eval_split="test",
    holdout=True,
    build=bool_builder(
        ["Does the premise entail the hypothesis?"],
        lambda r: {"premise": r["sentence1"], "hypothesis": r["sentence2"]},
        lambda r: r["gold_label"] == "entailment",
    ),
)


def _vitaminc_label(row: dict[str, Any]) -> str | None:
    return {
        "SUPPORTS": "supported",
        "REFUTES": "refuted",
        "NOT ENOUGH INFO": "not_enough_information",
    }.get(str(row["label"]).strip().upper())


VITAMINC = TaskSpec(
    name="vitaminc",
    hf_id="tals/vitaminc",
    config=None,
    split="train",
    eval_split="test",
    build=choice_builder(
        [
            "Does the evidence support or refute the claim?",
            "Given only the evidence, what is the status of the claim?",
        ],
        {
            "supported": "The evidence shows the claim is true",
            "refuted": "The evidence shows the claim is false",
            "not_enough_information": "The evidence does not settle the claim",
        },
        lambda r: {"claim": r["claim"], "evidence": clip_text(r["evidence"], 1500)},
        _vitaminc_label,
    ),
    max_train=1200,
)

QQP = TaskSpec(
    name="qqp",
    hf_id="nyu-mll/glue",
    config="qqp",
    split="train",
    eval_split="validation",
    build=bool_builder(
        [
            "Are these two questions asking the same thing?",
            "Would one answer satisfy both questions?",
        ],
        lambda r: {"question_1": r["question1"], "question_2": r["question2"]},
        lambda r: r["label"] == 1 if r["label"] >= 0 else None,
        yes="Duplicates: the same information need",
        no="Different questions, even if they share words",
    ),
    max_train=1000,
)

MRPC = TaskSpec(
    name="mrpc",
    hf_id="nyu-mll/glue",
    config="mrpc",
    split="train",
    eval_split="validation",
    build=bool_builder(
        [
            "Do these two sentences mean the same thing?",
            "Is the second sentence a paraphrase of the first?",
        ],
        lambda r: {"sentence_1": r["sentence1"], "sentence_2": r["sentence2"]},
        lambda r: r["label"] == 1 if r["label"] >= 0 else None,
    ),
    max_train=800,
)

MEDICAL_PAIRS = TaskSpec(
    name="medical_questions_pairs",
    hf_id="curaihealth/medical_questions_pairs",
    config=None,
    split="train",
    holdout=True,
    build=bool_builder(
        ["Are these two medical questions asking the same thing?"],
        lambda r: {"question_1": r["question_1"], "question_2": r["question_2"]},
        lambda r: r["label"] == 1,
    ),
)

STSB = TaskSpec(
    name="stsb",
    hf_id="nyu-mll/glue",
    config="stsb",
    split="train",
    eval_split="validation",
    build=score_builder(
        ["How similar in meaning are the two sentences?"],
        [
            "Completely dissimilar, different topics",
            "Same topic but not equivalent",
            "Share some details but differ in important ways",
            "Roughly equivalent, some details differ",
            "Mostly equivalent, minor details differ",
            "Completely equivalent",
        ],
        lambda r: {"sentence_1": r["sentence1"], "sentence_2": r["sentence2"]},
        lambda r: round(float(r["label"])),
        value_fn=lambda r: float(r["label"]),
    ),
    max_train=1000,
)

COLA = TaskSpec(
    name="cola",
    hf_id="nyu-mll/glue",
    config="cola",
    split="train",
    eval_split="validation",
    build=bool_builder(
        ["Is this sentence grammatically acceptable English?"],
        lambda r: r["sentence"],
        lambda r: r["label"] == 1 if r["label"] >= 0 else None,
    ),
    max_train=800,
)

SUBJ = TaskSpec(
    name="subjectivity",
    hf_id="SetFit/subj",
    config=None,
    split="train",
    eval_split="test",
    build=choice_builder(
        ["Is this sentence subjective or objective?"],
        {
            "objective": "States facts or plot without personal opinion",
            "subjective": "Expresses an opinion, evaluation or feeling",
        },
        lambda r: r["text"],
        lambda r: r["label_text"],
    ),
    max_train=600,
)

WIC = TaskSpec(
    name="wic",
    hf_id="aps/super_glue",
    config="wic",
    split="train",
    eval_split="validation",
    build=bool_builder(
        ["Is the word used with the same meaning in both sentences?"],
        lambda r: {"word": r["word"], "sentence_1": r["sentence1"], "sentence_2": r["sentence2"]},
        lambda r: r["label"] == 1 if r["label"] >= 0 else None,
    ),
    max_train=800,
)

# ----------------------------------------------------------------------------------------
# Toxicity, hate, spam (several with human agreement fractions as soft targets)
# ----------------------------------------------------------------------------------------

CIVIL_TOXIC = TaskSpec(
    name="civil_comments_toxic",
    hf_id="google/civil_comments",
    config=None,
    split="train",
    eval_split="test",
    build=bool_builder(
        ["Is this comment toxic?", "Would a reasonable moderator flag this comment as toxic?"],
        lambda r: clip_text(r["text"], 1000),
        lambda r: float(r["toxicity"]) >= 0.5,
        yes="Rude, disrespectful, hateful or likely to make someone leave the discussion",
        no="Civil, even if critical or negative",
        p_yes_fn=lambda r: float(r["toxicity"]),
    ),
    max_train=1500,
)

CIVIL_SEVERITY = TaskSpec(
    name="civil_comments_severity",
    hf_id="google/civil_comments",
    config=None,
    split="train",
    eval_split="test",
    build=score_builder(
        ["How toxic is this comment?"],
        [
            "Not toxic",
            "Mildly rude or dismissive",
            "Clearly toxic: insulting or hostile",
            "Severely toxic: hateful, threatening or abusive",
        ],
        lambda r: clip_text(r["text"], 1000),
        lambda r: min(3, int(float(r["toxicity"]) * 4)),
        value_fn=lambda r: float(r["toxicity"]) * 3.0,
    ),
    max_train=800,
)


def _hate_speech_prepare(ds) -> list[dict[str, Any]]:
    groups: dict[Any, dict[str, Any]] = {}
    for row in ds:
        g = groups.setdefault(row["comment_id"], {"text": row["text"], "counts": [0.0, 0.0, 0.0]})
        g["counts"][int(row["hatespeech"])] += 1.0
    return [g for g in groups.values() if sum(g["counts"]) >= 3]


HATE_SPEECH = TaskSpec(
    name="measuring_hate_speech",
    hf_id="ucberkeley-dlab/measuring-hate-speech",
    config=None,
    split="train",
    prepare=_hate_speech_prepare,
    build=score_builder(
        ["Does this comment contain hate speech?", "Rate the comment for hate speech."],
        [
            "Supportive or neutral, no hate speech",
            "Ambiguous or borderline",
            "Hate speech targeting a group",
        ],
        lambda r: clip_text(r["text"], 1000),
        lambda r: max(range(3), key=lambda i: r["counts"][i]),
        counts_fn=lambda r: r["counts"],
    ),
    max_train=1200,
)

TWEET_HATE = TaskSpec(
    name="tweet_hate",
    hf_id="cardiffnlp/tweet_eval",
    config="hate",
    split="train",
    eval_split="test",
    build=bool_builder(
        ["Is this tweet hateful towards immigrants or women?"],
        lambda r: r["text"],
        lambda r: r["label"] == 1,
    ),
    max_train=800,
)

TWEET_IRONY = TaskSpec(
    name="tweet_irony",
    hf_id="cardiffnlp/tweet_eval",
    config="irony",
    split="train",
    eval_split="test",
    holdout=True,
    build=bool_builder(
        ["Is this tweet ironic?"],
        lambda r: r["text"],
        lambda r: r["label"] == 1,
        yes="Says one thing to mean another, or highlights a contrast for effect",
        no="Meant literally",
    ),
)

ETHOS = TaskSpec(
    name="ethos",
    hf_id="iamollas/ethos",
    config=None,
    data_dir="binary",
    split="train",
    revision=PARQUET,
    holdout=True,
    build=bool_builder(
        ["Does this comment contain hate speech?"],
        lambda r: r["text"],
        lambda r: int(r["label"]) == 1,
    ),
)

SMS_SPAM = TaskSpec(
    name="sms_spam",
    hf_id="ucirvine/sms_spam",
    config=None,
    split="train",
    build=bool_builder(
        ["Is this SMS message spam?", "Is this an unsolicited promotional or scam message?"],
        lambda r: r["sms"],
        lambda r: r["label"] == 1,
    ),
    max_train=600,
)

ENRON_SPAM = TaskSpec(
    name="enron_spam",
    hf_id="SetFit/enron_spam",
    config=None,
    split="train",
    eval_split="test",
    revision=PARQUET,
    build=bool_builder(
        ["Is this email spam?"],
        lambda r: {"subject": r["subject"], "body": clip_text(r["text"], 1200)},
        lambda r: r["label"] == 1,
    ),
    max_train=600,
)

# ----------------------------------------------------------------------------------------
# Emotion (soft labels from multiple raters) and stance
# ----------------------------------------------------------------------------------------

GO_EMOTIONS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]


def _go_emotions_prepare(ds) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in ds:
        if row.get("example_very_unclear"):
            continue
        g = groups.setdefault(
            row["id"], {"text": row["text"], "raters": 0, "counts": collections.Counter()}
        )
        g["raters"] += 1
        for e in GO_EMOTIONS:
            if row.get(e):
                g["counts"][e] += 1
    out = []
    for g in groups.values():
        if g["raters"] >= 3 and g["counts"]:
            out.append(g)
    return out


def _go_emotions_target(row: dict[str, Any]) -> dict[str, float]:
    total = float(sum(row["counts"].values()))
    return {e: row["counts"].get(e, 0) / total for e in GO_EMOTIONS}


GOEMOTIONS = TaskSpec(
    name="go_emotions",
    hf_id="google-research-datasets/go_emotions",
    config="raw",
    split="train",
    prepare=_go_emotions_prepare,
    build=choice_builder(
        ["Which emotion does the writer of this Reddit comment express most?"],
        {e: None for e in GO_EMOTIONS},
        lambda r: r["text"],
        lambda r: max(GO_EMOTIONS, key=lambda e: (r["counts"].get(e, 0), e != "neutral")),
        target_fn=_go_emotions_target,
    ),
    max_train=1500,
)


def _stance(config: str, target: str, holdout: bool = False) -> TaskSpec:
    return TaskSpec(
        name=f"stance_{config.split('_')[1]}",
        hf_id="cardiffnlp/tweet_eval",
        config=config,
        split="train",
        eval_split="test",
        holdout=holdout,
        build=choice_builder(
            [f"What is the author's stance towards {target}?"],
            {
                "none": f"No stance on {target} can be inferred",
                "against": f"Opposes {target}",
                "favor": f"Supports {target}",
            },
            lambda r: r["text"],
            lambda r: ["none", "against", "favor"][r["label"]],
        ),
        max_train=400,
        max_eval=120,
    )


STANCE_ABORTION = _stance("stance_abortion", "legal abortion")
STANCE_ATHEISM = _stance("stance_atheism", "atheism")
STANCE_FEMINIST = _stance("stance_feminist", "the feminist movement")
STANCE_HILLARY = _stance("stance_hillary", "Hillary Clinton")
STANCE_CLIMATE = _stance(
    "stance_climate", "the view that climate change is a real concern", holdout=True
)

TWEET_SENTIMENT = TaskSpec(
    name="tweet_sentiment",
    hf_id="cardiffnlp/tweet_eval",
    config="sentiment",
    split="train",
    eval_split="test",
    build=score_builder(
        ["What is the sentiment of this tweet?"],
        ["Negative", "Neutral", "Positive"],
        lambda r: r["text"],
        lambda r: r["label"],
    ),
    max_train=800,
)

# ----------------------------------------------------------------------------------------
# Reading comprehension and commonsense (some with unanswerable swaps for the evidence head)
# ----------------------------------------------------------------------------------------

BOOLQ = TaskSpec(
    name="boolq",
    hf_id="google/boolq",
    config=None,
    split="train",
    eval_split="validation",
    swap_state_fraction=0.25,
    swap_state_key="passage",
    build=bool_builder(
        [
            "Based on the passage, is the answer to the question yes?",
            "Answer the question using the passage.",
        ],
        lambda r: {"passage": clip_text(r["passage"], 1800), "question": r["question"]},
        lambda r: bool(r["answer"]),
    ),
    max_train=1200,
)

MULTIRC = TaskSpec(
    name="multirc",
    hf_id="aps/super_glue",
    config="multirc",
    split="train",
    eval_split="validation",
    swap_state_fraction=0.25,
    swap_state_key="paragraph",
    build=bool_builder(
        ["According to the paragraph, is the candidate answer to the question correct?"],
        lambda r: {
            "paragraph": clip_text(r["paragraph"], 1800),
            "question": r["question"],
            "candidate_answer": r["answer"],
        },
        lambda r: r["label"] == 1 if r["label"] >= 0 else None,
    ),
    max_train=1000,
)


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return list(ast.literal_eval(value))
    return list(value)


def _list_options(texts: list[str]) -> dict[str, str | None]:
    return {k: clip_text(t, 400) for k, t in zip(letters(len(texts)), texts)}


RACE = TaskSpec(
    name="race",
    hf_id="ehovy/race",
    config="all",
    split="train",
    eval_split="test",
    swap_state_fraction=0.25,
    swap_state_key="article",
    build=choice_builder(
        ["Which option correctly answers the question about the article?"],
        lambda r: _list_options(r["options"]),
        lambda r: {"article": clip_text(r["article"], 2200), "question": r["question"]},
        lambda r: r["answer"].strip().lower(),
    ),
    max_train=1000,
)

COSMOS = TaskSpec(
    name="cosmos_qa",
    hf_id="allenai/cosmos_qa",
    config=None,
    split="train",
    eval_split="validation",
    revision=PARQUET,
    swap_state_fraction=0.2,
    swap_state_key="context",
    build=choice_builder(
        ["Which option best answers the question about the context?"],
        lambda r: _list_options([r["answer0"], r["answer1"], r["answer2"], r["answer3"]]),
        lambda r: {"context": clip_text(r["context"], 1500), "question": r["question"]},
        lambda r: letters(4)[r["label"]],
    ),
    max_train=1000,
)

DREAM = TaskSpec(
    name="dream",
    hf_id="dataset-org/dream",
    config=None,
    split="train",
    eval_split="test",
    revision=PARQUET,
    holdout=True,
    build=choice_builder(
        ["Which option correctly answers the question about the dialogue?"],
        lambda r: _list_options(r["choice"]),
        lambda r: {"dialogue": clip_text(" ".join(r["dialogue"]), 1800), "question": r["question"]},
        lambda r: (
            letters(len(r["choice"]))[r["choice"].index(r["answer"])]
            if r["answer"] in r["choice"]
            else None
        ),
    ),
)

HELLASWAG = TaskSpec(
    name="hellaswag",
    hf_id="Rowan/hellaswag",
    config=None,
    split="train",
    eval_split="validation",
    build=choice_builder(
        ["Which ending is the most plausible continuation of the text?"],
        lambda r: _list_options(r["endings"]),
        lambda r: r["ctx"],
        lambda r: letters(4)[int(r["label"])] if str(r["label"]).strip() != "" else None,
    ),
    max_train=800,
)

SWAG = TaskSpec(
    name="swag",
    hf_id="allenai/swag",
    config="regular",
    split="train",
    eval_split="validation",
    build=choice_builder(
        ["Which ending most plausibly continues the situation?"],
        lambda r: _list_options([r["ending0"], r["ending1"], r["ending2"], r["ending3"]]),
        lambda r: f"{r['sent1']} {r['sent2']}",
        lambda r: (
            letters(4)[int(r["label"])] if r["label"] is not None and r["label"] >= 0 else None
        ),
    ),
    max_train=600,
)

PIQA = TaskSpec(
    name="piqa",
    hf_id="ybisk/piqa",
    config=None,
    split="train",
    eval_split="validation",
    revision=PARQUET,
    build=choice_builder(
        ["Which solution achieves the goal?"],
        lambda r: _list_options([r["sol1"], r["sol2"]]),
        lambda r: {"goal": r["goal"]},
        lambda r: (
            letters(2)[int(r["label"])] if r["label"] is not None and r["label"] >= 0 else None
        ),
    ),
    max_train=600,
)

SIQA = TaskSpec(
    name="social_iqa",
    hf_id="allenai/social_i_qa",
    config=None,
    split="train",
    eval_split="validation",
    revision=PARQUET,
    build=choice_builder(
        ["Which option best answers the question about the situation?"],
        lambda r: _list_options([r["answerA"], r["answerB"], r["answerC"]]),
        lambda r: {"context": r["context"], "question": r["question"]},
        lambda r: letters(3)[int(r["label"]) - 1],
    ),
    max_train=800,
)

WINOGRANDE = TaskSpec(
    name="winogrande",
    hf_id="allenai/winogrande",
    config="winogrande_xl",
    split="train",
    eval_split="validation",
    build=choice_builder(
        ["Which option fills in the blank correctly?"],
        lambda r: _list_options([r["option1"], r["option2"]]),
        lambda r: r["sentence"],
        lambda r: (
            letters(2)[int(r["answer"]) - 1] if str(r["answer"]).strip() in ("1", "2") else None
        ),
    ),
    max_train=800,
)

COPA = TaskSpec(
    name="copa",
    hf_id="aps/super_glue",
    config="copa",
    split="train",
    eval_split="validation",
    holdout=True,
    build=choice_builder(
        ["Which alternative is the more plausible cause or effect of the premise, as asked?"],
        lambda r: _list_options([r["choice1"], r["choice2"]]),
        lambda r: {"premise": r["premise"], "asking_for": r["question"]},
        lambda r: letters(2)[r["label"]] if r["label"] >= 0 else None,
    ),
    max_eval=100,
)

COMMONSENSE_QA = TaskSpec(
    name="commonsense_qa",
    hf_id="tau/commonsense_qa",
    config=None,
    split="train",
    eval_split="validation",
    build=choice_builder(
        ["Which option answers the question?"],
        lambda r: {k.lower(): t for k, t in zip(r["choices"]["label"], r["choices"]["text"])},
        lambda r: r["question"],
        lambda r: r["answerKey"].strip().lower() or None,
    ),
    max_train=800,
)


def _arc(config: str, name: str) -> TaskSpec:
    return TaskSpec(
        name=name,
        hf_id="allenai/ai2_arc",
        config=config,
        split="train",
        eval_split="test",
        build=choice_builder(
            ["Which option correctly answers the science question?"],
            lambda r: {k.lower(): t for k, t in zip(r["choices"]["label"], r["choices"]["text"])},
            lambda r: r["question"],
            lambda r: r["answerKey"].strip().lower() or None,
        ),
        max_train=800,
    )


ARC_CHALLENGE = _arc("ARC-Challenge", "arc_challenge")
ARC_EASY = _arc("ARC-Easy", "arc_easy")

OPENBOOKQA = TaskSpec(
    name="openbookqa",
    hf_id="allenai/openbookqa",
    config="main",
    split="train",
    eval_split="test",
    build=choice_builder(
        ["Which option correctly completes or answers the science question?"],
        lambda r: {k.lower(): t for k, t in zip(r["choices"]["label"], r["choices"]["text"])},
        lambda r: r["question_stem"],
        lambda r: r["answerKey"].strip().lower() or None,
    ),
    max_train=800,
)

MEDQA = TaskSpec(
    name="medqa",
    hf_id="GBaker/MedQA-USMLE-4-options",
    config=None,
    split="train",
    eval_split="test",
    build=choice_builder(
        ["Which option is the correct answer to this medical exam question?"],
        lambda r: {k.lower(): t for k, t in r["options"].items()},
        lambda r: clip_text(r["question"], 2000),
        lambda r: r["answer_idx"].strip().lower(),
    ),
    max_train=600,
)

TRUTHFUL_QA = TaskSpec(
    name="truthful_qa",
    hf_id="EleutherAI/truthful_qa_mc",
    config=None,
    data_dir="multiple_choice",
    split="validation",
    revision=PARQUET,
    holdout=True,
    build=choice_builder(
        ["Which option is the true answer to the question?"],
        lambda r: _list_options(_as_list(r["choices"])),
        lambda r: r["question"],
        lambda r: letters(len(_as_list(r["choices"])))[int(r["label"])],
    ),
    max_eval=200,
)

# ----------------------------------------------------------------------------------------
# Preference judging (soft targets from vote ratios)
# ----------------------------------------------------------------------------------------


def _shp_target(row: dict[str, Any]) -> dict[str, float] | None:
    a, b = float(row["score_A"]), float(row["score_B"])
    if a <= 0 or b <= 0:
        return None
    return {"a": a / (a + b), "b": b / (a + b)}


SHP = TaskSpec(
    name="shp",
    hf_id="stanfordnlp/SHP",
    config=None,
    split="train",
    eval_split="validation",
    build=choice_builder(
        [
            "Which reply would the community find more helpful?",
            "Which response is the better answer to the post?",
        ],
        lambda r: {"a": clip_text(r["human_ref_A"], 700), "b": clip_text(r["human_ref_B"], 700)},
        lambda r: {"post": clip_text(r["history"], 1500)},
        lambda r: "a" if int(r["labels"]) == 1 else "b",
        target_fn=_shp_target,
    ),
    max_train=800,
)


def _hh_split(text: str) -> tuple[str, str] | None:
    marker = "\n\nAssistant:"
    idx = text.rfind(marker)
    if idx < 0:
        return None
    return text[:idx].strip(), text[idx + len(marker) :].strip()


def _hh_build(row: dict[str, Any], rng) -> Example | None:
    chosen = _hh_split(row["chosen"])
    rejected = _hh_split(row["rejected"])
    if chosen is None or rejected is None or chosen[0] != rejected[0]:
        return None
    if rng.random() < 0.5:
        options = {"a": clip_text(chosen[1], 600), "b": clip_text(rejected[1], 600)}
        label = "a"
    else:
        options = {"a": clip_text(rejected[1], 600), "b": clip_text(chosen[1], 600)}
        label = "b"
    q = Question(
        type="choice",
        instructions=pick(rng, ["Which final assistant reply is more helpful and harmless?"]),
        options=options,
    )
    return Example(state={"conversation": clip_text(chosen[0], 1500)}, question=q, label=label)


HH_RLHF = TaskSpec(
    name="hh_rlhf",
    hf_id="Anthropic/hh-rlhf",
    config=None,
    split="train",
    eval_split="test",
    holdout=True,
    build=_hh_build,
    max_eval=200,
)

# ----------------------------------------------------------------------------------------
# Truthfulness rating (ordinal)
# ----------------------------------------------------------------------------------------

LIAR = TaskSpec(
    name="liar",
    hf_id="chengxuphd/liar2",
    config=None,
    split="train",
    eval_split="test",
    build=score_builder(
        ["How truthful is this political statement, according to fact checkers?"],
        [
            "Pants on fire: absurdly false",
            "False",
            "Barely true",
            "Half true",
            "Mostly true",
            "True",
        ],
        lambda r: {
            "statement": r["statement"],
            "speaker": r.get("speaker"),
            "context": clip_text(r.get("context") or "", 300),
        },
        lambda r: int(r["label"]),
    ),
    max_train=600,
)


# ----------------------------------------------------------------------------------------
# Synthetic policy application (our own generator, see assay/data/policy.py)
# ----------------------------------------------------------------------------------------

POLICY = TaskSpec(
    name="policy_application",
    hf_id="synthetic",
    config=None,
    split="train",
    generator=generate_policy,
    max_train=3000,
    max_eval=300,
)


TASKS: list[TaskSpec] = [
    # topic
    AG_NEWS,
    DBPEDIA,
    BBC_NEWS,
    TREC,
    LEDGAR,
    SCOTUS,
    # intent
    BANKING77,
    CLINC,
    # sentiment
    IMDB,
    ROTTEN,
    SST2,
    AMAZON_POLARITY,
    YELP,
    AMAZON_STARS,
    APP_REVIEWS,
    SST5,
    DYNASENT,
    TWEET_SENTIMENT,
    # nli / verification / paraphrase / linguistics
    MNLI,
    SNLI,
    ANLI,
    CB,
    RTE,
    SCITAIL,
    VITAMINC,
    QQP,
    MRPC,
    MEDICAL_PAIRS,
    STSB,
    COLA,
    SUBJ,
    WIC,
    # toxicity / hate / spam
    CIVIL_TOXIC,
    CIVIL_SEVERITY,
    HATE_SPEECH,
    TWEET_HATE,
    TWEET_IRONY,
    ETHOS,
    SMS_SPAM,
    ENRON_SPAM,
    # emotion / stance
    GOEMOTIONS,
    STANCE_ABORTION,
    STANCE_ATHEISM,
    STANCE_FEMINIST,
    STANCE_HILLARY,
    STANCE_CLIMATE,
    # reading comprehension / commonsense / knowledge
    BOOLQ,
    MULTIRC,
    RACE,
    COSMOS,
    DREAM,
    HELLASWAG,
    SWAG,
    PIQA,
    SIQA,
    WINOGRANDE,
    COPA,
    COMMONSENSE_QA,
    ARC_CHALLENGE,
    ARC_EASY,
    OPENBOOKQA,
    MEDQA,
    TRUTHFUL_QA,
    # judging
    SHP,
    HH_RLHF,
    # truthfulness
    LIAR,
    # synthetic
    POLICY,
]
