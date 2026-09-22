import numpy as np

from assay.conformal import (
    act_threshold,
    binomial_cdf,
    decorate,
    evaluate,
    fit,
    prediction_set,
    set_threshold,
    should_act,
)
from assay.metrics import Scored
from assay.schema import Question


def calibrated_items(n: int, k: int, seed: int, qtype: str = "choice") -> list[Scored]:
    """Predictions whose labels are drawn from the predicted distribution: perfectly calibrated."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        p = np.exp(rng.normal(0.0, 2.0, k))
        p /= p.sum()
        y = int(rng.choice(k, p=p))
        out.append(Scored(p.tolist(), np.eye(k)[y].tolist(), y, qtype, "syn", True, 0.9))
    return out


def test_binomial_cdf_matches_direct_sum():
    from math import comb

    direct = sum(comb(20, i) * 0.3**i * 0.7 ** (20 - i) for i in range(4))
    assert abs(binomial_cdf(3, 20, 0.3) - direct) < 1e-12
    assert binomial_cdf(20, 20, 0.3) == 1.0


def test_set_threshold_gives_target_coverage_on_fresh_data():
    cal = calibrated_items(4000, 4, seed=1)
    test = calibrated_items(4000, 4, seed=2)
    q = set_threshold(cal, alpha=0.1)
    covered = np.mean([1.0 - s.probs[s.label_index] <= q for s in test])
    assert 0.87 <= covered <= 0.93


def test_act_threshold_bounds_error_on_fresh_data():
    cal = calibrated_items(4000, 4, seed=3)
    test = calibrated_items(4000, 4, seed=4)
    t = act_threshold(cal, alpha=0.1, delta=0.05)
    assert t is not None
    acted = [s for s in test if s.probs.max() >= t]
    error = np.mean([int(s.probs.argmax()) != s.label_index for s in acted])
    assert error <= 0.1
    assert len(acted) > 100


def test_act_threshold_is_none_when_nothing_passes():
    rng = np.random.default_rng(0)
    items = []
    for _ in range(200):
        p = [0.6, 0.4]
        y = int(rng.integers(0, 2))  # coin flip labels: 50% error everywhere
        items.append(Scored(p, np.eye(2)[y].tolist(), y, "bool", "syn", True, 0.5))
    assert act_threshold(items, alpha=0.1, delta=0.05) is None


def test_fit_evaluate_and_decorate():
    cal = calibrated_items(1500, 3, seed=5) + calibrated_items(1500, 2, seed=6, qtype="bool")
    thresholds = fit(cal, alpha=0.1, delta=0.05)
    assert set(thresholds["types"]) == {"choice", "bool"}
    ev = evaluate(cal, thresholds)
    assert ev["choice"]["coverage"] >= 0.89
    q = Question(type="choice", instructions="?", options={"a": None, "b": None, "c": None})
    answer = {"probabilities": {"a": 0.05, "b": 0.9, "c": 0.05}}
    decorate(answer, q, thresholds)
    assert answer["act"] == should_act([0.05, 0.9, 0.05], "choice", thresholds)
    assert "b" in answer["set"]
    assert [q.keys[i] for i in prediction_set([0.05, 0.9, 0.05], "choice", thresholds)] == answer[
        "set"
    ]
