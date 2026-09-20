import math
import random

import numpy as np
import pytest

from sezgi.calibrate import fit_temperature, rescale
from sezgi.data.registry import sord_target
from sezgi.metrics import Scored, auc, expected_calibration_error, summarize


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def test_ece_perfect_and_worst():
    conf = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    correct = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=float)
    assert expected_calibration_error(conf, correct) == pytest.approx(0.0, abs=1e-9)
    assert expected_calibration_error(conf, np.zeros(10)) == pytest.approx(0.9)


def test_summarize_basic():
    items = [
        Scored([0.7, 0.3], [1.0, 0.0], 0, "noul", "a", True, 0.9),
        Scored([0.2, 0.8], [1.0, 0.0], 0, "noul", "a", True, 0.1),
    ]
    s = summarize(items)
    assert s["accuracy"] == 0.5
    assert s["brier"] == pytest.approx(((0.3**2 + 0.3**2) + (0.8**2 + 0.8**2)) / 2)
    assert s["nll"] == pytest.approx((-math.log(0.7) - math.log(0.2)) / 2)


def test_auc():
    assert auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == pytest.approx(0.75)


def test_sord_target_peaks_at_center_and_sums_to_one():
    t = sord_target(1.0, 3, sigma=0.5)
    assert max(t, key=t.get) == "1"
    assert sum(t.values()) == pytest.approx(1.0)
    assert t["0"] == pytest.approx(t["2"])


def test_fit_temperature_recovers_overconfidence():
    rng = random.Random(0)
    items = []
    true_t = 2.0
    for _ in range(3000):
        logits = [rng.gauss(0, 2) for _ in range(3)]
        p_true = _softmax(logits)
        label = int(np.argmax(np.random.default_rng(rng.randrange(1 << 30)).multinomial(1, p_true)))
        # the model reports sharpened probabilities: softmax(logits * true_t)
        p_model = _softmax(np.asarray(logits) * true_t)
        target = [0.0, 0.0, 0.0]
        target[label] = 1.0
        items.append(Scored(p_model.tolist(), target, label, "choice", "s", True, 0.5))
    t = fit_temperature(items)
    assert abs(t - true_t) < 0.25
    before = summarize(items)["ece"]
    after = summarize(rescale(items, t))["ece"]
    assert after < before
