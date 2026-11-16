import numpy as np
import pytest

from src import metrics


def test_predict_and_scores():
    probs = np.array([[0.9, 0.05, 0.05], [0.2, 0.7, 0.1], [0.3, 0.1, 0.6], [0.6, 0.1, 0.3]])
    assert metrics.bias_score(probs) == pytest.approx([0.1, 0.8, 0.7, 0.4])
    assert metrics.predict(probs, 0.5).tolist() == [0, 1, 2, 0]
    assert metrics.predict(probs, 0.3).tolist() == [0, 1, 2, 2]


def test_per_class_perfect_and_ci():
    y = np.array([0, 1, 2, 0, 1, 2] * 10)
    res = metrics.per_class(y, y)
    assert res["macro_f1"] == 1.0
    ci = metrics.bootstrap_ci(y, y, n=50)
    assert ci["macro_f1"] == [1.0, 1.0]


def test_tune_thresholds_on_separable_data():
    y = np.array([0] * 20 + [1] * 10 + [2] * 10)
    probs = np.array([[0.9, 0.05, 0.05]] * 20 + [[0.2, 0.7, 0.1]] * 10 + [[0.2, 0.1, 0.7]] * 10)
    t = metrics.tune_thresholds(y, probs)
    assert t["val_macro_f1"] == 1.0
    assert 0.1 < t["biased"] <= 0.8
    assert t["uncertain_low"] <= t["biased"]
    assert "macro-F1" in metrics.format_table({"x": metrics.evaluate(y, probs, t["biased"], ci=False)})
