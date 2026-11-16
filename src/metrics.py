"""Detector metrics: decision rule, threshold tuning, per-class F1 with bootstrap CIs.

Numpy only, so it runs anywhere and both detectors (XLM-R and the second
mDeBERTa detector) are scored the same way.
"""
from __future__ import annotations

import numpy as np

from . import config

U = config.LABEL2ID["unbiased"]
A = config.LABEL2ID["against_SIDE_A"]
B = config.LABEL2ID["against_SIDE_B"]


def bias_score(probs: np.ndarray) -> np.ndarray:
    """s = 1 - P(unbiased)."""
    return 1.0 - np.asarray(probs)[:, U]


def predict(probs: np.ndarray, threshold: float) -> np.ndarray:
    """unbiased if s < threshold, otherwise the more likely "against" class."""
    probs = np.asarray(probs)
    against = np.where(probs[:, A] >= probs[:, B], A, B)
    return np.where(bias_score(probs) < threshold, U, against)


def per_class(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Precision, recall, F1 per class plus macro-F1."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {}
    for c, name in config.ID2LABEL.items():
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[name] = {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(np.sum(y_true == c))}
    out["macro_f1"] = float(np.mean([out[n]["f1"] for n in config.LABELS]))
    out["accuracy"] = float(np.mean(y_true == y_pred))
    return out


def bootstrap_ci(y_true, y_pred, n: int = config.BOOTSTRAP_SAMPLES, level: float = config.CI_LEVEL,
                 seed: int = config.SEED) -> dict:
    """Percentile bootstrap CIs (resampling posts) for every metric of per_class()."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        samples.append(per_class(y_true[idx], y_pred[idx]))
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100

    def ci(get):
        vals = np.array([get(s) for s in samples])
        return [float(np.percentile(vals, lo)), float(np.percentile(vals, hi))]

    out = {}
    for name in config.LABELS:
        out[name] = {m: ci(lambda s, name=name, m=m: s[name][m]) for m in ("precision", "recall", "f1")}
    out["macro_f1"] = ci(lambda s: s["macro_f1"])
    return out


def evaluate(y_true, probs, threshold: float, ci: bool = True) -> dict:
    y_pred = predict(probs, threshold)
    res = {"threshold": threshold, "n": int(len(y_true)), "point": per_class(y_true, y_pred)}
    if ci:
        res["ci"] = bootstrap_ci(y_true, y_pred)
    return res


def tune_thresholds(y_val, probs_val, grid: np.ndarray | None = None, uncertain_recall: float = 0.95) -> dict:
    """Pick thresholds on VALIDATION data only.

    biased: the threshold on s that maximises macro-F1.
    uncertain_low: the highest value <= biased such that `uncertain_recall` of the
    truly biased validation posts have s >= uncertain_low. Posts in
    [uncertain_low, biased) are sent to a human ("uncertain").
    """
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    y_val = np.asarray(y_val)
    scores = [(per_class(y_val, predict(probs_val, t))["macro_f1"], t) for t in grid]
    best_f1, best_t = max(scores, key=lambda x: (x[0], -abs(x[1] - 0.5)))
    s_biased = bias_score(probs_val)[y_val != U]
    low = float(np.quantile(s_biased, 1 - uncertain_recall)) if len(s_biased) else best_t
    return {"biased": float(best_t), "uncertain_low": float(min(low, best_t)), "val_macro_f1": float(best_f1)}


def format_table(results: dict[str, dict]) -> str:
    """Markdown table: rows = runs/datasets, columns = per-class F1 [95% CI] and macro-F1."""
    head = "| run | n | " + " | ".join(f"F1 {n}" for n in config.LABELS) + " | macro-F1 |"
    lines = [head, "|" + "---|" * (len(config.LABELS) + 3)]
    for run, r in results.items():
        cells = []
        for name in [*config.LABELS, "macro_f1"]:
            point = r["point"][name]["f1"] if name != "macro_f1" else r["point"]["macro_f1"]
            if "ci" in r:
                lo, hi = r["ci"][name]["f1"] if name != "macro_f1" else r["ci"]["macro_f1"]
                cells.append(f"{point:.3f} [{lo:.3f}, {hi:.3f}]")
            else:
                cells.append(f"{point:.3f}")
        lines.append(f"| {run} | {r['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
