"""The result plots draw from saved result tables, with no model loaded."""
import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from src import plots  # noqa: E402


def _f_rows(n=12, seed=0):
    rng = np.random.default_rng(seed)
    r = pd.DataFrame({
        "id": range(1, n + 1), "split": ["dev"] * 4 + ["held-out"] * (n - 4),
        "rounds": [0, 1, 2, 3] * (n // 4), "decision": ["unchanged", "approve", "escalate", "block"] * (n // 4),
        "second_before": rng.uniform(0.4, 1, n), "second_reference": rng.uniform(0.2, 0.8, n),
        "seconds": rng.uniform(0, 100, n),
    })
    r["second_last_rewrite"] = (r["second_before"] - 0.2).clip(0, 1)
    r["second_after"] = np.where(r["decision"] == "approve", r["second_last_rewrite"], r["second_before"])
    return r


def _h_rows(rule_shift=0.0):
    rows = []
    for i in range(1, 11):
        pair = "" if i <= 2 else f"S{(i + 1) // 2:02d}"
        base = 0.1 * i
        rows.append({"id": i, "type": "neutral" if i <= 2 else "swap", "swap_pair_id": pair,
                     "rounds": i % 4, "decision": ["unchanged", "approve", "escalate", "block"][i % 4],
                     "second_base": base, "second_last_rewrite": max(base - 0.1, 0),
                     "second_resp": max(base - rule_shift, 0), "seconds": 10.0 * i})
    return pd.DataFrame(rows)


def test_f_plots(tmp_path):
    r = _f_rows()
    figs = [
        plots.score_ladder(r, plots.F_STAGES, 0.55),
        plots.score_ladder(r, plots.F_STAGES, 0.55, only_rewritten=True),
        plots.decisions({"disagree_escalates": r, "retry_first": r}),
        plots.before_after(r, "second_before", "second_last_rewrite", 0.55),
        plots.change_per_text(r, "second_before", "second_last_rewrite"),
        plots.by_group(r, "split", plots.F_STAGES, 0.55),
        plots.cost({"disagree_escalates": r, "retry_first": r}),
    ]
    for i, fig in enumerate(figs):
        assert plots.save(fig, tmp_path / "figs" / f"{i}.png").stat().st_size > 0


def test_h_plots(tmp_path):
    g, h = _h_rows(), _h_rows(0.05)
    by_rule = {"disagree_escalates": g, "retry_first": h}
    figs = [
        plots.score_ladder(h, plots.H_STAGES),
        plots.by_group(h, "type", plots.H_STAGES),
        plots.swap_pairs(by_rule, "second_resp"),
        plots.decisions(by_rule),
    ]
    for i, fig in enumerate(figs):
        assert plots.save(fig, tmp_path / f"{i}.png").exists()


def test_stages_missing_column_are_skipped():
    r = _f_rows().drop(columns=["second_reference"])
    ax = plots.score_ladder(r, plots.F_STAGES).axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["original", "last rewrite", "final (reader sees)"]


def test_mean_ci():
    m, lo, hi = plots.mean_ci([0.2, 0.4, 0.6, 0.8])
    assert m == pytest.approx(0.5) and lo <= m <= hi
    assert np.isnan(plots.mean_ci([])[0])
