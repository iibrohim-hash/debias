"""Result plots for notebooks F (test sentences) and H (probes, both oversight rules).

Every function takes the results table a notebook saved (a pandas DataFrame read from the CSV), draws one
figure and returns it. Nothing here loads a model, so the plots run without a GPU:

    from src import plots
    fig = plots.score_ladder(runs, plots.F_STAGES)
    plots.save(fig, ARTIFACTS / "figures" / "f_score_ladder.png")

Colours are fixed per meaning, so the same thing has the same colour in every figure.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# Light chart surface and ink.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NEUTRAL = "#8a8985"  # baseline / "no rewrite"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

DECISION_ORDER = ("approve", "escalate", "block", "unchanged")
DECISION_COLORS = {"approve": BLUE, "escalate": ORANGE, "block": AQUA, "unchanged": NEUTRAL}
DECISION_LABELS = {"approve": "approved", "escalate": "to a person", "block": "blocked",
                   "unchanged": "unchanged (detector: not biased)"}
RULE_COLORS = {"baseline": NEUTRAL, "disagree_escalates": BLUE, "retry_first": ORANGE}

# (column, label) in reading order. F: test sentences; H: probes.
F_STAGES = [("second_before", "original"), ("second_last_rewrite", "last rewrite"),
            ("second_after", "final (reader sees)"), ("second_reference", "human reference")]
H_STAGES = [("second_base", "baseline answer"), ("second_last_rewrite", "last rewrite"),
            ("second_resp", "final (reader sees)")]


def _plt():
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False,
    })
    return plt


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _threshold(ax, value, axis="y"):
    if value is None:
        return
    line = ax.axhline if axis == "y" else ax.axvline
    line(value, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    if axis == "y":
        ax.annotate(f"biased\n≥ {value:.2f}", (1, value), xycoords=("axes fraction", "data"),
                    xytext=(4, 0), textcoords="offset points", ha="left", va="center", fontsize=8, color=INK_2,
                    annotation_clip=False)


def mean_ci(x, n=config.BOOTSTRAP_SAMPLES, seed=config.SEED):
    """Mean and bootstrap 95% CI."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    if len(x) == 1:
        return x[0], x[0], x[0]
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, (n, len(x))).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return x.mean(), lo, hi


def score_ladder(rows: pd.DataFrame, stages, threshold=None, title=None, only_rewritten=False):
    """Mean second-detector score (95% CI) at each stage: what rewriting does and what reaches the reader."""
    plt = _plt()
    if only_rewritten:
        rows = rows[rows["rounds"] > 0]
    stages = [(c, l) for c, l in stages if c in rows.columns]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, (col, label) in enumerate(stages):
        m, lo, hi = mean_ci(rows[col])
        color = NEUTRAL if i == 0 else BLUE
        ax.bar(i, m, width=0.55, color=color, edgecolor=SURFACE, linewidth=2)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], color=INK, capsize=4, linewidth=1.2)
        ax.annotate(f"{m:.2f}", (i, hi), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(range(len(stages)), [l for _, l in stages])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("second-detector bias score")
    _grid(ax)
    _threshold(ax, threshold)
    scope = f"{len(rows)} rewritten" if only_rewritten else f"{len(rows)}"
    ax.set_title(title or f"Bias score at each stage (mean, 95% CI, n = {scope})")
    fig.tight_layout()
    return fig


def decisions(tables: dict, title="What oversight decided"):
    """Stacked horizontal bar of decisions per run, e.g. {"disagree_escalates": g, "retry_first": h}."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 0.9 + 0.75 * len(tables)))
    names = list(tables)
    for y, name in enumerate(names):
        counts = tables[name]["decision"].value_counts()
        n, left = len(tables[name]), 0
        for d in DECISION_ORDER:
            k = int(counts.get(d, 0))
            if not k:
                continue
            ax.barh(y, k, left=left, height=0.6, color=DECISION_COLORS[d], edgecolor=SURFACE, linewidth=2,
                    label=DECISION_LABELS[d] if y == 0 else None)
            if k / n >= 0.06:
                ax.text(left + k / 2, y, str(k), ha="center", va="center", fontsize=9, color="white", weight="bold")
            left += k
    ax.set_yticks(range(len(names)), names)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlabel("number of answers")
    _grid(ax, "x")
    handles, labels = ax.get_legend_handles_labels()
    seen = {l: h for h, l in zip(handles, labels)}
    ax.legend(seen.values(), seen.keys(), ncol=4, loc="upper left", bbox_to_anchor=(0, -0.35 if len(names) < 3 else -0.2),
              fontsize=8, handlelength=1)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def before_after(rows: pd.DataFrame, before: str, after: str, threshold=None,
                 labels=("before", "last rewrite"), title=None, diagonal="no change (diagonal)",
                 note="{k} of {n} lower after rewriting"):
    """One dot per rewritten text: score before (x) vs after (y). Below the diagonal = less biased."""
    plt = _plt()
    r = rows[rows["rounds"] > 0]
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle=(0, (4, 3)), label=diagonal)
    for d in DECISION_ORDER[:3]:
        s = r[r["decision"] == d]
        if len(s):
            ax.scatter(s[before], s[after], s=42, color=DECISION_COLORS[d], edgecolor=SURFACE, linewidth=1.5,
                       label=f"{DECISION_LABELS[d]} ({len(s)})", zorder=3)
    if threshold is not None:
        ax.axhline(threshold, color=GRID, linewidth=1)
        ax.axvline(threshold, color=GRID, linewidth=1)
    lower = int((r[after] < r[before] - 0.02).sum())
    ax.text(0.98, 0.03, note.format(k=lower, n=len(r)), transform=ax.transAxes, ha="right",
            fontsize=8, color=INK_2)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel(f"bias score, {labels[0]}")
    ax.set_ylabel(f"bias score, {labels[1]}")
    ax.legend(loc="upper left", fontsize=8, handletextpad=0.2)
    ax.set_title(title or "Each rewritten text, before vs after")
    fig.tight_layout()
    return fig


def change_per_text(rows: pd.DataFrame, before: str, after: str, label_col="id", title=None):
    """Bar per rewritten text: after - before, sorted; negative = less biased. Coloured by decision."""
    plt = _plt()
    r = rows[rows["rounds"] > 0].copy()
    r["delta"] = r[after] - r[before]
    r = r.sort_values("delta")
    fig, ax = plt.subplots(figsize=(max(6.0, 0.22 * len(r) + 1.5), 3.6))
    x = np.arange(len(r))
    ax.bar(x, r["delta"], width=0.8, color=[DECISION_COLORS[d] for d in r["decision"]], edgecolor=SURFACE,
           linewidth=1)
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set_xticks(x, r[label_col].astype(str), fontsize=7)
    ax.set_xlabel(label_col)
    ax.set_ylabel("score change (after − before)")
    _grid(ax)
    plt_handles = [plt.Rectangle((0, 0), 1, 1, color=DECISION_COLORS[d]) for d in DECISION_ORDER[:3]
                   if (r["decision"] == d).any()]
    ax.legend(plt_handles, [DECISION_LABELS[d] for d in DECISION_ORDER[:3] if (r["decision"] == d).any()],
              fontsize=8, ncol=3, loc="upper left")
    ax.set_title(title or f"Change in bias score per rewritten text (n = {len(r)}; below 0 = less biased)")
    fig.tight_layout()
    return fig


def by_group(rows: pd.DataFrame, group: str, stages, threshold=None, title=None):
    """Mean score per group (e.g. probe type, dev/held-out) at each stage, grouped bars."""
    plt = _plt()
    stages = [(c, l) for c, l in stages if c in rows.columns]
    groups = list(rows[group].drop_duplicates().sort_values())
    colors = [NEUTRAL, BLUE, ORANGE, AQUA]
    w = 0.8 / len(stages)
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for j, (col, label) in enumerate(stages):
        means = [rows.loc[rows[group] == g, col].mean() for g in groups]
        xs = np.arange(len(groups)) + (j - (len(stages) - 1) / 2) * w
        ax.bar(xs, means, width=w, color=colors[j], edgecolor=SURFACE, linewidth=2, label=label)
        for x, m in zip(xs, means):
            ax.text(x, m + 0.015, f"{m:.2f}", ha="center", fontsize=7, color=INK_2)
    counts = rows[group].value_counts()
    ax.set_xticks(range(len(groups)), [f"{g} (n = {counts[g]})" for g in groups])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean second-detector score")
    _grid(ax)
    _threshold(ax, threshold)
    ax.legend(fontsize=8, ncol=len(stages), loc="upper left")
    ax.set_title(title or f"Bias score by {group}")
    fig.tight_layout()
    return fig


def swap_pairs(tables: dict, score_col: str, base_col: str = "second_base", title=None):
    """Symmetry check: for each swap pair, |score A - score B| for the baseline and each responsible run.

    `tables`: {"disagree_escalates": g, "retry_first": h}, each with id, swap_pair_id and the score columns.
    """
    plt = _plt()
    first = next(iter(tables.values()))

    def gaps(df, col):
        p = df[df["swap_pair_id"].astype(str) != ""].sort_values("id")
        p = p[p.groupby("swap_pair_id")["id"].transform("count") == 2]
        g = p.groupby("swap_pair_id")[col].agg(lambda s: abs(s.iloc[0] - s.iloc[1]))
        return g

    series = {"baseline": gaps(first, base_col)} | {name: gaps(df, score_col) for name, df in tables.items()}
    pairs = list(series["baseline"].sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(7.2, 0.28 * len(pairs) + 1.4))
    offsets = np.linspace(-0.2, 0.2, len(series))
    for (name, g), off in zip(series.items(), offsets):
        ax.scatter(g.reindex(pairs), np.arange(len(pairs)) + off, s=30, color=RULE_COLORS.get(name, BLUE),
                   edgecolor=SURFACE, linewidth=1, label=f"{name} (mean {g.mean():.2f})", zorder=3)
    for y in range(len(pairs) + 1):
        ax.axhline(y - 0.5, color=GRID, linewidth=0.6, zorder=1)
    ax.set_yticks(range(len(pairs)), pairs, fontsize=8)
    ax.set_xlim(-0.01, 1)
    ax.set_ylim(len(pairs) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("|score of side A − score of side B|  (0 = both sides treated alike)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(title or "Swap and loaded pairs: score gap between the two sides")
    fig.tight_layout()
    return fig


def cost(tables: dict, title="Cost of each rule"):
    """Mean rounds per rewritten answer and mean seconds per answer, one small panel each."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    names = list(tables)
    for ax, (col, label, fmt, rewritten) in zip(axes, [("rounds", "mean rounds (rewritten answers)", "{:.2f}", True),
                                                       ("seconds", "mean seconds per answer", "{:.0f} s", False)]):
        vals = []
        for n in names:
            df = tables[n]
            vals.append((df[df["rounds"] > 0] if rewritten else df)[col].mean())
        ax.barh(range(len(names)), vals, height=0.55, color=[RULE_COLORS.get(n, BLUE) for n in names],
                edgecolor=SURFACE, linewidth=2)
        for y, v in enumerate(vals):
            ax.text(v, y, " " + fmt.format(v), va="center", fontsize=9)
        ax.set_yticks(range(len(names)), names if ax is axes[0] else [""] * len(names))
        ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()
        ax.set_xlim(0, max(vals) * 1.25 if vals else 1)
        ax.set_title(label, fontsize=9, weight="normal", color=INK_2)
        _grid(ax, "x")
    fig.suptitle(title, x=0.02, ha="left", fontsize=11, weight="bold")
    fig.tight_layout()
    return fig


def save(fig, path, dpi=200):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
