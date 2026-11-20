"""Evaluate reviewers A and B on the labeled cases (data/reviewer_cases.csv).

The reviewers are not trained; their prompt is tuned. This module is the
"validation set" for that tuning, like the detector's val/test splits:

    from src import reviewer_eval
    calls, m = reviewer_eval.run("dev", note="v1: first prompt")   # tune on dev, as often as needed
    reviewer_eval.history()                                         # every version and its scores
    calls, m = reviewer_eval.run("heldout", note="final", final=True)  # ONCE, at the end

Each case is reviewed by both reviewers of its conflict. A reviewer should say
"not fair" exactly when the case is labeled unfair to its side.

Metrics:
    call_accuracy        share of single reviewer verdicts that match the label
    case_accuracy        both verdicts of a case right at once
    fair_both_approve    fair cases where BOTH reviewers say fair (agreement on fair rewrites)
    unfair_caught        unfair cases where the reviewer of the harmed side says not fair
    false_alarm_rate     verdicts "not fair" where the label says that side was treated fairly
    agreement            cases where the two reviewers give the same verdict
    symmetry             mirror pairs where swapping the sides swaps the verdicts
    fallback_rate        verdicts where the output could not be read even after the retry
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from . import config, reviewers, rewriter, utils

SPLITS = ("dev", "heldout")


def load_cases(split: str | None = None, path=None) -> pd.DataFrame:
    cases = pd.read_csv(path or config.REVIEWER_CASES_PATH, keep_default_na=False)
    if split:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        cases = cases[cases["split"] == split]
    return cases.reset_index(drop=True)


def _log_path():
    return utils.artifacts_dir() / "results" / "reviewer_runs.jsonl"


def history() -> pd.DataFrame:
    """One row per logged run: version, note, split and the main metrics."""
    path = _log_path()
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame([{k: r[k] for k in ("time_utc", "prompt_version", "split", "note", "model_id")}
                         | r["metrics"]["summary"] for r in rows])


def review_cases(cases: pd.DataFrame, generate_fn=None, batch_size: int = 8) -> pd.DataFrame:
    """One row per (case, reviewer) with the verdict and whether it matches the label."""
    items, meta = [], []
    for _, c in cases.iterrows():
        for p in config.PERSPECTIVES[c["conflict"]]:
            items.append((c["original"], c["revised"], p))
            meta.append({"case_id": c["case_id"], "pair_id": c["pair_id"], "member": c["member"],
                         "conflict": c["conflict"], "perspective": p, "unfair_to": c["unfair_to"],
                         "expected_fair": c["unfair_to"] != p})
    results = reviewers.review_many(items, generate_fn=generate_fn, batch_size=batch_size)
    rows = []
    for m, (res, info) in zip(meta, results):
        rows.append(m | {
            "fair": res["fair"],
            "correct": res["fair"] == m["expected_fair"],
            "phrases": " | ".join(i["phrase"] for i in res["issues"]),
            "problems": " | ".join(i["problem"] for i in res["issues"]),
            "other_side_phrases": " | ".join(i["phrase"] for i in info.get("other_side_issues", [])),
            "retries": info["retries"],
            "fell_back": info["fell_back"],
            "first_errors": " | ".join(info["attempts"][0]["errors"]) if info["attempts"] else "",
            "raw": info["attempts"][-1]["raw"] if info["attempts"] else "",
        })
    return pd.DataFrame(rows)


def compute_metrics(calls: pd.DataFrame) -> dict:
    def rate(s) -> float | None:
        return round(float(s.mean()), 3) if len(s) else None

    by_case = calls.groupby("case_id")
    case_ok = by_case["correct"].all()
    labels = by_case["unfair_to"].first()
    same = by_case["fair"].nunique() == 1
    both_fair = by_case["fair"].all()
    harmed = calls[calls["perspective"] == calls["unfair_to"]]

    # Symmetry: verdict of member a under perspective P == verdict of member b under the other perspective.
    sym = []
    for _, pair in calls.groupby("pair_id"):
        a = pair[pair["member"] == "a"].set_index("perspective")["fair"]
        b = pair[pair["member"] == "b"].set_index("perspective")["fair"]
        if len(a) and len(b):
            sym.append(all(a[p] == b[config.OTHER_PERSPECTIVE[p]] for p in a.index))

    summary = {
        "n_cases": int(calls["case_id"].nunique()),
        "n_calls": len(calls),
        "call_accuracy": rate(calls["correct"]),
        "case_accuracy": rate(case_ok),
        "fair_both_approve": rate(both_fair[labels == "none"]),
        "unfair_caught": rate(~harmed["fair"]),
        "false_alarm_rate": rate(~calls.loc[calls["expected_fair"], "fair"]),
        "agreement": rate(same),
        "symmetry": rate(pd.Series(sym, dtype=bool)),
        "fallback_rate": rate(calls["fell_back"]),
        "retry_rate": rate(calls["retries"] > 0),
    }
    per_perspective = {p: rate(g["correct"]) for p, g in calls.groupby("perspective")}
    wrong = calls.loc[~calls["correct"], "case_id"].unique().tolist()
    return {"summary": summary, "per_perspective": per_perspective, "wrong_cases": wrong}


def run(split: str, note: str, final: bool = False, generate_fn=None, batch_size: int = 8,
        cases: pd.DataFrame | None = None, log: bool = True) -> tuple[pd.DataFrame, dict]:
    """Review every case of a split, compute metrics and log them with the prompt version.

    The held-out split needs final=True: score it once, after tuning on dev is finished.
    """
    if split == "heldout" and not final:
        raise ValueError("held-out is scored ONCE at the end: pass final=True when tuning on dev is done")
    if split == "heldout" and log:
        done = history()
        if not done.empty and (done["split"] == "heldout").any():
            print("WARNING: held-out was already scored; report the FIRST held-out run, not this one.")
    cases = load_cases(split) if cases is None else cases
    calls = review_cases(cases, generate_fn=generate_fn, batch_size=batch_size)
    metrics = compute_metrics(calls)
    version = reviewers.prompt_version()
    model_id = rewriter.loaded_model_id() if generate_fn is None and not config.USE_FAKES else None
    if log:
        record = {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "prompt_version": version, "split": split, "note": note, "model_id": model_id,
                  "git_commit": utils.git_commit(), "metrics": metrics}
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        calls.to_csv(path.parent / f"reviewer_calls_{split}_{version}.csv", index=False)
        if split == "heldout":
            utils.save_result(f"reviewers_heldout_{version}", metrics, {"note": note, "model_id": model_id})
    return calls, metrics
