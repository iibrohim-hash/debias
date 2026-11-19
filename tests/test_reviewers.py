"""CPU-only tests for reviewers A/B and their evaluation. No model is downloaded:
review_many() gets hand-written outputs through `generate_fn`."""
import json

import pandas as pd
import pytest

from src import config, contract, reviewer_eval, reviewers

ORIG = "Side B fighters attacked the checkpoint, officials said."
REV = "Side B terrorists attacked the checkpoint, officials said."
UNFAIR = {"fair": False, "issues": [{"phrase": "Side B terrorists", "problem": "R02: loaded label"}]}
FAIR = {"fair": True, "issues": []}


def fixed(*outputs):
    """generate_fn that returns the given outputs in order, one per call."""
    queue = list(outputs)

    def gen(batch):
        return [queue.pop(0) for _ in batch]
    return gen


# --- prompt ------------------------------------------------------------------
def test_rubric_has_every_guidebook_rule():
    lines = reviewers.rubric().splitlines()
    assert lines[0].startswith("R01 ") and len(lines) >= 20


def test_reviewers_a_and_b_differ_only_in_the_side():
    a = reviewers.build_messages(ORIG, REV, "palestinian")
    b = reviewers.build_messages(ORIG, REV, "israeli")
    assert a[0] == b[0]  # same system prompt
    swapped = a[1]["content"].replace(reviewers.SIDES["palestinian"], "@@").replace(
        reviewers.SIDES["israeli"], reviewers.SIDES["palestinian"]).replace("@@", reviewers.SIDES["israeli"])
    assert swapped == b[1]["content"]


def test_every_perspective_has_a_side_description():
    assert set(reviewers.SIDES) == set(config.ALL_PERSPECTIVES)


# --- validation --------------------------------------------------------------
def test_valid_outputs():
    assert reviewers.validate_output(REV, UNFAIR) == []
    assert reviewers.validate_output(REV, FAIR) == []


def test_phrase_match_ignores_case_quotes_and_spaces():
    assert reviewers.phrase_in('"side b  Terrorists"', REV)
    assert not reviewers.phrase_in("", REV)


@pytest.mark.parametrize("out, msg", [
    ({"fair": True, "issues": UNFAIR["issues"]}, "issues"),
    ({"fair": False, "issues": []}, "no issue"),
    ({"fair": False, "issues": [{"phrase": "invented words", "problem": "R02"}]}, "not in the REVISED"),
    ({"fair": "no", "issues": []}, "schema"),
])
def test_invalid_outputs(out, msg):
    assert any(msg in e for e in reviewers.validate_output(REV, out))


# --- review_many: retry and fallback ------------------------------------------
def test_first_answer_ok():
    [(res, info)] = reviewers.review_many([(ORIG, REV, "palestinian")], fixed(json.dumps(UNFAIR)))
    assert res == UNFAIR and info["retries"] == 0 and not info["fell_back"]
    contract.validate_review(res)


def test_markdown_fences_are_accepted():
    [(res, _)] = reviewers.review_many([(ORIG, REV, "palestinian")], fixed("```json\n" + json.dumps(FAIR) + "\n```"))
    assert res == FAIR


def test_retry_then_ok():
    bad = json.dumps({"fair": False, "issues": [{"phrase": "not there", "problem": "R02"}]})
    [(res, info)] = reviewers.review_many([(ORIG, REV, "israeli")], fixed(bad, json.dumps(UNFAIR)))
    assert res == UNFAIR and info["retries"] == 1 and len(info["attempts"]) == 2


def test_two_failures_never_approve():
    [(res, info)] = reviewers.review_many([(ORIG, REV, "israeli")], fixed("no json", "still none"))
    assert res["fair"] is False and res["issues"][0]["problem"] == reviewers.UNREADABLE
    assert info["fell_back"]
    contract.validate_review(res)


def test_batch_keeps_order():
    items = [(ORIG, REV, "palestinian"), (ORIG, REV, "israeli"), (ORIG, ORIG, "ukrainian")]
    out = reviewers.review_many(items, fixed(json.dumps(UNFAIR), json.dumps(FAIR), json.dumps(FAIR)), batch_size=2)
    assert [r["fair"] for r, _ in out] == [False, True, True]


def test_wrong_perspective():
    with pytest.raises(ValueError):
        reviewers.review(ORIG, REV, "american")


def test_fake_mode_is_fair():
    assert reviewers.review(ORIG, REV, "palestinian") == FAIR


# --- labeled cases -------------------------------------------------------------
def test_cases_file():
    cases = reviewer_eval.load_cases()
    assert len(cases) >= 40 and cases["case_id"].is_unique
    assert set(cases["split"]) == set(reviewer_eval.SPLITS)
    for _, c in cases.iterrows():
        assert c["unfair_to"] == "none" or c["unfair_to"] in config.PERSPECTIVES[c["conflict"]]
    for pid, pair in cases.groupby("pair_id"):
        assert sorted(pair["member"]) == ["a", "b"], pid
        assert pair["split"].nunique() == 1, f"{pid}: a mirror pair must not be split across dev/held-out"
        a, b = pair.set_index("member")["unfair_to"][["a", "b"]]
        assert (a == b == "none") or b == config.OTHER_PERSPECTIVE.get(a), f"{pid}: labels must swap"


# --- metrics -------------------------------------------------------------------
def _calls(verdicts):
    """verdicts: {(case_id, perspective): fair} for one ideal-labeled mirror pair."""
    cases = pd.DataFrame([
        {"case_id": "X1a", "pair_id": "X1", "member": "a", "split": "dev", "conflict": "israel_palestine",
         "original": ORIG, "revised": REV, "unfair_to": "palestinian"},
        {"case_id": "X1b", "pair_id": "X1", "member": "b", "split": "dev", "conflict": "israel_palestine",
         "original": ORIG, "revised": REV, "unfair_to": "israeli"},
        {"case_id": "X2a", "pair_id": "X2", "member": "a", "split": "dev", "conflict": "israel_palestine",
         "original": ORIG, "revised": ORIG, "unfair_to": "none"},
        {"case_id": "X2b", "pair_id": "X2", "member": "b", "split": "dev", "conflict": "israel_palestine",
         "original": ORIG, "revised": ORIG, "unfair_to": "none"},
    ])
    order = [(c, p) for c in cases["case_id"] for p in ("palestinian", "israeli")]
    phrase_ok = {"fair": False, "issues": [{"phrase": "Side B", "problem": "R02"}]}
    outs = [json.dumps(FAIR if verdicts.get(k, True) else phrase_ok) for k in order]
    return reviewer_eval.run("dev", "test", cases=cases, generate_fn=fixed(*outs), log=False)


def test_perfect_reviewers():
    _, m = _calls({("X1a", "palestinian"): False, ("X1b", "israeli"): False})
    s = m["summary"]
    assert s["call_accuracy"] == s["case_accuracy"] == s["symmetry"] == 1.0
    assert s["unfair_caught"] == s["fair_both_approve"] == 1.0 and s["false_alarm_rate"] == 0.0
    assert s["agreement"] == 0.5  # unfair cases: the two reviewers should disagree


def test_asymmetric_reviewer_is_caught():
    # Palestinian reviewer catches its case; the Israeli reviewer misses the mirror case.
    _, m = _calls({("X1a", "palestinian"): False})
    s = m["summary"]
    assert s["symmetry"] == 0.5 and s["unfair_caught"] == 0.5 and m["wrong_cases"] == ["X1b"]


def test_heldout_needs_final():
    with pytest.raises(ValueError):
        reviewer_eval.run("heldout", "peek")


def test_fake_dev_run_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("DEBIAS_ARTIFACTS", str(tmp_path))
    calls, m = reviewer_eval.run("dev", "fake run")
    assert len(calls) == 2 * len(reviewer_eval.load_cases("dev"))
    assert len(reviewer_eval.history()) == 1
