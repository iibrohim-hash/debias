import pytest

from src import config, pipeline


def test_sentence_start_word_is_not_a_new_name():
    # Real case from the first full run: the rewrite added "The" at the start.
    original = '"Friends" actor David Schwimmer called out some advocates.'
    revised = 'The "Friends" actor David Schwimmer called out some advocates.'
    assert pipeline.fact_check(original, revised) == {"ok": True, "new_numbers": [], "new_names": []}


def test_new_numbers_and_names_still_fail():
    out = pipeline.fact_check("Troops fired near the border.", "Israeli troops fired 20 shots near the border.")
    assert not out["ok"] and out["new_numbers"] == ["20"] and out["new_names"] == ["Israeli"]


def test_pipeline_runs_end_to_end_in_fake_mode():
    result = pipeline.run("Hamas terrorists attacked the town.", log=False)
    assert result["decision"] in ("approve", "escalate", "retry", "block", "unchanged")


# ---------------------------------------------------------------- evaluator rules
OK = {"ok": True, "new_numbers": [], "new_names": []}
BAD = {"ok": False, "new_numbers": ["20"], "new_names": []}
FAIR = {"fair": True, "issues": []}
UNFAIR = {"fair": False, "issues": [{"phrase": "x", "problem": "R02"}]}


@pytest.mark.parametrize("rule", config.EVALUATOR_RULES)
def test_both_rules_approve_and_block_the_same_way(rule):
    assert pipeline.evaluate(OK, {"a": FAIR, "b": FAIR}, rule) == "approve"
    assert pipeline.evaluate(BAD, {"a": FAIR, "b": FAIR}, rule) == "retry"
    assert pipeline.evaluate(OK, {"a": UNFAIR, "b": UNFAIR}, rule) == "retry"
    assert pipeline.evaluate(OK, {"a": UNFAIR, "b": UNFAIR}, rule, last_round=True) == "block"
    assert pipeline.evaluate(OK, {}, rule) == "retry"  # unchanged text is never approved
    assert pipeline.evaluate(OK, {}, rule, last_round=True) == "block"


def test_disagreement_escalates_at_once_only_with_the_current_rule():
    split = {"a": UNFAIR, "b": FAIR}
    assert pipeline.evaluate(OK, split, "disagree_escalates") == "escalate"
    assert pipeline.evaluate(OK, split, "retry_first") == "retry"
    assert pipeline.evaluate(OK, split, "retry_first", last_round=True) == "escalate"


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError):
        pipeline.evaluate(OK, {"a": FAIR}, "majority")


def _patch_steps(monkeypatch, rewrites, verdicts):
    """Real detector/explainer/RAG fakes; scripted rewriter and reviewers."""
    rewrites, calls = iter(rewrites), {"review": 0, "feedback": []}

    def fake_rewrite(text, spans, rules, feedback):
        calls["feedback"].append(feedback)
        revised = next(rewrites)
        changes = [] if revised == text else [{"original": "terrorists", "revised": "fighters", "reason": "R02"}]
        return {"revised": revised, "changes": changes}

    def fake_review(original, revised, perspective):
        calls["review"] += 1
        return verdicts(perspective)

    monkeypatch.setattr(pipeline, "rewrite", fake_rewrite)
    monkeypatch.setattr(pipeline, "review", fake_review)
    monkeypatch.setattr(pipeline, "detect", lambda t, c: {"status": "biased", "score": 0.9, "label": "against_SIDE_B"})
    return calls


TEXT = "Hamas terrorists attacked the town."
FIXED = "Hamas fighters attacked the town."


def test_unchanged_rewrite_skips_the_reviewers_and_retries(monkeypatch):
    calls = _patch_steps(monkeypatch, [TEXT, FIXED], lambda p: FAIR)
    result, trace = pipeline.run_detailed(TEXT, log=False)
    assert result["decision"] == "approve" and result["final"] == FIXED
    assert trace["rounds"][0]["unchanged"] and trace["rounds"][0]["reviews"] == {}
    assert calls["review"] == 2  # only round 2 was reviewed
    assert pipeline.UNCHANGED_FEEDBACK in calls["feedback"][1]


def test_retry_first_gives_the_rewriter_more_rounds(monkeypatch):
    split = lambda p: UNFAIR if p == "palestinian" else FAIR
    _patch_steps(monkeypatch, [FIXED] * 3, split)
    result, trace = pipeline.run_detailed(TEXT, log=False, rule="disagree_escalates")
    assert result["decision"] == "escalate" and len(trace["rounds"]) == 1

    _patch_steps(monkeypatch, [FIXED] * 3, split)
    result, trace = pipeline.run_detailed(TEXT, log=False, rule="retry_first")
    assert result["decision"] == "escalate" and len(trace["rounds"]) == config.MAX_ROUNDS
    assert result["final"] == TEXT and trace["evaluator_rule"] == "retry_first"
    assert '"x": R02' in trace["rounds"][1]["feedback_in"]
