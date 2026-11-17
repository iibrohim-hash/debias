"""Run before pushing: `pytest`. Checks every shared function still matches the interface."""
import pytest

from src import config, contract
from src.detector import detect, explain
from src.pipeline import run
from src.rag import load_entries, retrieve
from src.reviewers import review
from src.rewriter import rewrite

TEXT = "Side A terrorists butchered civilians, officials said."


@pytest.mark.parametrize("conflict", config.CONFLICTS)
def test_detect_and_explain(conflict):
    contract.validate_detect(detect(TEXT, conflict))
    contract.validate_explain(explain(TEXT, conflict))


def test_bad_conflict_rejected():
    with pytest.raises(ValueError):
        detect(TEXT, "other")


def test_guidebook_has_r01():
    assert any(e.startswith("## R01") for e in load_entries())


@pytest.mark.parametrize("k", [3, 4])
def test_retrieve(k):
    contract.validate_retrieve(retrieve(TEXT, k), k)


def test_rewrite():
    contract.validate_rewrite(rewrite(TEXT, ["terrorists"], retrieve(TEXT)))
    contract.validate_rewrite(rewrite(TEXT, ["terrorists"], retrieve(TEXT), feedback="fix X"))


@pytest.mark.parametrize("perspective", config.ALL_PERSPECTIVES)
def test_review(perspective):
    contract.validate_review(review(TEXT, TEXT, perspective))


def test_labels_fixed():
    assert config.LABELS == ("unbiased", "against_SIDE_A", "against_SIDE_B")


@pytest.mark.parametrize("conflict", config.CONFLICTS)
def test_pipeline_end_to_end(conflict):
    out = run(TEXT, conflict)
    assert out["decision"] in ("unchanged", "approve", "retry", "block", "escalate")
    assert isinstance(out["change_note"], str)


def test_pipeline_detailed_trace_and_force():
    from src.pipeline import run_detailed

    result, trace = run_detailed(TEXT, "israel_palestine", log=False, force=True)
    assert trace["detect"]["status"] in config.DETECT_STATUSES
    assert trace["rules"][0].startswith("## R01")
    assert trace["rounds"] and "total" in trace["seconds"]
    assert result["change_note"]
