"""End-to-end pipeline (integration).

This is a plain sequential version of section 3 of the plan, so the team can run
10 examples end to end in Week 10 using whatever pieces exist (real or fake).
In Week 11 the same steps move into CrewAI agents; keep the decision rules here
as the reference.

    from src.pipeline import run
    result = run("Some chatbot answer ...", conflict="israel_palestine")
"""
from __future__ import annotations

import re

from . import config, contract, utils
from .detector import detect, explain
from .rag import retrieve
from .reviewers import review
from .rewriter import rewrite

_NUMBER = re.compile(r"\d[\d,.]*")
_NAME = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")


def fact_check(original: str, revised: str) -> dict:
    """Cheap checks: no new numbers, no new capitalised names.

    TODO: add the NLI check (config.NLI_MODEL): original must entail revised.
    """
    new_numbers = sorted(set(_NUMBER.findall(revised)) - set(_NUMBER.findall(original)))
    orig_words = set(original.split())
    new_names = sorted(
        n for n in set(_NAME.findall(revised)) - set(_NAME.findall(original))
        if not all(w in orig_words for w in n.split())  # ignore sentence-initial capitals
    )
    return {"ok": not new_numbers and not new_names, "new_numbers": new_numbers, "new_names": new_names}


def evaluate(fact: dict, reviews: dict[str, dict]) -> str:
    """approve | retry | escalate (block is decided after the last round)."""
    fair = [r["fair"] for r in reviews.values()]
    if len(set(fair)) > 1:
        return "escalate"  # reviewers disagree -> human
    if fact["ok"] and all(fair):
        return "approve"
    return "retry"


def _feedback(fact: dict, reviews: dict[str, dict]) -> str:
    lines = []
    if fact["new_numbers"]:
        lines.append(f"Do not add numbers: {fact['new_numbers']}")
    if fact["new_names"]:
        lines.append(f"Do not add names or actors: {fact['new_names']}")
    for persp, r in reviews.items():
        for issue in r["issues"]:
            lines.append(f"[{persp}] \"{issue['phrase']}\": {issue['problem']}")
    return "\n".join(lines)


def run(text: str, conflict: str = config.DEFAULT_CONFLICT, log: bool = True) -> dict:
    config.check_conflict(conflict)
    trace: dict = {"conflict": conflict, "input": text, "rounds": []}

    det = contract.validate_detect(detect(text, conflict))
    trace["detect"] = det
    if det["status"] == "unbiased":
        result = {"decision": "unchanged", "final": text, "changes": []}
    elif det["status"] == "uncertain":
        result = {"decision": "escalate", "final": text, "changes": [], "reason": "detector uncertain"}
    else:
        spans = contract.validate_explain(explain(text, conflict))
        query = " ".join(spans) if spans else text
        rules = contract.validate_retrieve(retrieve(query, config.RAG_K), config.RAG_K)
        trace.update(spans=spans, rules=[r.splitlines()[0] for r in rules])

        feedback = None
        result = None
        for round_no in range(1, config.MAX_ROUNDS + 1):
            rw = contract.validate_rewrite(rewrite(text, spans, rules, feedback))
            fact = fact_check(text, rw["revised"])
            reviews = {
                p: contract.validate_review(review(text, rw["revised"], p))
                for p in config.PERSPECTIVES[conflict]
            }
            decision = evaluate(fact, reviews)
            trace["rounds"].append({"round": round_no, "rewrite": rw, "fact_check": fact,
                                    "reviews": reviews, "decision": decision})
            if decision in ("approve", "escalate"):
                result = {"decision": decision, "final": rw["revised"] if decision == "approve" else text,
                          "changes": rw["changes"] if decision == "approve" else []}
                break
            feedback = _feedback(fact, reviews)
        if result is None:
            result = {"decision": "block", "final": text, "changes": [],
                      "reason": f"not approved after {config.MAX_ROUNDS} rounds"}

    result["change_note"] = change_note(result["changes"])
    trace["result"] = result
    if log:
        utils.audit_log(trace)
    return result


def change_note(changes: list[dict]) -> str:
    if not changes:
        return "No changes were made."
    lines = ["Changes made to this answer:"]
    lines += [f"- \"{c['original']}\" -> \"{c['revised']}\": {c['reason']}" for c in changes]
    return "\n".join(lines)
