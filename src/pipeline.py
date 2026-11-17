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
import time

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
    """Final answer + change note for one chatbot answer."""
    return run_detailed(text, conflict, log=log)[0]


def run_detailed(text: str, conflict: str = config.DEFAULT_CONFLICT, log: bool = True,
                 force: bool = False) -> tuple[dict, dict]:
    """Like run(), but also returns the full trace of every step with timings.

    force=True sends the text through explain/RAG/rewrite even when the detector
    says "unbiased" or "uncertain" (the real status is still recorded). For demos
    and debugging only; never for evaluation runs.
    """
    config.check_conflict(conflict)
    trace: dict = {"conflict": conflict, "input": text, "forced": force, "rounds": [], "seconds": {}}
    t = trace["seconds"]

    def timed(name, fn, *args):
        start = time.perf_counter()
        out = fn(*args)
        t[name] = round(t.get(name, 0.0) + time.perf_counter() - start, 2)
        return out

    det = contract.validate_detect(timed("detect", detect, text, conflict))
    trace["detect"] = det
    if det["status"] == "unbiased" and not force:
        result = {"decision": "unchanged", "final": text, "changes": []}
    elif det["status"] == "uncertain" and not force:
        result = {"decision": "escalate", "final": text, "changes": [], "reason": "detector uncertain"}
    else:
        spans = contract.validate_explain(timed("explain", explain, text, conflict))
        query = " ".join(spans) if spans else text
        rules = contract.validate_retrieve(timed("retrieve", retrieve, query, config.RAG_K), config.RAG_K)
        trace.update(spans=spans, query=query, rules=[r.splitlines()[0] for r in rules])

        feedback = None
        result = None
        for round_no in range(1, config.MAX_ROUNDS + 1):
            rw = contract.validate_rewrite(timed("rewrite", rewrite, text, spans, rules, feedback))
            fact = fact_check(text, rw["revised"])
            reviews = {
                p: contract.validate_review(timed("review", review, text, rw["revised"], p))
                for p in config.PERSPECTIVES[conflict]
            }
            decision = evaluate(fact, reviews)
            trace["rounds"].append({"round": round_no, "rewrite": rw, "fact_check": fact,
                                    "reviews": reviews, "decision": decision, "feedback_in": feedback})
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
    t["total"] = round(sum(v for k, v in t.items() if k != "total"), 2)
    if log:
        utils.audit_log(trace)
    return result, trace


def change_note(changes: list[dict]) -> str:
    if not changes:
        return "No changes were made."
    lines = ["Changes made to this answer:"]
    lines += [f"- \"{c['original']}\" -> \"{c['revised']}\": {c['reason']}" for c in changes]
    return "\n".join(lines)
