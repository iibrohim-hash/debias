"""End-to-end pipeline (integration).

This is a plain sequential version of section 3 of the plan, so the team can run
10 examples end to end in Week 10 using whatever pieces exist (real or fake).
In Week 11 the same steps move into CrewAI agents; keep the decision rules here
as the reference.

    from src.pipeline import run
    result = run("Some chatbot answer ...", conflict="israel_palestine")
"""
from __future__ import annotations

import time

from . import config, contract, utils
from .detector import detect, explain
from .rag import retrieve
from .reviewers import review
from .rewriter import find_new_facts, rewrite


def fact_check(original: str, revised: str) -> dict:
    """Cheap checks: no new numbers, no new capitalised names.

    Uses the rewriter's own check (rewriter.find_new_facts), so both agree: a word
    capitalised only because it now starts a sentence ("The", "According") is not
    a new name. Words that were REMOVED are not caught here; the reviewers see the
    original and the rewrite side by side.

    TODO: add the NLI check (config.NLI_MODEL): original must entail revised.
    """
    new = find_new_facts(original, revised)
    return {"ok": not new["new_numbers"] and not new["new_names"], **new}


def evaluate(fact: dict, reviews: dict[str, dict], rule: str | None = None,
             last_round: bool = False) -> str:
    """approve | retry | escalate | block for one round.

    rule (config.EVALUATOR_RULES, default config.EVALUATOR_RULE):
      "disagree_escalates": reviewers disagree -> a human at once.
      "retry_first":        reviewers disagree -> retry with feedback; a human only if they
                            still disagree in the last round.
    Both: approve needs both reviewers fair and the fact check ok; a round that is not
    approved and not escalated is retried, or blocked (original kept) in the last round.
    No reviews (the rewriter returned the text unchanged) is never approved.
    """
    rule = rule or config.EVALUATOR_RULE
    if rule not in config.EVALUATOR_RULES:
        raise ValueError(f"unknown evaluator rule {rule!r}; choose from {config.EVALUATOR_RULES}")
    fair = [r["fair"] for r in reviews.values()]
    if fair and all(fair) and fact["ok"]:
        return "approve"
    if len(set(fair)) > 1 and (rule == "disagree_escalates" or last_round):
        return "escalate"  # reviewers disagree -> human
    return "block" if last_round else "retry"


UNCHANGED_FEEDBACK = ("Your last answer left the text unchanged. Rewrite the flagged words "
                      "using the rules, and list every change.")


def _feedback(fact: dict, reviews: dict[str, dict], unchanged: bool = False) -> str:
    lines = [UNCHANGED_FEEDBACK] if unchanged else []
    if fact["new_numbers"]:
        lines.append(f"Do not add numbers: {fact['new_numbers']}")
    if fact["new_names"]:
        lines.append(f"Do not add names or actors: {fact['new_names']}")
    for persp, r in reviews.items():
        for issue in r["issues"]:
            lines.append(f"[{persp}] \"{issue['phrase']}\": {issue['problem']}")
    return "\n".join(lines)


def run(text: str, conflict: str = config.DEFAULT_CONFLICT, log: bool = True,
        rule: str | None = None) -> dict:
    """Final answer + change note for one chatbot answer."""
    return run_detailed(text, conflict, log=log, rule=rule)[0]


def run_detailed(text: str, conflict: str = config.DEFAULT_CONFLICT, log: bool = True,
                 force: bool = False, rule: str | None = None) -> tuple[dict, dict]:
    """Like run(), but also returns the full trace of every step with timings.

    force=True sends the text through explain/RAG/rewrite even when the detector
    says "unbiased" or "uncertain" (the real status is still recorded). For demos
    and debugging only; never for evaluation runs.
    rule: the evaluator rule (see evaluate); default config.EVALUATOR_RULE.
    """
    config.check_conflict(conflict)
    rule = rule or config.EVALUATOR_RULE
    if rule not in config.EVALUATOR_RULES:
        raise ValueError(f"unknown evaluator rule {rule!r}; choose from {config.EVALUATOR_RULES}")
    trace: dict = {"conflict": conflict, "input": text, "forced": force, "evaluator_rule": rule,
                   "rounds": [], "seconds": {}}
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
        for round_no in range(1, config.MAX_ROUNDS + 1):
            last_round = round_no == config.MAX_ROUNDS
            rw = contract.validate_rewrite(timed("rewrite", rewrite, text, spans, rules, feedback))
            fact = fact_check(text, rw["revised"])
            # Unchanged text (the rewriter gave up or changed nothing): nothing for the reviewers to judge.
            unchanged = rw["revised"].strip() == text.strip()
            reviews = {} if unchanged else {
                p: contract.validate_review(timed("review", review, text, rw["revised"], p))
                for p in config.PERSPECTIVES[conflict]
            }
            decision = evaluate(fact, reviews, rule, last_round)
            trace["rounds"].append({"round": round_no, "rewrite": rw, "fact_check": fact, "unchanged": unchanged,
                                    "reviews": reviews, "decision": decision, "feedback_in": feedback})
            if decision != "retry":
                result = {"decision": decision, "final": rw["revised"] if decision == "approve" else text,
                          "changes": rw["changes"] if decision == "approve" else []}
                if decision == "block":
                    result["reason"] = f"not approved after {config.MAX_ROUNDS} rounds"
                break
            feedback = _feedback(fact, reviews, unchanged)

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
