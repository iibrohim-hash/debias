"""Checks for the shared interface (project plan, section 5).

Each validator raises ContractError if a function returns the wrong shape. The
tests and the pipeline both call these, so a broken interface fails loudly
instead of breaking someone else's code later.

Do not change these without telling the team.
"""
from __future__ import annotations

from . import config


class ContractError(ValueError):
    pass


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ContractError(msg)


def _is_str_list(x) -> bool:
    return isinstance(x, list) and all(isinstance(i, str) for i in x)


def validate_detect(out) -> dict:
    # {"status": "unbiased"|"uncertain"|"biased", "score": float, "label": str}
    _require(isinstance(out, dict), "detect() must return a dict")
    _require(set(out) >= {"status", "score", "label"}, "detect() needs keys status, score, label")
    _require(out["status"] in config.DETECT_STATUSES, f"status must be one of {config.DETECT_STATUSES}")
    _require(isinstance(out["score"], float) and 0.0 <= out["score"] <= 1.0, "score must be a float in [0, 1]")
    _require(out["label"] in config.LABELS, f"label must be one of {config.LABELS}")
    return out


def validate_explain(out) -> list:
    _require(_is_str_list(out), "explain() must return a list of strings")
    return out


def validate_retrieve(out, k: int) -> list:
    _require(_is_str_list(out), "retrieve() must return a list of strings")
    _require(len(out) >= 1, "retrieve() returned nothing")
    _require(
        any(entry.lstrip("# ").startswith(config.ALWAYS_INCLUDED_RULE) for entry in out),
        f"retrieve() must always include {config.ALWAYS_INCLUDED_RULE}",
    )
    _require(len(out) <= k + 1, "retrieve() returned more than k entries (+1 for R01)")
    return out


def validate_rewrite(out) -> dict:
    # {"revised": str, "changes": [{"original": str, "revised": str, "reason": str}]}
    _require(isinstance(out, dict), "rewrite() must return a dict")
    _require(isinstance(out.get("revised"), str), "rewrite() needs a 'revised' string")
    _require(isinstance(out.get("changes"), list), "rewrite() needs a 'changes' list")
    for c in out["changes"]:
        _require(
            isinstance(c, dict) and all(isinstance(c.get(k), str) for k in ("original", "revised", "reason")),
            "each change needs string keys original, revised, reason",
        )
    return out


def validate_review(out) -> dict:
    # {"fair": bool, "issues": [{"phrase": str, "problem": str}]}
    _require(isinstance(out, dict), "review() must return a dict")
    _require(isinstance(out.get("fair"), bool), "review() needs a boolean 'fair'")
    _require(isinstance(out.get("issues"), list), "review() needs an 'issues' list")
    for i in out["issues"]:
        _require(
            isinstance(i, dict) and all(isinstance(i.get(k), str) for k in ("phrase", "problem")),
            "each issue needs string keys phrase, problem",
        )
    return out
