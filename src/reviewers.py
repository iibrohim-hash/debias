"""Perspective reviewers (Person B prompts, Madina).

Interface (do not change without telling the team):
    review(original, revised, perspective)
        -> {"fair": bool, "issues": [{"phrase": str, "problem": str}]}
    perspective: "palestinian" | "israeli" | "ukrainian" | "russian"

One prompt template with the perspective as a parameter. Reviewers cite exact
phrases and check against the rubric; never ask the model to role-play a
nationality.

Current state: fake stand-in that always says fair.
"""
from __future__ import annotations

from . import config


def _fake_review(original: str, revised: str, perspective: str) -> dict:
    return {"fair": True, "issues": []}


def review(original: str, revised: str, perspective: str) -> dict:
    if perspective not in config.ALL_PERSPECTIVES:
        raise ValueError(f"perspective must be one of {config.ALL_PERSPECTIVES}")
    if config.USE_FAKES:
        return _fake_review(original, revised, perspective)
    # TODO(Madina): real Llama version goes here.
    return _fake_review(original, revised, perspective)
