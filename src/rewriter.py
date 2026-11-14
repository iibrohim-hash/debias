"""Llama rewriter (Person B, Madina).

Interface (do not change without telling the team):
    rewrite(text, spans, rules, feedback=None)
        -> {"revised": str, "changes": [{"original", "revised", "reason"}]}

Current state: fake stand-in that returns the text unchanged. Replace with
Llama 3.1 8B Instruct in 4-bit (config.REWRITER_MODEL, fallback
config.REWRITER_FALLBACK_MODEL), system prompt from prompts/, greedy decoding,
max_new_tokens=config.MAX_NEW_TOKENS, one retry on broken JSON.
"""
from __future__ import annotations

from . import config


def _fake_rewrite(text: str, spans: list[str], rules: list[str], feedback: str | None) -> dict:
    return {"revised": text, "changes": []}


def rewrite(text: str, spans: list[str], rules: list[str], feedback: str | None = None) -> dict:
    if config.USE_FAKES:
        return _fake_rewrite(text, spans, rules, feedback)
    # TODO(Madina): real Llama version goes here.
    return _fake_rewrite(text, spans, rules, feedback)
