"""Perspective reviewers A and B (Person B prompts, Madina).

Interface (do not change without telling the team):
    review(original, revised, perspective)
        -> {"fair": bool, "issues": [{"phrase": str, "problem": str}]}
    perspective: "palestinian" | "israeli" | "ukrainian" | "russian"

How it works:
- Reviewer A and Reviewer B are the same prompt (prompts/reviewer_system.txt,
  prompts/reviewer_user.txt) with only the checked side changed, so they are
  symmetric by construction. The model checks wording against a rubric; it is
  never asked to role-play a nationality.
- The rubric is built from data/guidebook.md (rule id, title, first sentence of
  "Rule:"), so it follows Amen's guidebook automatically.
- Uses the rewriter's model (rewriter.get_model), so Llama is loaded only once.
- Output is parsed and validated: schema, "fair" consistent with "issues", and
  every cited phrase must really be in the revised text. On failure the model
  gets ONE retry with the errors; if that fails too, the reviewer answers
  "not fair" (unreadable output must never approve a rewrite) and the failure
  goes to the audit log.
- DEBIAS_FAKE=1 keeps the fake stand-in (always fair); tests and CI use it.
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from string import Template
from typing import Callable

from . import config, contract, rewriter, utils

PROMPT_FILES = ("reviewer_system.txt", "reviewer_user.txt", "reviewer_retry.txt")

# What "the checked side" means for each perspective. Keep the descriptions parallel.
SIDES = {
    "palestinian": "Palestinians and Palestinian groups and institutions (for example Hamas, "
                   "the Palestinian Authority, people in Gaza and the West Bank)",
    "israeli": "Israelis and Israeli groups and institutions (for example the Israeli government, "
               "the Israeli army, Israeli settlers, people in Israel)",
    "ukrainian": "Ukrainians and Ukrainian groups and institutions (for example the Ukrainian government, "
                 "the Ukrainian army, people in Ukraine)",
    "russian": "Russians and Russian groups and institutions (for example the Russian government, "
               "the Russian army, people in Russia)",
}

UNREADABLE = "reviewer output could not be validated"


# ---------------------------------------------------------------------------
# Fake stand-in (DEBIAS_FAKE=1)
# ---------------------------------------------------------------------------
def _fake_review(original: str, revised: str, perspective: str) -> dict:
    return {"fair": True, "issues": []}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
def _prompt(name: str) -> str:
    return (config.PROMPTS_DIR / name).read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def rubric() -> str:
    """One line per guidebook rule: "R02 Loaded labels ...: <first sentence of Rule:>"."""
    text = config.GUIDEBOOK_PATH.read_text(encoding="utf-8")
    lines = []
    for block in re.split(r"(?m)^## ", text)[1:]:
        head, _, body = block.partition("\n")
        m = re.search(r"(?m)^Rule:\s*(.+)$", body)
        if not re.match(r"R\d+", head) or not m:
            continue
        first = re.split(r"(?<=[.!?])\s", m.group(1).strip(), maxsplit=1)[0]
        lines.append(f"{head.strip()}: {first}")
    return "\n".join(lines)


def prompt_version() -> str:
    """Short hash of everything that shapes the reviewer prompt (files + rubric)."""
    blob = "\n".join(_prompt(n) for n in PROMPT_FILES) + rubric()
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def _check_perspective(perspective: str) -> None:
    if perspective not in config.ALL_PERSPECTIVES:
        raise ValueError(f"perspective must be one of {config.ALL_PERSPECTIVES}")


def build_messages(original: str, revised: str, perspective: str) -> list[dict]:
    _check_perspective(perspective)
    user = Template(_prompt("reviewer_user.txt")).substitute(
        side=SIDES[perspective],
        other_side=SIDES[config.OTHER_PERSPECTIVE[perspective]],
        rubric=rubric(),
        original=original,
        revised=revised,
    )
    return [
        {"role": "system", "content": _prompt("reviewer_system.txt").strip()},
        {"role": "user", "content": user.strip()},
    ]


def _retry_message(errors: list[str]) -> dict:
    content = Template(_prompt("reviewer_retry.txt")).substitute(errors="\n".join(f"- {e}" for e in errors))
    return {"role": "user", "content": content.strip()}


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("’", "'")).strip().lower()


def phrase_in(phrase: str, text: str) -> bool:
    """Case- and whitespace-insensitive substring check (quotes stripped)."""
    p = _norm(phrase.strip(" \"'“”."))
    return bool(p) and p in _norm(text)


def validate_output(revised: str, out: dict) -> list[str]:
    """Problems with a parsed reviewer output (empty list = valid)."""
    try:
        contract.validate_review(out)
    except contract.ContractError as e:
        return [f"wrong schema: {e}"]
    errors = []
    if out["fair"] and out["issues"]:
        errors.append('"fair" is true but "issues" is not empty')
    if not out["fair"] and not out["issues"]:
        errors.append('"fair" is false but no issue is listed')
    for i, issue in enumerate(out["issues"], 1):
        if not phrase_in(issue["phrase"], revised):
            errors.append(f'issue {i}: phrase "{issue["phrase"]}" is not in the REVISED text')
        if not issue["problem"].strip():
            errors.append(f"issue {i}: problem is empty")
    return errors


def _clean(out: dict) -> dict:
    issues = [{"phrase": i["phrase"], "problem": i["problem"]} for i in out["issues"]]
    return {"fair": out["fair"], "issues": issues}


# ---------------------------------------------------------------------------
# review()
# ---------------------------------------------------------------------------
GenerateFn = Callable[[list[list[dict]]], list]


def _default_generate(batch: list[list[dict]]) -> list[dict]:
    return rewriter.generate_detailed(batch)


def review_many(
    items: list[tuple[str, str, str]],
    generate_fn: GenerateFn | None = None,
    batch_size: int = 8,
) -> list[tuple[dict, dict]]:
    """Review many (original, revised, perspective) items, batched on the GPU.

    Returns one (result, info) per item; info = {"fake", "model_id", "retries",
    "fell_back", "attempts": [{"attempt", "raw", "errors"}]}.
    `generate_fn(batch_of_messages) -> list[str | dict]` replaces the model in tests.
    """
    for _, _, p in items:
        _check_perspective(p)
    if config.USE_FAKES and generate_fn is None:
        return [(_fake_review(*it), {"fake": True, "model_id": None, "retries": 0, "fell_back": False,
                                     "attempts": []}) for it in items]

    gen = generate_fn or _default_generate
    msgs = [build_messages(*it) for it in items]
    infos = [{"fake": False, "model_id": None, "retries": 0, "fell_back": False, "attempts": []} for _ in items]
    results: list[dict | None] = [None] * len(items)

    for attempt in (1, 2):
        todo = [i for i in range(len(items)) if results[i] is None]
        for start in range(0, len(todo), batch_size):
            chunk = todo[start:start + batch_size]
            outs = gen([msgs[i] for i in chunk])
            for i, g in zip(chunk, outs):
                raw = g if isinstance(g, str) else g["text"]
                try:
                    out = rewriter.parse_output(raw)
                    errors = validate_output(items[i][1], out)
                except ValueError as e:
                    out, errors = None, [str(e)]
                infos[i]["attempts"].append({"attempt": attempt, "raw": raw, "errors": errors})
                if not errors:
                    results[i] = _clean(out)
                elif attempt == 1:
                    infos[i]["retries"] = 1
                    msgs[i] = msgs[i] + [{"role": "assistant", "content": raw}, _retry_message(errors)]

    model_id = rewriter.loaded_model_id() if generate_fn is None else None
    for i, (original, revised, perspective) in enumerate(items):
        infos[i]["model_id"] = model_id
        if results[i] is None:
            infos[i]["fell_back"] = True
            results[i] = {"fair": False, "issues": [{"phrase": "", "problem": UNREADABLE}]}
            utils.audit_log({"event": "review_failed", "model_id": model_id, "prompt_version": prompt_version(),
                             "perspective": perspective, "original": original, "revised": revised,
                             "attempts": infos[i]["attempts"]})
    return list(zip(results, infos))


def review_detailed(original: str, revised: str, perspective: str,
                    generate_fn: GenerateFn | None = None) -> tuple[dict, dict]:
    return review_many([(original, revised, perspective)], generate_fn)[0]


def review(original: str, revised: str, perspective: str) -> dict:
    _check_perspective(perspective)
    return review_detailed(original, revised, perspective)[0]
