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
- The model lists every wording problem and which side each one HURTS (v2).
  The code keeps only the problems that hurt the checked side: fair = none left.
  (v1 asked the model to report only its own side; it often reported problems
  that hurt the other side instead.) Problems for the other side are kept in
  info["other_side_issues"] for inspection.
- Output is parsed and validated: schema, "hurts" is one of the two side names,
  and every cited phrase must really be in the revised text. On failure the model
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

from . import config, rewriter, utils

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

# Short names the model uses in "hurts".
SIDE_NAMES = {p: f"{p.capitalize()} side" for p in SIDES}

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
    """Per guidebook rule: "R02 Loaded labels ...: <first sentence of Rule:>" and, on
    the next line, the guidebook's "Neutral:" example as acceptable wording (v3: on
    Llama, v2 flagged the guidebook's own neutral wordings, e.g. "military campaign")."""
    text = config.GUIDEBOOK_PATH.read_text(encoding="utf-8")
    lines = []
    for block in re.split(r"(?m)^## ", text)[1:]:
        head, _, body = block.partition("\n")
        m = re.search(r"(?m)^Rule:\s*(.+)$", body)
        if not re.match(r"R\d+", head) or not m:
            continue
        first = re.split(r"(?<=[.!?])\s", m.group(1).strip(), maxsplit=1)[0]
        lines.append(f"{head.strip()}: {first}")
        ok = re.search(r"(?m)^Neutral:\s*(.+)$", body)
        if ok:
            lines.append(f"   Acceptable wording: {ok.group(1).strip()}")
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
    other = config.OTHER_PERSPECTIVE[perspective]
    user = Template(_prompt("reviewer_user.txt")).substitute(
        side=SIDES[perspective],
        other_side=SIDES[other],
        side_name=SIDE_NAMES[perspective],
        other_name=SIDE_NAMES[other],
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


def hurts_side(hurts: str, perspective: str) -> str | None:
    """Map the model's "hurts" text to a perspective of this conflict (None if unclear)."""
    h = _norm(hurts).replace(" side", "").strip(" \"'.")
    for p in (perspective, config.OTHER_PERSPECTIVE[perspective]):
        if h == p:
            return p
    return None


def validate_output(revised: str, out: dict, perspective: str) -> list[str]:
    """Problems with a parsed reviewer output (empty list = valid).

    Expected: {"issues": [{"phrase": str, "hurts": str, "problem": str}]}.
    """
    if not isinstance(out, dict) or not isinstance(out.get("issues"), list):
        return ['wrong schema: expected {"issues": [...]}']
    names = (SIDE_NAMES[perspective], SIDE_NAMES[config.OTHER_PERSPECTIVE[perspective]])
    errors = []
    for i, issue in enumerate(out["issues"], 1):
        if not isinstance(issue, dict) or not all(isinstance(issue.get(k), str) for k in ("phrase", "hurts", "problem")):
            errors.append(f"issue {i}: needs string keys phrase, hurts, problem")
            continue
        if not phrase_in(issue["phrase"], revised):
            errors.append(f'issue {i}: phrase "{issue["phrase"]}" is not in the REVISED text')
        if hurts_side(issue["hurts"], perspective) is None:
            errors.append(f'issue {i}: hurts must be "{names[0]}" or "{names[1]}", not "{issue["hurts"]}"')
        if not issue["problem"].strip():
            errors.append(f"issue {i}: problem is empty")
    return errors


def _split(out: dict, perspective: str) -> tuple[dict, list[dict]]:
    """Contract result for the checked side, plus the issues that hurt the other side."""
    own, other = [], []
    for i in out["issues"]:
        item = {"phrase": i["phrase"], "problem": i["problem"]}
        (own if hurts_side(i["hurts"], perspective) == perspective else other).append(item)
    return {"fair": not own, "issues": own}, other


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
    "fell_back", "attempts": [{"attempt", "raw", "errors"}], "other_side_issues"}.
    `generate_fn(batch_of_messages) -> list[str | dict]` replaces the model in tests.
    """
    for _, _, p in items:
        _check_perspective(p)
    if config.USE_FAKES and generate_fn is None:
        return [(_fake_review(*it), {"fake": True, "model_id": None, "retries": 0, "fell_back": False,
                                     "attempts": [], "other_side_issues": []}) for it in items]

    gen = generate_fn or _default_generate
    msgs = [build_messages(*it) for it in items]
    infos = [{"fake": False, "model_id": None, "retries": 0, "fell_back": False, "attempts": [],
              "other_side_issues": []} for _ in items]
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
                    errors = validate_output(items[i][1], out, items[i][2])
                except ValueError as e:
                    out, errors = None, [str(e)]
                infos[i]["attempts"].append({"attempt": attempt, "raw": raw, "errors": errors})
                if not errors:
                    results[i], infos[i]["other_side_issues"] = _split(out, items[i][2])
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
