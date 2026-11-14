"""Replace side names with [SIDE_A] / [SIDE_B] (Person A, tasks 5 and 10).

The detector must learn how bias is expressed, not which names appear. The same
function is used in training, in detect() at run time, and for Russia-Ukraine.

The term lists are a first draft (v0). A teammate must review them before the
final training run, and every change must be reported, because the lists define
what the detector can and cannot see.

Deliberately NOT replaced, because they are framing choices the detector should
see: loaded labels such as "Zionist entity", "terrorists", "Nazis", "regime".
Contested places (e.g. Jerusalem, Crimea, Donbas) are left as they are for now;
decide as a team.
"""
from __future__ import annotations

import re

from . import config

# Terms written in ALL CAPS are matched case-sensitively (acronyms such as "PA"),
# everything else case-insensitively. Plurals and adjective forms are listed
# explicitly; possessives ("Israel's") are handled automatically.
TERMS: dict[str, dict[str, list[str]]] = {
    "israel_palestine": {
        "SIDE_A": [
            "Israel", "Israeli", "Israelis",
            "Israel Defense Forces", "Israel Defence Forces", "IDF",
            "Israeli army", "Israeli military", "Israeli government",
            "Netanyahu", "Benjamin Netanyahu", "Knesset", "Tel Aviv",
            "Mossad", "Shin Bet",
        ],
        "SIDE_B": [
            "Palestine", "Palestinian", "Palestinians",
            "Palestinian Authority", "PA", "PLO",
            "Hamas", "Fatah", "Islamic Jihad", "Palestinian Islamic Jihad",
            "Qassam Brigades", "al-Qassam Brigades",
            "Gaza", "Gazan", "Gazans", "Gaza Strip", "West Bank",
            "Ramallah", "Abbas", "Mahmoud Abbas", "Sinwar", "Yahya Sinwar",
        ],
    },
    "russia_ukraine": {
        "SIDE_A": [
            "Russia", "Russian", "Russians", "Russian Federation",
            "Kremlin", "Moscow", "Putin", "Vladimir Putin",
            "Russian army", "Russian military", "Russian forces",
            "Wagner", "Wagner Group", "Shoigu", "Lavrov",
        ],
        "SIDE_B": [
            "Ukraine", "Ukrainian", "Ukrainians",
            "Kyiv", "Kiev", "Zelensky", "Zelenskyy", "Volodymyr Zelensky",
            "Volodymyr Zelenskyy", "Ukrainian army", "Ukrainian military",
            "Ukrainian forces", "Armed Forces of Ukraine", "AFU",
        ],
    },
}

_PLACEHOLDER = {"SIDE_A": config.PLACEHOLDER_A, "SIDE_B": config.PLACEHOLDER_B}


def _term_pattern(term: str) -> str:
    escaped = re.escape(term).replace(r"\ ", r"\s+")
    return escaped if term.isupper() else f"(?i:{escaped})"


def _build(conflict: str) -> tuple[re.Pattern, dict[str, str]]:
    lookup: dict[str, str] = {}
    entries: list[tuple[str, str]] = []
    for side, terms in TERMS[conflict].items():
        for term in terms:
            entries.append((term, side))
            lookup[term.lower()] = side
    entries.sort(key=lambda e: len(e[0]), reverse=True)  # longest match first
    alternation = "|".join(_term_pattern(t) for t, _ in entries)
    pattern = re.compile(rf"(?<![\w\[])(?:{alternation})(?![\w\]])")
    return pattern, lookup


_COMPILED = {c: _build(c) for c in TERMS}


def mask_sides(text: str, conflict: str = config.DEFAULT_CONFLICT) -> str:
    """Replace every side name in `text` with [SIDE_A] or [SIDE_B]."""
    config.check_conflict(conflict)
    pattern, lookup = _COMPILED[conflict]

    def repl(match: re.Match) -> str:
        key = re.sub(r"\s+", " ", match.group(0)).lower()
        return _PLACEHOLDER[lookup[key]]

    return pattern.sub(repl, text)


def find_leftovers(text: str) -> list[str]:
    """Side names from ANY conflict still present in `text` (run after masking).

    Checking all conflicts catches, for example, Russia/Ukraine names that leak
    into Israel-Palestine data and vice versa.
    """
    found: list[str] = []
    for pattern, _ in _COMPILED.values():
        found.extend(m.group(0) for m in pattern.finditer(text))
    return found
