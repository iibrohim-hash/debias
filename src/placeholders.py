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


def _build(conflict: str) -> tuple[re.Pattern, list[tuple[re.Pattern, str]]]:
    entries: list[tuple[str, str]] = [
        (term, side) for side, terms in TERMS[conflict].items() for term in terms
    ]
    entries.sort(key=lambda e: len(e[0]), reverse=True)  # longest match first
    alternation = "|".join(_term_pattern(t) for t, _ in entries)
    # Boundaries: not inside a word, but "_" counts as a boundary so hashtags such
    # as "#Gaza_War" are masked too ("#[SIDE_B]_War").
    pattern = re.compile(rf"(?<![^\W_])(?:{alternation})(?![^\W_])")
    # Per-term patterns to find the side of a match. Matching (not str.lower()) is
    # needed because case-insensitive regex also matches Unicode variants such as
    # the dotless "ı" in "ısrael", which str.lower() does not map back.
    sides = [(re.compile(_term_pattern(t)), side) for t, side in entries]
    return pattern, sides


_COMPILED = {c: _build(c) for c in TERMS}


def _build_hashtag(conflict: str) -> re.Pattern:
    """Side names glued inside hashtags ("#IsraelHamasWar"), no word boundaries.

    Single-word terms only; acronyms shorter than 3 letters ("PA") are skipped
    because they match inside unrelated hashtags.
    """
    terms = [t for terms in TERMS[conflict].values() for t in terms
             if " " not in t and not (t.isupper() and len(t) < 3)]
    terms.sort(key=len, reverse=True)
    return re.compile("|".join(_term_pattern(t) for t in terms))


_HASHTAG_TERMS = {c: _build_hashtag(c) for c in TERMS}
_HASHTAG = re.compile(r"#[^\W_][\w]*")


def _side_of(word: str, sides: list[tuple[re.Pattern, str]]) -> str:
    return next(s for p, s in sides if p.fullmatch(word))


def mask_sides(text: str, conflict: str = config.DEFAULT_CONFLICT) -> str:
    """Replace every side name in `text` with [SIDE_A] or [SIDE_B].

    Also inside hashtags: "#IsraelHamasWar" -> "#[SIDE_A][SIDE_B]War",
    "#HamasTerrorists" -> "#[SIDE_B]Terrorists" (the framing word stays).
    """
    config.check_conflict(conflict)
    pattern, sides = _COMPILED[conflict]
    inner = _HASHTAG_TERMS[conflict]

    def repl_word(match: re.Match) -> str:
        return _PLACEHOLDER[_side_of(match.group(0), sides)]

    def repl_hashtag(match: re.Match) -> str:
        return inner.sub(repl_word, match.group(0))

    text = _HASHTAG.sub(repl_hashtag, text)
    return pattern.sub(repl_word, text)


def find_leftovers(text: str, conflict: str | None = None) -> list[str]:
    """Side names still present in `text` (run after masking).

    conflict=None checks every conflict, which also catches, for example,
    Russia/Ukraine names inside Israel-Palestine data. Pass a conflict to check
    only that conflict's names (these should be zero after masking).
    """
    conflicts = [config.check_conflict(conflict)] if conflict else list(_COMPILED)
    found: list[str] = []
    for c in conflicts:
        found.extend(m.group(0) for m in _COMPILED[c][0].finditer(text))
    return found


def find_glued(text: str, conflict: str = config.DEFAULT_CONFLICT) -> list[str]:
    """Side names glued inside longer tokens, e.g. "#IsraelAtWar", "GazaUnderAttack".

    Word-boundary masking cannot catch these; use this to measure how often they occur.
    """
    config.check_conflict(conflict)
    found = []
    for token in re.findall(r"\S+", text):
        for side_terms in TERMS[conflict].values():
            for term in side_terms:
                if " " in term or term.isupper() or len(term) < 4:
                    continue
                if term.lower() in token.lower() and not re.fullmatch(rf"\W*{re.escape(term)}\W*", token, re.I):
                    found.append(token)
                    break
    return sorted(set(found))
