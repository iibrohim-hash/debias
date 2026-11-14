"""Guidebook retrieval (Person D, Yasa).

Interface (do not change without telling the team):
    retrieve(query, k=3) -> list[str]   # guidebook entries as text; R01 always included

Current state: simple stand-in that parses data/guidebook.md and returns R01 plus
the first k-1 other entries. Replace the body of retrieve() with the
sentence-transformers + FAISS version (config.EMBEDDING_MODEL, IndexFlatIP on
normalized vectors). Keep load_entries() so everyone splits the guidebook the
same way.
"""
from __future__ import annotations

import re
from pathlib import Path

from . import config


def load_entries(path: Path = config.GUIDEBOOK_PATH) -> list[str]:
    """Split the guidebook at '## ' headings. Each entry keeps its heading line."""
    text = Path(path).read_text(encoding="utf-8")
    parts = re.split(r"(?m)^(?=## )", text)
    return [p.strip() for p in parts if p.startswith("## ")]


def entry_id(entry: str) -> str:
    """'## R07 Hidden agency' -> 'R07'."""
    return entry.split()[1]


def retrieve(query: str, k: int = config.RAG_K) -> list[str]:
    entries = load_entries()
    always = [e for e in entries if entry_id(e) == config.ALWAYS_INCLUDED_RULE]
    others = [e for e in entries if entry_id(e) != config.ALWAYS_INCLUDED_RULE]
    # TODO(Yasa): rank `others` by embedding similarity to `query`.
    return always + others[: max(k - len(always), 0)]
