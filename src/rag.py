"""Guidebook retrieval (Person D, Yasa).

Interface (do not change without telling the team):
    retrieve(query, k=3) -> list[str]   # guidebook entries as text; R01 always included

Ranks guidebook entries by cosine similarity to the query (config.EMBEDDING_MODEL,
FAISS IndexFlatIP on normalized vectors) and returns R01 first, then the top k other
entries (at most k + 1 in total). The index is rebuilt automatically when
data/guidebook.md changes. When DEBIAS_FAKE=1 or sentence-transformers / faiss are
not installed (CI installs only pytest), a word-overlap ranking is used instead.
"""
from __future__ import annotations

import hashlib
import re
import warnings
from functools import lru_cache
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


# ---------------------------------------------------------------------------
# Fallback: word overlap (no ML libraries needed)
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it",
    "of", "on", "or", "that", "the", "this", "to", "was", "were", "with",
}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if w not in _STOPWORDS}


def _rank_overlap(query: str, entries: list[str]) -> list[int]:
    q = _words(query)
    scores = [len(q & _words(e)) for e in entries]
    return sorted(range(len(entries)), key=lambda i: -scores[i])  # stable: ties keep guidebook order


# ---------------------------------------------------------------------------
# Real implementation: sentence-transformers + FAISS (loads lazily)
# ---------------------------------------------------------------------------
_index: dict = {}  # {"hash": str, "entries": list[str], "index": faiss.Index}


@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


def _build_index(entries: list[str]):
    import faiss

    vectors = _load_model().encode(entries, normalize_embeddings=True, convert_to_numpy=True)
    index = faiss.IndexFlatIP(vectors.shape[1])  # inner product = cosine on normalized vectors
    index.add(vectors.astype("float32"))
    return index


def _get_index(path: Path) -> tuple[list[str], object]:
    """Return (entries, index), rebuilding when the guidebook file has changed."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if _index.get("hash") != digest:
        entries = load_entries(path)
        _index.update(hash=digest, entries=entries, index=_build_index(entries))
    return _index["entries"], _index["index"]


def rebuild_index() -> None:
    """Force a rebuild on the next retrieve() call (e.g. in a notebook after editing)."""
    _index.clear()


def _rank_faiss(query: str, path: Path) -> tuple[list[str], list[int]]:
    entries, index = _get_index(path)
    q = _load_model().encode([query], normalize_embeddings=True, convert_to_numpy=True)
    _, ids = index.search(q.astype("float32"), len(entries))
    return entries, [int(i) for i in ids[0] if i >= 0]


@lru_cache(maxsize=1)
def _ml_available() -> bool:
    try:
        import faiss  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError:
        warnings.warn("sentence-transformers or faiss not installed; retrieve() uses word overlap")
        return False
    return True


def _use_fake() -> bool:
    return config.USE_FAKES or not _ml_available()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
def rank(query: str) -> list[str]:
    """All guidebook entries, most similar to `query` first (R01 not moved)."""
    path = config.GUIDEBOOK_PATH  # read at call time so tests and notebooks can point elsewhere
    if _use_fake():
        entries = load_entries(path)
        order = _rank_overlap(query, entries)
    else:
        entries, order = _rank_faiss(query, path)
    return [entries[i] for i in order]


def retrieve(query: str, k: int = config.RAG_K) -> list[str]:
    ranked = rank(query)
    top = ranked[: max(k, 0)]
    always = [e for e in ranked if entry_id(e) == config.ALWAYS_INCLUDED_RULE]
    # R01 first so the rewriter always sees the evidence-only rule; +1 only if it was not in the top k.
    return always + [e for e in top if entry_id(e) != config.ALWAYS_INCLUDED_RULE]


def retrieve_ids(query: str, k: int = config.RAG_K) -> list[str]:
    """Rule ids of retrieve(query, k), e.g. ['R01', 'R07', 'R03']. Used by the evaluation."""
    return [entry_id(e) for e in retrieve(query, k)]
