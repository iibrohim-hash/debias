"""FigNews 2024 preparation for the detector (Person A, tasks A1-A5).

    from src import fignews
    raw = fignews.load_raw()                 # all annotation rows, bias subtask
    posts, report = fignews.prepare(raw)     # one row per post, cleaned, masked, split

`posts` columns: post_id, source_language, id, is_original_english, text_en,
text_masked, label, label_id, n_annotators, agreement, split ("train"/"val"/"test").

Everything is deterministic given config.SEED, so every teammate gets the same split.
Pure pandas: runs on CPU, in CI and locally.
"""
from __future__ import annotations

import re

import pandas as pd

from . import config
from .placeholders import find_glued, find_leftovers, mask_sides

KEY = ["source_language", "id"]


# ---------------------------------------------------------------------------
# A1. Load
# ---------------------------------------------------------------------------
def load_raw(path: str | None = None) -> pd.DataFrame:
    """Bias-subtask annotation rows at the pinned revision.

    `path`: a local copy of FIGNEWS-2024.tsv (optional). Otherwise loaded with
    `datasets` from the Hub.
    """
    if path:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    else:
        from datasets import load_dataset

        ds = load_dataset(config.FIGNEWS_DATASET, revision=config.FIGNEWS_REVISION, split="train")
        df = ds.to_pandas().astype(str)
    return df[df["subtask"] == config.FIGNEWS_SUBTASK].reset_index(drop=True)


# ---------------------------------------------------------------------------
# A2. Merge annotators
# ---------------------------------------------------------------------------
def merge_annotators(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per post with the majority label over ALL raw labels.

    Voting happens on the original 7 labels, so a post whose majority is
    "Unclear" is dropped rather than counted as unbiased.
    """
    rows = []
    for (lang, pid), g in raw.groupby(KEY, sort=False):
        counts = g["label"].value_counts()
        top, n_top, n = counts.index[0], int(counts.iloc[0]), int(len(g))
        tie = len(counts) > 1 and int(counts.iloc[1]) == n_top
        first = g.iloc[0]
        rows.append({
            "source_language": lang,
            "id": pid,
            "text": first["text"],
            "english_mt": first["english_mt"],
            "raw_label": top,
            "n_annotators": n,
            "agreement": n_top / n,
            "kept_vote": (n_top >= config.MIN_ANNOTATOR_AGREEMENT) and not tie and n_top / n > 0.5,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# A3. Clean
# ---------------------------------------------------------------------------
_URL = re.compile(r"(https?://\S+|www\.\S+|\bt\.co/\S+)")
_HANDLE = re.compile(r"(?<!\w)@\w+")
_SPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = _URL.sub(" ", text)
    text = _HANDLE.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def _dedup_key(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.lower()).strip()


# ---------------------------------------------------------------------------
# A4. Split  /  A5. Mask
# ---------------------------------------------------------------------------
def _stratified_take(df: pd.DataFrame, frac: float, by: list[str], seed: int) -> pd.Index:
    taken = []
    for _, g in df.groupby(by, sort=True):
        n = int(round(len(g) * frac))
        taken.append(g.sample(n=n, random_state=seed).index)
    taken = [t for t in taken if len(t)]
    return taken[0].append(taken[1:]) if taken else df.index[:0]


def prepare(raw: pd.DataFrame, seed: int = config.SEED, min_words: int = 3) -> tuple[pd.DataFrame, dict]:
    """Merge, map labels, clean, deduplicate, split and mask. Returns (posts, report)."""
    report: dict = {"annotation_rows": len(raw)}
    posts = merge_annotators(raw)
    report["posts"] = len(posts)

    posts = posts[posts["kept_vote"]]
    report["after_agreement"] = len(posts)

    posts = posts[posts["raw_label"].isin(config.FIGNEWS_LABEL_MAP)].copy()
    posts["label"] = posts["raw_label"].map(config.FIGNEWS_LABEL_MAP)
    posts["label_id"] = posts["label"].map(config.LABEL2ID)
    report["after_label_map"] = len(posts)

    posts["is_original_english"] = posts["source_language"] == "English"
    posts["text_en"] = posts["text"].where(posts["is_original_english"], posts["english_mt"]).map(clean_text)
    posts = posts[posts["text_en"].str.split().str.len() >= min_words]
    report["after_cleaning"] = len(posts)

    # Duplicates: keep one copy, preferring the original English version.
    posts["_dup"] = posts["text_en"].map(_dedup_key)
    posts = posts.sort_values(["is_original_english", "agreement"], ascending=[False, False])
    posts = posts.drop_duplicates("_dup", keep="first").drop(columns="_dup")
    report["after_dedup"] = len(posts)

    # Test = original English posts only (stratified by label).
    posts["split"] = "train"
    english = posts[posts["is_original_english"]]
    test_idx = _stratified_take(english, config.TEST_FRACTION_OF_ENGLISH, ["label"], seed)
    posts.loc[test_idx, "split"] = "test"
    rest = posts[posts["split"] == "train"]
    val_idx = _stratified_take(rest, config.VAL_FRACTION, ["label", "source_language"], seed)
    posts.loc[val_idx, "split"] = "val"

    posts["text_masked"] = posts["text_en"].map(lambda t: mask_sides(t, "israel_palestine"))
    posts["post_id"] = posts["source_language"].str[:2].str.upper() + "-" + posts["id"].astype(str)

    # Leftover check: Israel/Palestine names must be 0; other-conflict names and
    # glued hashtags are reported so the team can decide whether to handle them.
    same = posts["text_masked"].map(lambda t: find_leftovers(t, "israel_palestine"))
    other = posts["text_masked"].map(lambda t: find_leftovers(t, "russia_ukraine"))
    glued = posts["text_masked"].map(find_glued)
    report["posts_with_leftover_names"] = int((same.str.len() > 0).sum())
    report["posts_with_russia_ukraine_names"] = int((other.str.len() > 0).sum())
    report["posts_with_glued_names"] = int((glued.str.len() > 0).sum())
    report["glued_examples"] = glued[glued.str.len() > 0].explode().value_counts().head(15).to_dict()
    report["class_counts"] = (
        posts.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict(orient="index")
    )
    report["test_english_only"] = bool(posts.loc[posts["split"] == "test", "is_original_english"].all())

    cols = ["post_id", "source_language", "id", "is_original_english", "text_en", "text_masked",
            "label", "label_id", "n_annotators", "agreement", "split"]
    return posts[cols].sort_values("post_id").reset_index(drop=True), report


def class_weights(train: pd.DataFrame) -> list[float]:
    """w_c = N / (K * N_c), in config.LABELS order."""
    counts = train["label_id"].value_counts().reindex(range(len(config.LABELS)), fill_value=0)
    n, k = len(train), len(config.LABELS)
    return [n / (k * c) if c else 0.0 for c in counts]
