"""Data prep on a small synthetic FigNews-shaped table (no download needed)."""
import pandas as pd
import pytest

from src import config, fignews

pytest.importorskip("pandas")


def _rows(lang, pid, labels, text, mt=None):
    return [{"source_language": lang, "id": str(pid), "subtask": "Bias", "label": lab,
             "text": text, "english_mt": mt or text} for lab in labels]


def _raw():
    rows = []
    rows += _rows("English", 1, ["Unbiased"] * 3, "Talks resumed in Cairo today https://t.co/x @user")
    rows += _rows("English", 2, ["Biased against Israel", "Biased against Israel", "Unbiased"],
                  "Israeli occupiers committed another crime #IsraelHamasWar")
    rows += _rows("English", 3, ["Biased against Palestine"] * 2, "Hamas terrorists butchered families")
    rows += _rows("English", 4, ["Unbiased", "Unclear", "Biased against Israel"], "No majority here at all")
    rows += _rows("English", 5, ["Unclear", "Unclear", "Unbiased"], "Majority label is unclear here")
    rows += _rows("Arabic", 1, ["Biased against Palestine"] * 3, "نص", "Gaza militants attacked the town again")
    rows += _rows("Hebrew", 7, ["Unbiased"] * 2, "טקסט", "Talks resumed in Cairo today")  # duplicate of EN-1
    for i in range(10, 40):  # enough English posts for a test split
        lab = ["Unbiased", "Biased against Israel", "Biased against Palestine"][i % 3]
        rows += _rows("English", i, [lab] * 3, f"Post number {i} about the war in Gaza")
    return pd.DataFrame(rows)


def test_prepare():
    posts, rep = fignews.prepare(_raw())
    ids = set(posts.post_id)
    assert "EN-4" not in ids  # no strict majority
    assert "EN-5" not in ids  # majority "Unclear" is dropped, not counted as unbiased
    assert "HE-7" not in ids and "EN-1" in ids  # duplicate: original English kept
    assert posts.set_index("post_id").loc["EN-2", "label"] == "against_SIDE_A"
    assert posts.set_index("post_id").loc["AR-1", "text_en"].startswith("Gaza militants")  # english_mt used
    en1 = posts.set_index("post_id").loc["EN-1", "text_en"]
    assert "http" not in en1 and "@user" not in en1
    assert "#[SIDE_A][SIDE_B]War" in posts.set_index("post_id").loc["EN-2", "text_masked"]
    assert rep["posts_with_leftover_names"] == 0
    assert posts[posts.split == "test"].is_original_english.all()
    assert set(posts.split) == {"train", "val", "test"}


def test_prepare_is_deterministic():
    a, _ = fignews.prepare(_raw())
    b, _ = fignews.prepare(_raw())
    pd.testing.assert_frame_equal(a, b)


def test_class_weights():
    df = pd.DataFrame({"label_id": [0, 0, 0, 1, 2, 2]})
    w = fignews.class_weights(df)
    assert w == pytest.approx([6 / 9, 6 / 3, 6 / 6])
    assert len(w) == len(config.LABELS)
