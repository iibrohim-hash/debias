"""Bias detector: XLM-R, 3 classes (Person A, Iroda).

Interface (do not change without telling the team):
    detect(text, conflict="israel_palestine") -> {"status", "score", "label"}
    explain(text, conflict="israel_palestine") -> list[str]

Until a trained model exists (DEBIAS_DETECTOR_PATH unset) or when
DEBIAS_FAKE=1, both functions return fake values so teammates can build on them.
"""
from __future__ import annotations

from functools import lru_cache

from . import config
from .placeholders import mask_sides


def _use_fake() -> bool:
    return config.USE_FAKES or not config.DETECTOR_PATH


# ---------------------------------------------------------------------------
# Fake stand-ins
# ---------------------------------------------------------------------------
def _fake_detect(text: str, conflict: str) -> dict:
    return {"status": "biased", "score": 0.9, "label": "against_SIDE_B"}


def _fake_explain(text: str, conflict: str) -> list[str]:
    loaded = ("terrorists", "butchered", "massacre", "regime", "savage", "heroic")
    return [w for w in loaded if w in text.lower()]


# ---------------------------------------------------------------------------
# Real implementation (loads lazily, GPU if available)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(config.DETECTOR_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(config.DETECTOR_PATH).to(device).eval()
    return tok, model, device


def set_model_path(path: str) -> None:
    """Use the trained detector in `path` (a folder with the model and
    thresholds.json, e.g. on Drive). Same as setting DEBIAS_DETECTOR_PATH before
    importing, but works after the setup cell has already imported config."""
    config.DETECTOR_PATH = str(path)
    _load.cache_clear()
    thresholds.cache_clear()


@lru_cache(maxsize=1)
def thresholds() -> dict:
    """Tuned thresholds saved next to the model (thresholds.json from training);
    config.THRESHOLDS only if the model folder has none."""
    import json
    from pathlib import Path

    path = Path(config.DETECTOR_PATH or "") / "thresholds.json"
    if config.DETECTOR_PATH and path.exists():
        return json.loads(path.read_text())
    return dict(config.THRESHOLDS)


def _status(score: float) -> str:
    t = thresholds()
    if score >= t["biased"]:
        return "biased"
    if score >= t["uncertain_low"]:
        return "uncertain"
    return "unbiased"


def predict_proba(texts: list[str], conflict: str = config.DEFAULT_CONFLICT):
    """Class probabilities, shape (n, 3), for already-unmasked texts."""
    import torch

    tok, model, device = _load()
    masked = [mask_sides(t, conflict) for t in texts]
    enc = tok(masked, truncation=True, max_length=config.MAX_SEQ_LEN, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        return torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()


def detect(text: str, conflict: str = config.DEFAULT_CONFLICT) -> dict:
    config.check_conflict(conflict)
    if _use_fake():
        return _fake_detect(text, conflict)
    p = predict_proba([text], conflict)[0]
    score = float(1.0 - p[config.LABEL2ID["unbiased"]])
    # label = the more likely of the two "against" classes; "unbiased" if not biased
    status = _status(score)
    a, b = config.LABEL2ID["against_SIDE_A"], config.LABEL2ID["against_SIDE_B"]
    label = "unbiased" if status == "unbiased" else config.ID2LABEL[a if p[a] >= p[b] else b]
    return {"status": status, "score": score, "label": label}


def explain(text: str, conflict: str = config.DEFAULT_CONFLICT, top_k: int = 5) -> list[str]:
    """Words with the highest Integrated Gradients attribution toward the predicted
    "against" class. Placeholders are never returned."""
    config.check_conflict(conflict)
    if _use_fake():
        return _fake_explain(text, conflict)

    import torch
    from captum.attr import LayerIntegratedGradients

    tok, model, device = _load()
    masked = mask_sides(text, conflict)
    enc = tok(masked, truncation=True, max_length=config.MAX_SEQ_LEN, return_tensors="pt").to(device)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    with torch.no_grad():
        probs = torch.softmax(model(input_ids=ids, attention_mask=mask).logits, dim=-1)[0]
    a, b = config.LABEL2ID["against_SIDE_A"], config.LABEL2ID["against_SIDE_B"]
    target = a if probs[a] >= probs[b] else b

    baseline = torch.full_like(ids, tok.pad_token_id)
    baseline[0, 0], baseline[0, -1] = ids[0, 0], ids[0, -1]  # keep <s> and </s>
    lig = LayerIntegratedGradients(
        lambda i, m: model(input_ids=i, attention_mask=m).logits, model.get_input_embeddings()
    )
    attr = lig.attribute(ids, baselines=baseline, additional_forward_args=(mask,), target=target, n_steps=50)
    scores = attr.sum(dim=-1)[0].detach().cpu().tolist()

    # Merge sentencepiece pieces into words ("▁" starts a new word).
    words: list[tuple[str, float]] = []
    for piece, s in zip(tok.convert_ids_to_tokens(ids[0]), scores):
        if piece in tok.all_special_tokens:
            continue
        if piece.startswith("▁") or not words:
            words.append((piece.lstrip("▁"), s))
        else:
            words[-1] = (words[-1][0] + piece, words[-1][1] + s)

    ranked = sorted(words, key=lambda w: w[1], reverse=True)
    out: list[str] = []
    for word, s in ranked:
        clean = word.strip(".,;:!?\"'()")
        if s <= 0 or not clean or "SIDE_" in clean or clean in out:
            continue
        out.append(clean)
        if len(out) == top_k:
            break
    return out
