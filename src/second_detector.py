"""Second detector: mDeBERTa-v3, trained on a different split (A_detector, A9).

Evaluation only. It never runs inside the pipeline and nothing is tuned on it.
It scores the original answer and the final answer, so we can check whether the
rewrite lowered the bias with a model that did NOT guide the rewrite (the main
XLM-R detector chose the flagged words, so re-scoring with it would be circular).

    from src import second_detector
    second_detector.set_model_path(ARTIFACTS / "detector" / "mdeberta-v3-base_seed42" / "model")
    second_detector.scores(["text 1", "text 2"])   # -> [s1, s2], s = 1 - P(unbiased)
    second_detector.status(0.7)                    # -> "biased" | "uncertain" | "unbiased"
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from . import config
from .placeholders import mask_sides

DEFAULT_RUN = f"{config.SECOND_DETECTOR_BASE_MODEL.split('/')[-1]}_seed{config.SEED}"
_LOADED_WORDS = ("terrorists", "butchered", "massacre", "regime", "savage", "heroic", "slaughtered")
_state: dict = {"path": None}


def set_model_path(path) -> None:
    _state["path"] = str(path)
    _load.cache_clear()
    thresholds.cache_clear()


def _use_fake() -> bool:
    return config.USE_FAKES or not _state["path"]


def _fake_score(text: str) -> float:
    return 0.9 if any(w in text.lower() for w in _LOADED_WORDS) else 0.1


@lru_cache(maxsize=1)
def _load():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(_state["path"])
    # float32: mDeBERTa can load with some fp16 weights under transformers 5 (see train_detector.py)
    model = AutoModelForSequenceClassification.from_pretrained(_state["path"]).float().to(device).eval()
    return tok, model, device


@lru_cache(maxsize=1)
def thresholds() -> dict:
    """Thresholds tuned on the second detector's own validation set."""
    path = Path(_state["path"] or "") / "thresholds.json"
    if _state["path"] and path.exists():
        return json.loads(path.read_text())
    return dict(config.THRESHOLDS)


def scores(texts: list[str], conflict: str = config.DEFAULT_CONFLICT, batch_size: int = 32) -> list[float]:
    """Bias score s = 1 - P(unbiased) for each (unmasked) text."""
    config.check_conflict(conflict)
    if _use_fake():
        return [_fake_score(t) for t in texts]
    import torch

    tok, model, device = _load()
    out: list[float] = []
    for i in range(0, len(texts), batch_size):
        masked = [mask_sides(t, conflict) for t in texts[i:i + batch_size]]
        enc = tok(masked, truncation=True, max_length=config.MAX_SEQ_LEN, padding=True,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            p = torch.softmax(model(**enc).logits, dim=-1)[:, config.LABEL2ID["unbiased"]]
        out += (1.0 - p).cpu().tolist()
    return out


def status(score: float) -> str:
    t = thresholds()
    if score >= t["biased"]:
        return "biased"
    if score >= t["uncertain_low"]:
        return "uncertain"
    return "unbiased"
