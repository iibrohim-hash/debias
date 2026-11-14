"""Shared settings for the whole team.

Every notebook and every module imports its constants from here. If you need a
different value, change it here in a pull request and tell the team, do not
redefine it inside your notebook. That is what keeps runs on Colab, Kaggle and
local Jupyter comparable.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Conflicts, sides, labels
# ---------------------------------------------------------------------------
CONFLICTS = ("israel_palestine", "russia_ukraine")
DEFAULT_CONFLICT = "israel_palestine"

# [SIDE_A] / [SIDE_B] meaning per conflict (project plan, section 4).
SIDES = {
    "israel_palestine": {"SIDE_A": "Israel side", "SIDE_B": "Palestine side"},
    "russia_ukraine": {"SIDE_A": "Russia side", "SIDE_B": "Ukraine side"},
}
PLACEHOLDER_A = "[SIDE_A]"
PLACEHOLDER_B = "[SIDE_B]"

# Detector classes. The order is fixed: class ids are saved inside the model.
LABELS = ("unbiased", "against_SIDE_A", "against_SIDE_B")
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

DETECT_STATUSES = ("unbiased", "uncertain", "biased")

# Reviewer perspectives: (Reviewer A, Reviewer B) per conflict.
PERSPECTIVES = {
    "israel_palestine": ("palestinian", "israeli"),
    "russia_ukraine": ("ukrainian", "russian"),
}
ALL_PERSPECTIVES = tuple(p for pair in PERSPECTIVES.values() for p in pair)

# ---------------------------------------------------------------------------
# Models (Hugging Face ids)
# ---------------------------------------------------------------------------
DETECTOR_BASE_MODEL = "FacebookAI/xlm-roberta-base"
SECOND_DETECTOR_BASE_MODEL = "microsoft/mdeberta-v3-base"  # evaluation only
REWRITER_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
REWRITER_FALLBACK_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# DeBERTa-v3 trained on MNLI; confirm the exact checkpoint before the fact-check work.
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# Where trained detector weights live (private HF Hub repo or a Drive folder).
# Set DEBIAS_DETECTOR_PATH in your notebook/session when the model exists.
DETECTOR_PATH = os.environ.get("DEBIAS_DETECTOR_PATH")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
FIGNEWS_DATASET = "CAMeL-Lab/FIGNEWS-2024"
# Pin the dataset revision so everyone loads exactly the same rows.
FIGNEWS_REVISION = "44e3d1f65c2e1dad3732c83b31ad4023bec18e2d"
MIN_ANNOTATOR_AGREEMENT = 2  # keep posts where >= 2 of 3 annotators agree

# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
SEED = 42
SEEDS = (42, 43, 44)  # the 3 seeds for the final detector runs
VAL_FRACTION = 0.15
MAX_SEQ_LEN = 256
BOOTSTRAP_SAMPLES = 1000
CI_LEVEL = 0.95

# Detector decision rule on s = 1 - P(unbiased).
# Tuned on validation only (task A7). Until then these are placeholders.
#   s <  UNCERTAIN_LOW            -> "unbiased"
#   UNCERTAIN_LOW <= s < BIASED   -> "uncertain" (goes to a human)
#   s >= BIASED                   -> "biased"
THRESHOLDS = {"uncertain_low": 0.3, "biased": 0.5}
THRESHOLDS_TUNED = False

# ---------------------------------------------------------------------------
# Generation / pipeline
# ---------------------------------------------------------------------------
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.0  # greedy decoding so reruns give the same text
MAX_ROUNDS = 3
RAG_K = 3
ALWAYS_INCLUDED_RULE = "R01"  # evidence-only rule

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROMPTS_DIR = ROOT / "prompts"
GUIDEBOOK_PATH = DATA_DIR / "guidebook.md"
TEST_SENTENCES_PATH = DATA_DIR / "test_sentences.csv"
PROBES_PATH = DATA_DIR / "probes.csv"

# Set DEBIAS_FAKE=1 to force the fake stand-ins (tests and CI do this).
USE_FAKES = os.environ.get("DEBIAS_FAKE", "0") == "1"


def check_conflict(conflict: str) -> str:
    if conflict not in CONFLICTS:
        raise ValueError(f"conflict must be one of {CONFLICTS}, got {conflict!r}")
    return conflict
