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
# The other reviewer of the same conflict (used for the mirror-pair symmetry check).
OTHER_PERSPECTIVE = {a: b for pair in PERSPECTIVES.values() for a, b in (pair, pair[::-1])}

# ---------------------------------------------------------------------------
# Models (Hugging Face ids)
# ---------------------------------------------------------------------------
DETECTOR_BASE_MODEL = "FacebookAI/xlm-roberta-base"
SECOND_DETECTOR_BASE_MODEL = "microsoft/mdeberta-v3-base"  # evaluation only
REWRITER_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
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
# Posts have 2 to 50 annotators (MAIN and IAA sheets). Generalised rule: the top
# label needs >= MIN_ANNOTATOR_AGREEMENT votes AND a strict majority (> 50%).
# For 3-annotator posts this is exactly "2 of 3 agree"; 2-annotator posts need both.
FIGNEWS_SUBTASK = "Bias"
# Exact label strings in FIGNEWS-2024.tsv (checked 16 Nov 2026 at FIGNEWS_REVISION).
# Not mapped, so dropped: "Biased against others", "Biased against both Palestine
# and Israel", "Unclear", "Not Applicable".
FIGNEWS_LABEL_MAP = {
    "Unbiased": "unbiased",
    "Biased against Israel": "against_SIDE_A",
    "Biased against Palestine": "against_SIDE_B",
}
# A post is identified by (source_language, id): ids repeat across languages.
TEST_FRACTION_OF_ENGLISH = 0.4  # share of original English posts held out as test

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
# Evaluator rule (team decision pending, see the Oversight slide):
#   "disagree_escalates": reviewers disagree in any round -> a human at once (current rule).
#   "retry_first": retry with the reviewers' feedback while problems remain; after the
#                  last round a disagreement goes to a human, anything else is blocked.
EVALUATOR_RULES = ("disagree_escalates", "retry_first")
EVALUATOR_RULE = "disagree_escalates"
RAG_K = 3
EXPLAIN_STEPS = 50  # Integrated Gradients steps (Captum)
EXPLAIN_BATCH = 10  # steps per GPU batch; lower it if the explainer runs out of memory
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
REVIEWER_CASES_PATH = DATA_DIR / "reviewer_cases.csv"  # labeled set for tuning the reviewers

# Set DEBIAS_FAKE=1 to force the fake stand-ins (tests and CI do this).
USE_FAKES = os.environ.get("DEBIAS_FAKE", "0") == "1"


def check_conflict(conflict: str) -> str:
    if conflict not in CONFLICTS:
        raise ValueError(f"conflict must be one of {CONFLICTS}, got {conflict!r}")
    return conflict
