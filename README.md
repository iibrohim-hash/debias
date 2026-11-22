# debias: political bias mitigation in chatbot answers

67-341 Generative AI & Governance, CMU-Q.
Team: Iroda (A, detector), Madina (B, Llama rewriter + reviewer prompts), Amen (C, guidebook), Yasa (D, RAG).

The system takes a chatbot answer about a contested conflict, finds biased wording, rewrites only
that wording without changing facts, checks the rewrite from both sides' perspectives, and returns
one answer plus a change note. Main case: Israel-Palestine (detector trained on FigNews 2024).
Generalization case: Russia-Ukraine (no training).

## Quick start for teammates

1. Open `notebooks/00_template.ipynb` (Colab: File > Open notebook > GitHub > `iibrohim-hash/debias`),
   then save a copy as `notebooks/<letter>_<topic>.ipynb`, e.g. `B_rewriter.ipynb`, `C_guidebook.ipynb`, `D_rag.ipynb`.
2. Always run the first setup cell first. It installs the same library versions for everyone, loads the
   shared code and sets the seed. If Colab asks to restart, restart and run it again.
3. Add your Hugging Face token as a secret named `HF_TOKEN` (Colab: key icon on the left; Kaggle:
   Add-ons > Secrets). Never paste the token into a notebook: the repo is public.

Rules:

- Take labels, seeds, model names, thresholds etc. from `src/config.py`; do not redefine them in your
  notebook. If you need a change, tell the group.
- Function signatures are fixed (see [Shared interface](#shared-interface)). Your file already exists
  with a fake version:
  - Madina: `src/rewriter.py`, `src/reviewers.py`, `prompts/`
  - Yasa: `src/rag.py`
  - Amen: `data/guidebook.md`, `data/test_sentences.csv`
  - Iroda: `src/detector.py`, `src/placeholders.py`
- Model weights and checkpoints go to Drive or Hugging Face Hub, not GitHub.
- Work on your own branch (e.g. `madina/rewriter`) and open a pull request into `main`. Tests run automatically.
- Open request: someone please review the side-name list in `src/placeholders.py` (what gets replaced
  with `[SIDE_A]` / `[SIDE_B]`).

## Layout

```
src/
  config.py        shared constants: labels, sides, model ids, seeds, thresholds, paths
  utils.py         seeds, Colab/Kaggle/local paths, HF token, save_result, audit log
  contract.py      validators for the shared interface (do not change without telling the team)
  placeholders.py  [SIDE_A]/[SIDE_B] masking for both conflicts       (A, Iroda)
  detector.py      detect(), explain()                                 (A, Iroda)
  fignews.py       FigNews prep: annotator vote, cleaning, split, masking (A)
  train_detector.py  fine-tuning with class weights, threshold tuning    (A)
  metrics.py       per-class F1, bootstrap CIs, thresholds (both detectors) (A)
  second_detector.py  mDeBERTa scores before/after the rewrite (evaluation only) (A)
  baseline.py      plain-Llama baseline answers + baseline vs responsible table
  rag.py           retrieve()                                          (D, Yasa)
  rewriter.py      rewrite()                                           (B, Madina)
  reviewers.py     review(): Reviewer A/B, one prompt, side as parameter (B)
  reviewer_eval.py reviewer accuracy, symmetry, false alarms on labeled cases (B)
  pipeline.py      run(): detector -> explain -> RAG -> rewrite -> fact check -> reviewers -> evaluator
data/
  guidebook.md               (C, Amen)
  test_sentences.csv         id, conflict, biased, neutral, expected_rules   (C)
  reviewer_cases.csv         40 labeled mirror cases (dev / heldout) for tuning the reviewers (B)
  probes.csv                 id, conflict, type, prompt, swap_pair_id (50, draft) (B, D)
  russia_ukraine_to_label.csv id, outlet, text, label_1, label_2             (D)
prompts/                     (B, Madina)
notebooks/                   one per person; start from 00_template.ipynb
  F_full_pipeline.ipynb      all parts end to end on the 50 test sentences + second detector
  G_baseline_vs_responsible.ipynb  plain Llama vs the pipeline on the probes (final comparison)
tests/                       interface tests, run by GitHub Actions on every push
```

Functions not built yet return fake values with the right shape, so everyone can work in parallel
and `src.pipeline.run()` works end to end from day one.

## Working in Colab, Kaggle or local Jupyter

1. Open `notebooks/00_template.ipynb` (Colab: File > Open notebook > GitHub > `iibrohim-hash/debias`).
   Save a copy as `notebooks/<letter>_<topic>.ipynb`.
2. Run the setup cell first. It clones/pulls the repo, installs `requirements.txt`, adds the repo
   to the path, seeds everything, mounts Drive (Colab) and prints the environment.
   If Colab asks to restart after the install, restart and run the setup cell again.
3. Hugging Face token: Colab > Secrets (key icon) > `HF_TOKEN`; Kaggle > Add-ons > Secrets > `HF_TOKEN`.
   Never paste a token into a notebook or a file. This repo is public.

Local Jupyter (Python 3.10+):

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt torch jupyter pytest
```

## Rules that keep everyone consistent

- **Take constants from `src/config.py`.** Labels, label order, seeds, model ids, thresholds,
  `MAX_NEW_TOKENS`, `MAX_ROUNDS`, `RAG_K`, the FigNews revision. Do not redefine them in a notebook.
  Need a change? Open a PR and tell the team.
- **Same library versions.** `requirements.txt` pins the ML libraries. torch/numpy/pandas/scikit-learn
  use the runtime's versions; `utils.print_env()` shows them.
- **Same data.** FigNews is loaded with `config.FIGNEWS_REVISION`. Side names are always masked with
  `src.placeholders.mask_sides`, in training and at run time.
- **Record every result** with `utils.save_result(name, metrics)`. It stores the git commit, seed and
  package versions next to the numbers.
- **Weights and checkpoints** go to `utils.artifacts_dir()` (Drive on Colab, `/kaggle/working` on
  Kaggle) or a private Hugging Face Hub repo. Never to GitHub.
- **Interface.** Signatures are in `src/contract.py` and the plan, section 5. Run `pytest` before you push.
- **Notebooks for experiments, `src/` for finished functions.** Clear large outputs before committing.

## Git workflow

- Work on your own branch (`iroda/detector`, `madina/rewriter`, `amen/guidebook`, `yasa/rag`),
  open a pull request into `main`, and wait for the tests to pass.
- In the setup cell, set `BRANCH` to your branch while developing.
- From Colab: File > Save a copy in GitHub, choose your branch.

## Shared interface

```python
# conflict is "israel_palestine" (default) or "russia_ukraine"
detect(text, conflict="israel_palestine") -> {"status": "unbiased"|"uncertain"|"biased", "score": float, "label": str}
explain(text, conflict="israel_palestine") -> list[str]
retrieve(query, k=3) -> list[str]            # R01 always included
rewrite(text, spans, rules, feedback=None) -> {"revised": str, "changes": [{"original", "revised", "reason"}]}
review(original, revised, perspective) -> {"fair": bool, "issues": [{"phrase", "problem"}]}
# perspective: "palestinian" | "israeli" | "ukrainian" | "russian"
```

Detector labels (order fixed): `0 unbiased`, `1 against_SIDE_A`, `2 against_SIDE_B`. Bias score `s = 1 - P(unbiased)`.

| Conflict | [SIDE_A] | [SIDE_B] | Reviewer A | Reviewer B |
|---|---|---|---|---|
| israel_palestine | Israel side | Palestine side | palestinian | israeli |
| russia_ukraine | Russia side | Ukraine side | ukrainian | russian |

## Data licenses

FigNews 2024: CC BY-SA 4.0. DNIPRO: check the license (non-commercial use) before committing any article text.
