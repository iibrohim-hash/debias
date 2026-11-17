"""Fine-tune the detector on prepared FigNews posts (Person A, tasks A6-A7, A9).

    from src import fignews, train_detector
    posts, report = fignews.prepare(fignews.load_raw())
    run = train_detector.train(posts, seed=42, sample=500)   # quick check first
    run = train_detector.train(posts, seed=42)               # full run

The same function trains the second detector:
    train_detector.train(posts2, model_name=config.SECOND_DETECTOR_BASE_MODEL, ...)
where posts2 comes from fignews.prepare(raw, seed=<other seed>) (a different split).

Saves the model, tokenizer, tuned thresholds and metrics under
utils.artifacts_dir()/detector/<run_name>/. Needs a GPU for full runs (T4: fp16).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, fignews, metrics, utils


def _dataset(df: pd.DataFrame, tokenizer):
    from datasets import Dataset

    ds = Dataset.from_pandas(df[["text_masked", "label_id"]].rename(columns={"label_id": "labels"}),
                             preserve_index=False)
    return ds.map(lambda b: tokenizer(b["text_masked"], truncation=True, max_length=config.MAX_SEQ_LEN),
                  batched=True, remove_columns=["text_masked"])


def _weighted_trainer_cls():
    import torch
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        """Cross-entropy with class weights w_c = N / (K * N_c)."""

        def __init__(self, *args, class_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))
            loss = loss_fn(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def predict_probs(model, tokenizer, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Softmax probabilities for already-masked texts."""
    import torch

    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], truncation=True, max_length=config.MAX_SEQ_LEN,
                        padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out.append(torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(config.LABELS)))


def train(
    posts: pd.DataFrame,
    seed: int = config.SEED,
    model_name: str = config.DETECTOR_BASE_MODEL,
    run_name: str | None = None,
    sample: int | None = None,
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 16,
    ci: bool = True,
    save_checkpoints: bool = True,
    fp16: bool | None = None,
) -> dict:
    """Train one seed, tune thresholds on val, evaluate on test. Returns a run dict."""
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, TrainingArguments)

    utils.set_seed(seed)
    short = model_name.split("/")[-1]
    run_name = run_name or f"{short}_seed{seed}" + (f"_sample{sample}" if sample else "")
    out_dir = utils.artifacts_dir() / "detector" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = posts[posts["split"] == "train"]
    val_df = posts[posts["split"] == "val"]
    test_df = posts[posts["split"] == "test"]
    if sample:  # quick run: small stratified sample of every split
        def take(d: pd.DataFrame, n: int) -> pd.DataFrame:
            idx = [g.sample(n=min(len(g), max(1, n * len(g) // len(d))), random_state=seed).index
                   for _, g in d.groupby("label")]
            return d.loc[idx[0].append(idx[1:])]

        train_df, val_df, test_df = take(train_df, sample), take(val_df, sample // 4), take(test_df, sample // 4)

    weights = fignews.class_weights(train_df)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(config.LABELS), id2label=config.ID2LABEL, label2id=config.LABEL2ID,
        dtype=torch.float32,
    )
    # Weights must be fp32 for mixed-precision training (fp16 only for the maths).
    # mDeBERTa otherwise ends up with some fp16 parameters under transformers 5 and
    # training fails with "Attempting to unscale FP16 gradients".
    model = model.float()

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        warmup_steps=0.1,  # transformers 5: a float in [0, 1) is a ratio of total steps
        weight_decay=0.01,
        eval_strategy="epoch",
        # Per-epoch checkpoints keep the best epoch. Model weights only (~1.1 GB, no
        # optimizer state) so they fit on a free Drive. save_checkpoints=False for
        # quick runs or small disks.
        save_strategy="epoch" if save_checkpoints else "no",
        save_only_model=True,
        save_total_limit=1,
        load_best_model_at_end=save_checkpoints,
        metric_for_best_model="macro_f1",
        # fp16=False trains in full fp32 (slower); use it if the loss becomes nan.
        fp16=torch.cuda.is_available() if fp16 is None else fp16,
        seed=seed,
        data_seed=seed,
        report_to="none",
        logging_steps=50,
    )

    def compute_metrics(pred):
        logits, labels = pred
        probs = torch.softmax(torch.tensor(logits).float(), dim=-1).numpy()
        return {"macro_f1": metrics.per_class(labels, metrics.predict(probs, 0.5))["macro_f1"]}

    trainer = _weighted_trainer_cls()(
        model=model,
        args=args,
        train_dataset=_dataset(train_df, tokenizer),
        eval_dataset=_dataset(val_df, tokenizer),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        class_weights=weights,
    )
    trainer.train()

    model = trainer.model
    probs_val = predict_probs(model, tokenizer, val_df["text_masked"].tolist())
    probs_test = predict_probs(model, tokenizer, test_df["text_masked"].tolist())
    thresholds = metrics.tune_thresholds(val_df["label_id"].to_numpy(), probs_val)
    result = {
        "run_name": run_name,
        "model_name": model_name,
        "seed": seed,
        "sample": sample,
        "class_weights": weights,
        "sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "thresholds": thresholds,
        "val": metrics.evaluate(val_df["label_id"].to_numpy(), probs_val, thresholds["biased"], ci=False),
        "test": metrics.evaluate(test_df["label_id"].to_numpy(), probs_test, thresholds["biased"], ci=ci),
    }

    trainer.save_model(str(out_dir / "model"))
    tokenizer.save_pretrained(str(out_dir / "model"))
    (out_dir / "model" / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    pd.DataFrame(probs_test, columns=list(config.LABELS)).assign(
        post_id=test_df["post_id"].to_numpy(), label=test_df["label"].to_numpy()
    ).to_csv(out_dir / "test_probs.csv", index=False)
    utils.save_result(f"detector_{run_name}", result)
    result["model_dir"] = str(out_dir / "model")
    return result


def summarize_seeds(runs: list[dict]) -> dict:
    """Mean and std of test per-class F1 and macro-F1 over seeds."""
    rows = {}
    for name in [*config.LABELS, "macro_f1"]:
        vals = [r["test"]["point"][name]["f1"] if name != "macro_f1" else r["test"]["point"]["macro_f1"]
                for r in runs]
        rows[name] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}
    return rows


def load_thresholds(model_dir: str | Path) -> dict:
    return json.loads((Path(model_dir) / "thresholds.json").read_text())
