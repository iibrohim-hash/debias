"""Environment helpers shared by all notebooks: seeds, paths, tokens, run records."""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import config


def runtime() -> str:
    """Return "colab", "kaggle" or "local"."""
    if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle").exists():
        return "kaggle"
    return "local"


def set_seed(seed: int = config.SEED) -> None:
    """Seed python, numpy and torch (if installed)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def artifacts_dir() -> Path:
    """Folder for checkpoints, results and logs (never committed to git).

    Override with DEBIAS_ARTIFACTS. Defaults: Drive on Colab, /kaggle/working on
    Kaggle, ./artifacts locally.
    """
    if os.environ.get("DEBIAS_ARTIFACTS"):
        path = Path(os.environ["DEBIAS_ARTIFACTS"])
    elif runtime() == "colab":
        path = Path("/content/drive/MyDrive/debias")
    elif runtime() == "kaggle":
        path = Path("/kaggle/working/artifacts")
    else:
        path = config.ROOT / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_hf_token() -> str | None:
    """Read the Hugging Face token from env, Colab secrets or Kaggle secrets.

    Never paste a token into a notebook: this repo is public.
    """
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    if runtime() == "colab":
        try:
            from google.colab import userdata

            return userdata.get("HF_TOKEN")
        except Exception:
            return None
    if runtime() == "kaggle":
        try:
            from kaggle_secrets import UserSecretsClient

            return UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            return None
    return None


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(config.ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _version(pkg: str) -> str:
    try:
        from importlib.metadata import version

        return version(pkg)
    except Exception:
        return "not installed"


TRACKED_PACKAGES = (
    "torch", "transformers", "datasets", "accelerate", "bitsandbytes",
    "sentence-transformers", "faiss-cpu", "captum", "scikit-learn",
    "pandas", "numpy",
)


def environment() -> dict:
    """Everything needed to say where a result came from."""
    gpu = "none"
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return {
        "runtime": runtime(),
        "python": platform.python_version(),
        "gpu": gpu,
        "git_commit": git_commit(),
        "packages": {p: _version(p) for p in TRACKED_PACKAGES},
    }


def print_env() -> None:
    env = environment()
    print(f"runtime={env['runtime']}  python={env['python']}  gpu={env['gpu']}  commit={env['git_commit']}")
    for pkg, ver in env["packages"].items():
        print(f"  {pkg:22s} {ver}")


def save_result(name: str, metrics: dict, extra: dict | None = None) -> Path:
    """Save metrics as JSON together with seed, commit and package versions."""
    record = {
        "name": name,
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": metrics,
        "extra": extra or {},
        "environment": environment(),
    }
    out = artifacts_dir() / "results" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=str))
    return out


def audit_log(record: dict, path: Path | None = None) -> None:
    """Append one pipeline decision to a JSON-lines audit log."""
    path = path or artifacts_dir() / "audit_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
