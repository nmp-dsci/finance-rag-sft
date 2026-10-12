"""Paths, pinned inputs and settings. One place; nothing else reads `os.environ` for these.

MLflow is nmp-central-ai's (PLATFORM.md) on :5000. It is never replaced by a local store
(PLATFORM.md rule 1).
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(os.environ.get("FINRAG_ROOT", Path(__file__).resolve().parents[2]))
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"  # embedding caches; never committed
SPLITS_DIR = DATA_DIR / "splits"
RUNS_DIR = ROOT / "runs"  # the record: one folder per scored run, committed
OUTPUTS_DIR = ROOT / "outputs"  # checkpoints; never committed (weights go to MLflow)
CONFIGS_DIR = ROOT / "configs"
FRONTEND_DIST = ROOT / "frontend" / "dist"

PROJECT = "finance-rag-sft"
EXPERIMENT = "finance-rag-sft/evals"
REGISTERED_MODEL = "finance-rag-sft-bge-base"

# The dataset, pinned. A revision change is a new split version, never an edit to v1.
DATASET = "philschmid/finanical-rag-embedding-dataset"
DATASET_REVISION = "e0b17819cf52d444066c99f4a176f5717e066300"
DATASET_FILE = "data/train-00000-of-00001.parquet"
SPLIT_VERSION = "v1"


@dataclass(frozen=True)
class BaseModel:
    key: str
    hub_id: str
    revision: str
    dim: int
    label: str


BASE_MODELS: dict[str, BaseModel] = {
    "bge-base": BaseModel(
        "bge-base",
        "BAAI/bge-base-en-v1.5",
        "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        768,
        "bge-base (base)",
    ),
    "bge-small": BaseModel(
        "bge-small",
        "BAAI/bge-small-en-v1.5",
        "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        384,
        "bge-small (base)",
    ),
}
SMALL_DIM = 64  # the Matryoshka cut scored beside each model's full size


@dataclass(frozen=True)
class Settings:
    mlflow_tracking_uri: str
    code_sha: str
    env: str  # "local" | "aws" (PLATFORM.md required tag)
    device: str | None  # None = Sentence Transformers' choice: cuda, then mps, then cpu


def _git_sha() -> str:
    if sha := os.environ.get("GIT_SHA"):  # set in the container, which has no .git
        return sha[:7]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@cache
def settings() -> Settings:
    return Settings(
        mlflow_tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        code_sha=_git_sha(),
        env=os.environ.get("FINRAG_ENV", "local"),
        device=os.environ.get("FINRAG_DEVICE") or None,
    )
