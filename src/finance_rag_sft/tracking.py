"""Log to the central MLflow (nmp-central-ai, http://localhost:5000).

MLflow is the index, never the record: a run's record is its folder under `runs/`, and
`log_run` replays that folder. Never falls back to a local store (PLATFORM.md rule 1).
"""

from __future__ import annotations

import functools
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from finance_rag_sft import config, runs


class TrackingDownError(RuntimeError):
    """The central MLflow is not reachable; the remedy is `make platform-up`."""


def preflight() -> str:
    uri = config.settings().mlflow_tracking_uri.rstrip("/")
    try:
        with urllib.request.urlopen(f"{uri}/health", timeout=3) as r:  # noqa: S310 - platform URL
            if r.status != 200:
                raise TrackingDownError(f"{uri}/health returned {r.status}")
    except (urllib.error.URLError, OSError) as e:
        raise TrackingDownError(
            f"MLflow at {uri} is not reachable ({e}). Start the platform: `make platform-up`"
        ) from e
    return uri


def is_up() -> bool:
    try:
        preflight()
    except TrackingDownError:
        return False
    return True


def required_tags(code_sha: str | None = None) -> dict[str, str]:
    """The tags PLATFORM.md requires on every run. No model API is billed: billing=none."""
    s = config.settings()
    return {
        "project": config.PROJECT,
        "git_sha": code_sha or s.code_sha,
        "env": s.env,
        "billing": "none",
    }


def _start() -> Any:
    import mlflow

    preflight()
    mlflow.set_tracking_uri(config.settings().mlflow_tracking_uri)
    mlflow.set_experiment(config.EXPERIMENT)
    return mlflow


@functools.cache
def _experiment_id() -> str | None:
    try:
        mlflow = _start()
        exp = mlflow.get_experiment_by_name(config.EXPERIMENT)
    except Exception:  # noqa: BLE001 - a link is a convenience; the platform may be down
        return None
    return str(exp.experiment_id) if exp else None


def run_url(mlflow_run_id: str | None) -> str | None:
    eid = _experiment_id() if mlflow_run_id else None
    if not eid:
        return None
    base = config.settings().mlflow_tracking_uri.rstrip("/")
    return f"{base}/#/experiments/{eid}/runs/{mlflow_run_id}"


def log_smoke(metrics: dict[str, float]) -> str:
    mlflow = _start()
    with mlflow.start_run(run_name="smoke") as run:
        mlflow.set_tags({**required_tags(), "kind": "smoke"})
        mlflow.log_params({"k": 10, "fixture": "cli.SMOKE_RANKS"})
        mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in metrics.items()})
        rid: str = run.info.run_id
    return rid


def _metric_key(metric: str, dim: str) -> str:
    return f"{metric.replace('@', '_at_')}_{'d' + dim if dim.isdigit() else dim}"


def log_run(folder: Path, register: bool = True) -> str:
    """Replay a run folder into MLflow; writes the MLflow run id back into run.json."""
    mlflow = _start()
    doc: dict[str, Any] = json.loads((folder / "run.json").read_text())
    retr = doc.get("retriever") or {}
    with mlflow.start_run(run_name=doc["run_id"]) as run:
        mlflow.set_tags(
            {
                **required_tags(doc.get("code_sha")),
                "kind": doc["kind"],
                "split": doc["split"],
                "split_fingerprint": doc["split_fingerprint"],
                "eval_split": str(doc.get("eval_split")),
                "model": str(retr.get("model")),
                "label": doc["label"],
                "run_dir": f"runs/{doc['run_id']}",
                "device": str(doc.get("device")),
            }
        )
        params: dict[str, Any] = {
            "retriever": retr.get("type"),
            "model": retr.get("model"),
            "revision": retr.get("revision"),
            "checkpoint": retr.get("checkpoint"),
            "train_run": retr.get("train_run"),
            "n_queries": doc.get("n_queries"),
            "n_passages": doc.get("n_passages"),
            "k": doc.get("k"),
            "dims": ",".join(doc.get("dims", [])),
        }
        train = doc.get("train")
        if train:
            params |= {f"train.{k}": v for k, v in train["config"].items()}
            params |= {"n_pairs": train["n_pairs"], "chosen_epoch": train["chosen"]["epoch"]}
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        for dim, ms in (doc.get("metrics") or {}).items():
            mlflow.log_metrics({_metric_key(m, dim): v for m, v in ms.items()})
        if check := doc.get("check"):
            mlflow.log_metrics({"check_reference": check["reference"], "check_got": check["got"]})
        if train:
            for e in train["epochs"]:
                for dim, ms in e["val"].items():
                    mlflow.log_metrics(
                        {"val_" + _metric_key(m, dim): v for m, v in ms.items()},
                        step=round(e["epoch"]),
                    )
            for point in train.get("loss_curve", []):
                mlflow.log_metric("train_loss", point["loss"], step=point["step"])
            mlflow.log_metric("train_seconds", train["train_seconds"])
        for name, secs in (doc.get("timing") or {}).items():
            mlflow.log_metric(f"seconds_{name}", secs)
        mlflow.log_artifact(str(folder / "run.json"))
        if (folder / "queries.jsonl").exists():
            mlflow.log_artifact(str(folder / "queries.jsonl"))
        if train and register:
            chosen = config.ROOT / train["chosen"]["checkpoint"]
            if chosen.exists():
                from finance_rag_sft.harness import load_st

                mlflow.sentence_transformers.log_model(
                    load_st(str(chosen)),
                    name="model",
                    registered_model_name=config.REGISTERED_MODEL,
                )
        rid: str = run.info.run_id
    doc["mlflow_run_id"] = rid
    runs.save_doc(doc)
    return rid
