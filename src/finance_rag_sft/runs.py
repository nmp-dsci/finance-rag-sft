"""Run folders: the record. `runs/<id>/run.json` + `queries.jsonl`; MLflow replays them.

A run written on another machine (an EC2 GPU, a container) is complete on disk and is logged
to MLflow later with `finrag log-run runs/<id>`.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from finance_rag_sft import config
from finance_rag_sft.harness import K, QueryResult


def new_run_id(kind: str, slug: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{kind}-{re.sub(r'[^a-z0-9.-]+', '-', slug.lower()).strip('-')}"


def run_dir(run_id: str) -> Path:
    return config.RUNS_DIR / run_id


def write_run(doc: dict[str, Any], results: dict[str, list[QueryResult]] | None = None) -> Path:
    folder = run_dir(doc["run_id"])
    folder.mkdir(parents=True, exist_ok=True)
    if results:
        dims = list(results)
        first = results[dims[0]]
        lines = []
        for i, r in enumerate(first):
            row: dict[str, Any] = {"qid": r.qid, "ranks": {}}
            for dim in dims:
                rr = results[dim][i]
                assert rr.qid == r.qid
                row["ranks"][dim] = {
                    "golden_ranks": rr.golden_ranks,
                    "top": [[p, round(s, 5)] for p, s in rr.top],
                }
            lines.append(json.dumps(row, separators=(",", ":")))
        (folder / "queries.jsonl").write_text("\n".join(lines) + "\n")
    save_doc(doc)
    return folder


def save_doc(doc: dict[str, Any]) -> None:
    path = run_dir(doc["run_id"]) / "run.json"
    path.write_text(json.dumps(doc, indent=2) + "\n")


def read_run(run_id: str) -> dict[str, Any]:
    path = run_dir(run_id) / "run.json"
    if not path.exists():
        raise FileNotFoundError(run_id)
    doc: dict[str, Any] = json.loads(path.read_text())
    return doc


def list_runs() -> list[dict[str, Any]]:
    if not config.RUNS_DIR.exists():
        return []
    docs = []
    for path in config.RUNS_DIR.glob("*/run.json"):
        try:
            docs.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return sorted(docs, key=lambda d: (d.get("created", ""), d["run_id"]), reverse=True)


@lru_cache(maxsize=32)
def _queries(path: str, mtime_ns: int) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text().splitlines():
        if line:
            row = json.loads(line)
            out[row["qid"]] = row
    return out


def read_queries(run_id: str) -> dict[str, dict[str, Any]]:
    path = run_dir(run_id) / "queries.jsonl"
    if not path.exists():
        return {}
    return _queries(str(path), path.stat().st_mtime_ns)


def golden_ranks(run_id: str, dim: str) -> dict[str, list[int]]:
    return {q: row["ranks"][dim]["golden_ranks"] for q, row in read_queries(run_id).items()}


def run_doc(
    *,
    run_id: str,
    kind: str,
    retriever: dict[str, Any] | None,
    split: Any,
    eval_split: str | None,
    dims: list[str],
    n_queries: int,
    metrics: dict[str, dict[str, float]],
    timing: dict[str, float],
    device: str | None,
    note: str = "",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "kind": kind,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "label": retriever["label"] if retriever else extra.pop("label", run_id),
        "retriever": retriever,
        "split": split.version,
        "split_fingerprint": split.fingerprint,
        "eval_split": eval_split,
        "dims": dims,
        "n_queries": n_queries,
        "n_passages": len(split.pids),
        "k": K,
        "metrics": metrics,
        "timing": timing,
        "device": device,
        "code_sha": config.settings().code_sha,
        "mlflow_run_id": None,
        "note": note,
        **extra,
    }


def summary(doc: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "run_id",
        "kind",
        "created",
        "label",
        "retriever",
        "split",
        "eval_split",
        "dims",
        "n_queries",
        "metrics",
        "mlflow_run_id",
        "note",
    )
    return {k: doc.get(k) for k in keys}
