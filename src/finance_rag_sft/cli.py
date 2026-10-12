"""`finrag` — split, baseline, check, train, eval, compare, log-run, serve, smoke, doctor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from finance_rag_sft import config, runs

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
console = Console()

# Golden ranks for five made-up questions: the smoke run scores these, never a model.
SMOKE_RANKS = [[1], [3], [12], [2, 7], [40]]


def _log(folder: Path, mlflow: bool) -> None:
    if not mlflow:
        console.print(f"[yellow]not logged to MLflow[/]: run `finrag log-run {folder}` later")
        return
    from finance_rag_sft import tracking

    rid = tracking.log_run(folder)
    console.print(f"MLflow run {rid}  {tracking.run_url(rid) or ''}")


def _metrics_table(title: str, metrics: dict[str, dict[str, float]]) -> Table:
    t = Table(title=title)
    t.add_column("dim")
    names = ("recall@1", "recall@5", "recall@10", "ndcg@10", "mrr@10")
    for n in names:
        t.add_column(n, justify="right")
    for dim, ms in metrics.items():
        t.add_row(dim, *(f"{ms[n]:.4f}" for n in names))
    return t


@app.command()
def smoke() -> None:
    """Zero-model run: score a fixed toy ranking offline and log one kind=smoke run."""
    import numpy as np

    from finance_rag_sft import tracking
    from finance_rag_sft.harness import METRICS, per_query

    rows = [per_query(r) for r in SMOKE_RANKS]
    metrics = {m: float(np.mean([r[m] for r in rows])) for m in METRICS}
    rid = tracking.log_smoke(metrics)
    console.print(metrics)
    console.print(f"smoke run logged: {rid}")


@app.command()
def split(force: Annotated[bool, typer.Option(help="rewrite an existing split")] = False) -> None:
    """Write split v1 (IDs and hashes) from the pinned dataset. Once; it is committed."""
    from transformers import AutoTokenizer

    from finance_rag_sft.data import build_split, fetch_dataset, read_rows

    out = config.SPLITS_DIR / config.SPLIT_VERSION
    if (out / "manifest.json").exists() and not force:
        raise typer.BadParameter(f"{out} exists and is frozen; --force rewrites it")
    path = fetch_dataset()
    rows = read_rows(path)
    base = config.BASE_MODELS["bge-base"]
    tok = AutoTokenizer.from_pretrained(base.hub_id, revision=base.revision)
    passages = sorted({" ".join(c.split()) for _, c in rows})
    truncated = sum(1 for ids in tok(passages)["input_ids"] if len(ids) > 512)
    manifest = build_split(rows, out, truncated, hashlib.sha256(path.read_bytes()).hexdigest())
    console.print_json(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "passages",
                    "queries",
                    "pairs",
                    "groups",
                    "duplicates",
                    "multi_golden_queries",
                    "truncated_passages",
                    "fingerprint",
                )
            }
        )
    )


def _evaluate_and_write(
    retriever: object, kind: str, slug: str, eval_split: str, note: str, mlflow: bool
) -> str:
    from finance_rag_sft.data import load_split
    from finance_rag_sft.harness import evaluate

    s = load_split()
    metrics, results, timing = evaluate(retriever, s, eval_split)  # type: ignore[arg-type]
    run_id = runs.new_run_id(kind, slug)
    doc = runs.run_doc(
        run_id=run_id,
        kind=kind,
        retriever=retriever.describe(),  # type: ignore[attr-defined]
        split=s,
        eval_split=eval_split,
        dims=list(metrics),
        n_queries=len(s.qids(eval_split)),
        metrics=metrics,
        timing=timing,
        device=getattr(retriever, "device", None),
        note=note,
    )
    folder = runs.write_run(doc, results)
    console.print(_metrics_table(f"{run_id}  ({eval_split}, n={doc['n_queries']})", metrics))
    _log(folder, mlflow)
    return run_id


@app.command()
def baseline(
    retrievers: Annotated[
        str, typer.Option(help="comma list: bm25, bge-small, bge-base")
    ] = "bm25,bge-small,bge-base",
    split_name: Annotated[str, typer.Option("--split")] = "test",
    note: str = "",
    mlflow: Annotated[bool, typer.Option(help="log to the central MLflow")] = True,
) -> None:
    """M2: score the untouched retrievers, each ranking on its own."""
    from finance_rag_sft.harness import BM25Retriever, DenseRetriever

    for key in [r.strip() for r in retrievers.split(",") if r.strip()]:
        retriever = BM25Retriever() if key == "bm25" else DenseRetriever.base(key)
        _evaluate_and_write(retriever, "baseline", key, split_name, note, mlflow)


@app.command()
def check(mlflow: Annotated[bool, typer.Option()] = True) -> None:
    """M2 harness check: Phil Schmid's setup, to land near his base nDCG@10 of 0.768."""
    from datasets import Dataset

    from finance_rag_sft.data import read_rows
    from finance_rag_sft.harness import DenseRetriever, _unit, aggregate, score_matrix

    rows = read_rows()
    ds = Dataset.from_dict(
        {
            "anchor": [q for q, _ in rows],
            "positive": [c for _, c in rows],
            "id": list(range(len(rows))),
        }
    ).train_test_split(test_size=0.1, seed=42)
    rids = [f"r{i}" for i in range(len(rows))]
    test = ds["test"]
    qids = [f"r{i}" for i in test["id"]]
    retriever = DenseRetriever.base("bge-base")
    corpus = _unit(retriever._encode([c for _, c in rows]))
    queries = _unit(retriever._encode(list(test["anchor"])))
    results = score_matrix(queries @ corpus.T, rids, qids, {q: [q] for q in qids})
    m = aggregate(results)
    from finance_rag_sft.data import load_split

    s = load_split()
    run_id = runs.new_run_id("check", "phil-schmid-setup")
    doc = runs.run_doc(
        run_id=run_id,
        kind="check",
        retriever=retriever.describe(),
        split=s,
        eval_split=None,
        dims=["768"],
        n_queries=len(qids),
        metrics={"768": m},
        timing={},
        device=retriever.device,
        note="harness check, not split v1",
        check={
            "reference": 0.768,
            "got": round(m["ndcg@10"], 4),
            "metric": "ndcg@10 at 768",
            "note": "Phil Schmid's setup: random 90/10 rows (datasets seed 42), corpus = all 7,000 rows as separate documents, relevant = the question's own row only",
        },
    )
    folder = runs.write_run(doc)
    console.print(f"harness check: nDCG@10 {m['ndcg@10']:.4f} against Phil's 0.768")
    _log(folder, mlflow)


@app.command()
def train(
    config_path: Annotated[Path, typer.Option("--config")] = config.CONFIGS_DIR
    / "train_bge_base.yaml",
    max_steps: Annotated[int | None, typer.Option(help="stop early (smoke a training run)")] = None,
    note: str = "",
    mlflow: Annotated[bool, typer.Option(help="log and register in the central MLflow")] = True,
) -> None:
    """M3: fine-tune, score each epoch on val, choose one."""
    from finance_rag_sft.train import load_config
    from finance_rag_sft.train import train as run_train

    cfg = load_config(config_path)
    if max_steps:
        cfg.max_steps = max_steps
    doc = run_train(cfg, note=note)
    t = Table(title=f"{doc['run_id']}: val by epoch")
    for c in ("epoch", "step", "recall@10 768", "ndcg@10 768", "recall@10 64", "loss"):
        t.add_column(c, justify="right")
    for e in doc["train"]["epochs"]:
        v = e["val"]
        t.add_row(
            str(e["epoch"]),
            str(e["step"]),
            f"{v['768']['recall@10']:.4f}",
            f"{v['768']['ndcg@10']:.4f}",
            f"{v['64']['recall@10']:.4f}",
            f"{e['loss'] or 0:.4f}",
        )
    console.print(t)
    console.print(
        f"chosen: epoch {doc['train']['chosen']['epoch']} ({doc['train']['chosen']['by']}), {doc['train']['train_seconds']} s on {doc['device']}"
    )
    _log(runs.run_dir(doc["run_id"]), mlflow)


@app.command("eval")
def eval_cmd(
    train_run: Annotated[str, typer.Option(help="the train run whose chosen checkpoint to score")],
    split_name: Annotated[str, typer.Option("--split")] = "test",
    force: Annotated[bool, typer.Option(help="score test again for the same checkpoint")] = False,
    note: str = "",
    mlflow: Annotated[bool, typer.Option()] = True,
) -> None:
    """M4: score a tuned checkpoint. Test is scored once per checkpoint."""
    from finance_rag_sft.harness import DenseRetriever

    doc = runs.read_run(train_run)
    chosen = doc["train"]["chosen"]
    if split_name == "test" and not force:
        done = [
            r["run_id"]
            for r in runs.list_runs()
            if r["kind"] == "eval"
            and r["eval_split"] == "test"
            and (r.get("retriever") or {}).get("checkpoint") == chosen["checkpoint"]
        ]
        if done:
            raise typer.BadParameter(
                f"test already scored for {chosen['checkpoint']}: {done[0]} (--force to repeat)"
            )
    retriever = DenseRetriever.tuned(Path(chosen["checkpoint"]), train_run, round(chosen["epoch"]))
    _evaluate_and_write(
        retriever, "eval", f"bge-base-tuned-e{round(chosen['epoch'])}", split_name, note, mlflow
    )


@app.command("compare")
def compare_cmd(a: str, b: str, dim: str = "768") -> None:
    """Paired comparison: a against b on the same questions."""
    from finance_rag_sft.compare import compare_ranks

    da, db = runs.read_run(a), runs.read_run(b)
    if (da["split"], da["eval_split"]) != (db["split"], db["eval_split"]):
        raise typer.BadParameter("runs scored different splits")
    c = compare_ranks(runs.golden_ranks(a, dim), runs.golden_ranks(b, dim))
    console.print(f"[b]{da['label']}[/] vs [b]{db['label']}[/] at {dim}, n={c['n']}")
    console.print(
        f"fixed {c['fixed']}  broken {c['broken']}  both hit {c['both_hit']}  both miss {c['both_miss']}  exact McNemar p={c['p_exact']:.3g}"
    )
    for m, d in c["delta"].items():
        console.print(
            f"  delta {m}: {d['value']:+.4f}  (95% CI {d['ci_low']:+.4f} to {d['ci_high']:+.4f})"
        )


@app.command("runs")
def runs_cmd() -> None:
    """Every run, newest first."""
    t = Table()
    for c in ("run", "kind", "split", "recall@10 by dim"):
        t.add_column(c)
    for d in runs.list_runs():
        r10 = "  ".join(
            f"{dim}:{ms['recall@10']:.4f}" for dim, ms in (d.get("metrics") or {}).items()
        )
        t.add_row(d["run_id"], d["kind"], str(d.get("eval_split")), r10)
    console.print(t)


@app.command("log-run")
def log_run(folder: Path, register: bool = True) -> None:
    """Replay a run folder (e.g. one written on a GPU host) into the central MLflow."""
    from finance_rag_sft import tracking

    rid = tracking.log_run(folder, register=register)
    console.print(f"MLflow run {rid}  {tracking.run_url(rid) or ''}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8084, reload: bool = False) -> None:
    """The viewer's API (and the built frontend) on :8084."""
    import uvicorn

    uvicorn.run(
        "finance_rag_sft.serving.app:create_app", factory=True, host=host, port=port, reload=reload
    )


@app.command()
def doctor() -> None:
    """Device, torch build, GPU driver and MLflow reachability: run before training anywhere."""
    import torch

    from finance_rag_sft import tracking

    console.print(
        f"torch {torch.__version__}  cuda build {torch.version.cuda}  mps {torch.backends.mps.is_available()}"
    )
    if torch.cuda.is_available():
        console.print(f"cuda device {torch.cuda.get_device_name(0)}")
    else:
        console.print("cuda: not available")
    console.print(
        f"FINRAG_DEVICE={config.settings().device or '(auto)'}  env={config.settings().env}  sha={config.settings().code_sha}"
    )
    console.print(
        f"MLflow {config.settings().mlflow_tracking_uri}: {'up' if tracking.is_up() else 'DOWN'}"
    )


if __name__ == "__main__":
    app()
