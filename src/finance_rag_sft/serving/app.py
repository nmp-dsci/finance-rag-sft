"""The viewer's API (contract: docs/api.md), and the built frontend.

Reads split v1, the pinned dataset and run folders. Never queries MLflow for data; it links
to each run there. Live search loads each retriever once and reuses cached corpus embeddings.
"""

from __future__ import annotations

import threading
import time
from functools import cache
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel as Body

from finance_rag_sft import config, runs, tracking
from finance_rag_sft.compare import compare_ranks
from finance_rag_sft.data import Split, load_split, split_stats

BASE_HUB = config.BASE_MODELS["bge-base"].hub_id


class SearchBody(Body):
    query: str
    retrievers: list[str]
    dim: str = "768"
    k: int = 10


@cache
def _mlflow_state(bucket: int) -> str:  # cached per 30 s bucket
    return "up" if tracking.is_up() else "down"


def _with_url(doc: dict[str, Any]) -> dict[str, Any]:
    url = (
        tracking.run_url(doc.get("mlflow_run_id"))
        if _mlflow_state(int(time.time() // 30)) == "up"
        else None
    )
    return {**doc, "mlflow_url": url}


def _summary(doc: dict[str, Any]) -> dict[str, Any]:
    return _with_url(runs.summary(doc))


def _is_base(doc: dict[str, Any]) -> bool:
    r = doc.get("retriever") or {}
    return r.get("model") == BASE_HUB and not r.get("checkpoint")


def _latest(kind: str, pred: Any) -> dict[str, Any] | None:
    for d in runs.list_runs():
        if d["kind"] == kind and d.get("eval_split") == "test" and pred(d):
            return d
    return None


def _hit(
    split: Split, rank: int, pid: str, score: float | None, golden: set[str]
) -> dict[str, Any]:
    return {
        "rank": rank,
        "pid": pid,
        "text": split.passage_text.get(pid, ""),
        "score": score,
        "golden": pid in golden,
    }


def create_app() -> FastAPI:
    app = FastAPI(title="finance-rag-sft", version="0.1.0")
    lock = threading.Lock()
    retrievers: dict[str, Any] = {}

    def split() -> Split:
        try:
            return load_split()
        except FileNotFoundError as e:
            raise HTTPException(503, str(e)) from e

    def run_or_404(run_id: str) -> dict[str, Any]:
        try:
            return runs.read_run(run_id)
        except FileNotFoundError as e:
            raise HTTPException(404, f"no run {run_id}") from e

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "code_sha": config.settings().code_sha,
            "mlflow": _mlflow_state(int(time.time() // 30)),
        }

    # ---- overview ---------------------------------------------------------------------

    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        try:
            s: Split | None = load_split()
        except FileNotFoundError:
            s = None
        base = _latest("baseline", _is_base)
        tuned = _latest("eval", lambda d: bool((d.get("retriever") or {}).get("checkpoint")))
        check = next((d for d in runs.list_runs() if d["kind"] == "check"), None)
        train = next((d for d in runs.list_runs() if d["kind"] == "train"), None)
        cmp = None
        if base and tuned:
            c = compare_ranks(
                runs.golden_ranks(tuned["run_id"], "768"), runs.golden_ranks(base["run_id"], "768")
            )
            cmp = {
                "a": _summary(tuned),
                "b": _summary(base),
                "dim": "768",
                **{k: v for k, v in c.items() if not k.endswith("_qids")},
            }

        def r10(d: dict[str, Any] | None, dim: str = "768") -> str:
            return f"{d['metrics'][dim]['recall@10']:.3f}" if d else ""

        milestones = [
            {
                "id": "M1",
                "title": "Frozen split v1",
                "status": "done" if s else "pending",
                "detail": f"{s.manifest['queries']['test']} test questions over {len(s.pids)} passages"
                if s
                else "run `finrag split`",
            },
            {
                "id": "M2",
                "title": "Baseline",
                "status": "done" if base else "pending",
                "detail": f"bge-base recall@10 {r10(base)} at 768"
                if base
                else "run `finrag baseline`",
            },
            {
                "id": "M2",
                "title": "Harness check",
                "status": "done" if check else "pending",
                "detail": f"nDCG@10 {check['check']['got']} against Phil Schmid's 0.768"
                if check
                else "run `finrag check`",
            },
            {
                "id": "M3",
                "title": "Fine-tune",
                "status": "done" if train else "pending",
                "detail": f"chosen epoch {train['train']['chosen']['epoch']} on val"
                if train
                else "run `finrag train`",
            },
            {
                "id": "M4",
                "title": "Test once and compare",
                "status": "done" if tuned else "pending",
                "detail": f"tuned recall@10 {r10(tuned)} at 768" if tuned else "run `finrag eval`",
            },
        ]
        baselines = [
            _summary(d)
            for d in runs.list_runs()
            if d["kind"] == "baseline" and d.get("eval_split") == "test"
        ]
        return {
            "project": config.PROJECT,
            "code_sha": config.settings().code_sha,
            "split": {
                "version": s.version,
                "passages": len(s.pids),
                "queries": {k: s.manifest["queries"][k] for k in ("train", "val", "test")},
            }
            if s
            else None,
            "base": _summary(base) if base else None,
            "tuned": _summary(tuned) if tuned else None,
            "compare": cmp,
            "baselines": baselines,
            "milestones": milestones,
        }

    # ---- split / data explorer --------------------------------------------------------

    @app.get("/api/split")
    def get_split() -> dict[str, Any]:
        return split_stats(split())

    @app.get("/api/split/queries")
    def split_queries(
        split_name: str = Query("all", alias="split"),
        q: str = "",
        offset: int = 0,
        limit: int = Query(50, le=500),
    ) -> dict[str, Any]:
        s = split()
        needle = q.lower().strip()
        items = [
            {
                "qid": qid,
                "split": s.query_split[qid],
                "text": s.query_text[qid],
                "golden": s.golden[qid],
            }
            for qid in s.qids(split_name)
            if not needle or needle in s.query_text[qid].lower()
        ]
        return {"total": len(items), "items": items[offset : offset + limit]}

    @app.get("/api/split/queries/{qid}")
    def split_query(qid: str) -> dict[str, Any]:
        s = split()
        if qid not in s.query_text:
            raise HTTPException(404, f"no question {qid}")
        scored = []
        for d in runs.list_runs():
            row = runs.read_queries(d["run_id"]).get(qid)
            if row:
                for dim, r in row["ranks"].items():
                    scored.append(
                        {
                            "run_id": d["run_id"],
                            "label": d["label"],
                            "dim": dim,
                            "golden_rank": r["golden_ranks"][0],
                        }
                    )
        return {
            "qid": qid,
            "split": s.query_split[qid],
            "text": s.query_text[qid],
            "golden": [
                {"pid": p, "text": s.passage_text[p], "n_questions": len(s.questions_of.get(p, []))}
                for p in s.golden[qid]
            ],
            "runs": scored,
        }

    @app.get("/api/split/passages")
    def split_passages(
        q: str = "",
        split_name: str = Query("all", alias="split"),
        offset: int = 0,
        limit: int = Query(50, le=500),
    ) -> dict[str, Any]:
        s = split()
        needle = q.lower().strip()
        items = [
            {
                "pid": p,
                "split": s.passage_split[p],
                "text": s.passage_text[p],
                "words": len(s.passage_text[p].split()),
                "n_questions": len(s.questions_of.get(p, [])),
            }
            for p in s.pids
            if split_name in ("all", s.passage_split[p])
            and (not needle or needle in s.passage_text[p].lower())
        ]
        return {"total": len(items), "items": items[offset : offset + limit]}

    @app.get("/api/split/passages/{pid}")
    def split_passage(pid: str) -> dict[str, Any]:
        s = split()
        if pid not in s.passage_text:
            raise HTTPException(404, f"no passage {pid}")
        return {
            "pid": pid,
            "split": s.passage_split[pid],
            "text": s.passage_text[pid],
            "words": len(s.passage_text[pid].split()),
            "rows": s.passage_rows[pid],
            "questions": [
                {"qid": q, "split": s.query_split[q], "text": s.query_text[q]}
                for q in s.questions_of.get(pid, [])
            ],
        }

    # ---- runs -------------------------------------------------------------------------

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        return [_summary(d) for d in runs.list_runs()]

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        return _with_url(run_or_404(run_id))

    def _dim(doc: dict[str, Any], dim: str | None) -> str:
        dims: list[str] = doc.get("dims") or []
        if not dims:
            raise HTTPException(400, f"{doc['run_id']} has no per-question results")
        if dim is None:
            return dims[0]
        if dim not in dims:
            raise HTTPException(400, f"{doc['run_id']} has dims {dims}, not {dim}")
        return dim

    def _own_dim(doc: dict[str, Any], dim: str) -> str:
        """The other side of a comparison uses `dim` when it has it, else its full size:
        BM25 has only "lexical", bge-small has 384 where bge-base has 768."""
        dims: list[str] = doc.get("dims") or []
        if not dims:
            raise HTTPException(400, f"{doc['run_id']} has no per-question results")
        return dim if dim in dims else dims[0]

    def _vs(doc: dict[str, Any], vs: str, dim: str) -> tuple[dict[str, Any], str]:
        other = run_or_404(vs)
        if (other["split"], other["eval_split"]) != (doc["split"], doc["eval_split"]):
            raise HTTPException(400, "runs scored different splits")
        return other, _own_dim(other, dim)

    @app.get("/api/runs/{run_id}/queries")
    def run_queries(
        run_id: str,
        dim: str | None = None,
        filter_name: str = Query("all", alias="filter"),
        q: str = "",
        offset: int = 0,
        limit: int = Query(100, le=1000),
        vs: str | None = None,
    ) -> dict[str, Any]:
        doc = run_or_404(run_id)
        d = _dim(doc, dim)
        s = split()
        mine = runs.read_queries(run_id)
        theirs, vd = (runs.read_queries(vs), _vs(doc, vs, d)[1]) if vs else (None, d)
        if filter_name in ("fixed", "broken") and theirs is None:
            raise HTTPException(400, f"filter={filter_name} needs vs=<run>")
        needle = q.lower().strip()
        items = []
        for qid, row in mine.items():
            text = s.query_text.get(qid, qid)
            if needle and needle not in text.lower():
                continue
            rank = row["ranks"][d]["golden_ranks"][0]
            item: dict[str, Any] = {
                "qid": qid,
                "text": text,
                "golden_rank": rank,
                "hit": rank <= 10,
            }
            if theirs is not None:
                other = theirs[qid]["ranks"][vd]["golden_ranks"][0]
                a, b = rank <= 10, other <= 10
                item["vs_golden_rank"] = other
                item["change"] = (
                    "fixed"
                    if a and not b
                    else "broken"
                    if b and not a
                    else "same_hit"
                    if a
                    else "same_miss"
                )
            keep = {
                "all": True,
                "hit": item["hit"],
                "miss": not item["hit"],
                "fixed": item.get("change") == "fixed",
                "broken": item.get("change") == "broken",
            }.get(filter_name)
            if keep is None:
                raise HTTPException(400, f"unknown filter {filter_name}")
            if keep:
                items.append(item)
        out: dict[str, Any] = {
            "total": len(items),
            "dim": d,
            "items": items[offset : offset + limit],
        }
        if vs:
            out["vs_dim"] = vd
        return out

    def _block(
        doc: dict[str, Any], row: dict[str, Any], dim: str, s: Split, golden: set[str]
    ) -> dict[str, Any]:
        r = row["ranks"][dim]
        top = [_hit(s, i + 1, p, sc, golden) for i, (p, sc) in enumerate(r["top"])]
        outside = [
            _hit(s, rank, p, sc, golden) for p, rank, sc in r.get("golden", []) if rank > len(top)
        ]
        return {
            "run_id": doc["run_id"],
            "label": doc["label"],
            "golden_rank": r["golden_ranks"][0],
            "top": top,
            "golden_hits": outside,
        }

    @app.get("/api/runs/{run_id}/queries/{qid}")
    def run_query(
        run_id: str, qid: str, dim: str | None = None, vs: str | None = None
    ) -> dict[str, Any]:
        doc = run_or_404(run_id)
        d = _dim(doc, dim)
        s = split()
        row = runs.read_queries(run_id).get(qid)
        if row is None:
            raise HTTPException(404, f"{run_id} did not score {qid}")
        golden = set(s.golden[qid])
        vs_block = None
        if vs:
            other, vd = _vs(doc, vs, d)
            orow = runs.read_queries(vs).get(qid)
            if orow is not None:
                vs_block = _block(other, orow, vd, s, golden) | {"dim": vd}
        return {
            "qid": qid,
            "text": s.query_text[qid],
            "split": s.query_split[qid],
            "golden": [{"pid": p, "text": s.passage_text[p]} for p in s.golden[qid]],
            "dim": d,
            "run": _block(doc, row, d, s, golden),
            "vs": vs_block,
        }

    @app.get("/api/compare")
    def compare(a: str, b: str, dim: str = "768") -> dict[str, Any]:
        da, db = run_or_404(a), run_or_404(b)
        if (da["split"], da["eval_split"]) != (db["split"], db["eval_split"]):
            raise HTTPException(400, "runs scored different splits")
        dim_a, dim_b = _own_dim(da, dim), _own_dim(db, dim)
        c = compare_ranks(runs.golden_ranks(a, dim_a), runs.golden_ranks(b, dim_b))
        return {
            "a": _summary(da),
            "b": _summary(db),
            "dim": dim_a,
            "dim_b": dim_b,
            **{k: v for k, v in c.items() if not k.endswith("_qids")},
        }

    # ---- live retrieval -----------------------------------------------------------------

    def _catalogue() -> dict[str, dict[str, Any]]:
        from finance_rag_sft.harness import BM25Retriever, DenseRetriever

        out: dict[str, dict[str, Any]] = {
            "bm25": {"make": BM25Retriever, "label": "BM25", "dims": ["lexical"]},
        }
        for key, m in config.BASE_MODELS.items():
            out[key] = {
                "make": lambda key=key: DenseRetriever.base(key),
                "label": m.label,
                "dims": [str(m.dim), str(config.SMALL_DIM)],
            }
        for d in runs.list_runs():
            if d["kind"] == "train":
                ch = d["train"]["chosen"]
                if (config.ROOT / ch["checkpoint"]).exists():
                    from pathlib import Path

                    out[f"tuned:{d['run_id']}"] = {
                        "make": lambda d=d, ch=ch: DenseRetriever.tuned(
                            Path(ch["checkpoint"]), d["run_id"], round(ch["epoch"])
                        ),
                        "label": f"bge-base tuned (epoch {round(ch['epoch'])})",
                        "dims": ["768", str(config.SMALL_DIM)],
                    }
        return out

    def _retriever(rid: str) -> Any:
        cat = _catalogue()
        if rid not in cat:
            raise HTTPException(400, f"unknown retriever {rid}")
        with lock:
            if rid not in retrievers:
                retrievers[rid] = cat[rid]["make"]()
            return retrievers[rid]

    @app.get("/api/retrievers")
    def list_retrievers() -> list[dict[str, Any]]:
        s = split()
        out = []
        for rid, c in _catalogue().items():
            ready = True
            if rid != "bm25":
                ready = c["make"]().corpus_cache(s).exists()
            out.append({"id": rid, "label": c["label"], "dims": c["dims"], "ready": ready})
        return out

    @app.post("/api/search")
    def search(body: SearchBody) -> dict[str, Any]:
        from finance_rag_sft.data import qid_of

        s = split()
        if not body.query.strip():
            raise HTTPException(400, "empty query")
        qid = qid_of(body.query)
        matched = qid if qid in s.query_text else None
        golden = set(s.golden[matched]) if matched else set()
        results = []
        for rid in body.retrievers:
            r = _retriever(rid)
            dim = "lexical" if rid == "bm25" else body.dim
            if dim not in r.dims():
                dim = r.dims()[0]
            t0 = time.time()
            with lock:
                scores = r.scores(s, [body.query], dim)[0]
            order = scores.argsort()[::-1][: body.k]
            ms = round((time.time() - t0) * 1000, 1)
            top = [_hit(s, i + 1, s.pids[j], float(scores[j]), golden) for i, j in enumerate(order)]
            results.append({"retriever": rid, "label": r.label, "dim": dim, "ms": ms, "top": top})
        return {
            "query": body.query,
            "matched_qid": matched,
            "golden": sorted(golden),
            "results": results,
        }

    _mount_frontend(app)
    return app


def _mount_frontend(app: FastAPI) -> None:
    dist = config.FRONTEND_DIST
    if not (dist / "index.html").exists():
        return
    if (dist / "assets").exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        target = dist / path
        if path and target.is_file():
            return FileResponse(target)
        return FileResponse(dist / "index.html", headers={"Cache-Control": "no-cache"})
