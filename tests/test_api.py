from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from finance_rag_sft import config, runs
from finance_rag_sft.data import build_split, load_split_from
from finance_rag_sft.harness import aggregate, score_matrix


@pytest.fixture
def client(
    rows: list[tuple[str, str]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    build_split(rows, tmp_path / "splits" / "v1")
    split = load_split_from(rows, tmp_path / "splits" / "v1")
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "FRONTEND_DIST", tmp_path / "no-dist")
    from finance_rag_sft.serving import app as app_mod

    monkeypatch.setattr(app_mod, "load_split", lambda: split)
    monkeypatch.setattr(app_mod.tracking, "is_up", lambda: False)
    qids = split.qids("test") or split.qids("all")[:3]
    rng = np.random.default_rng(0)

    def write(kind: str, label: str, retriever: dict[str, Any], seed_shift: float) -> str:
        scores = rng.random((len(qids), len(split.pids))) + seed_shift
        results = {"768": score_matrix(scores, split.pids, qids, split.golden)}
        results["64"] = results["768"]
        run_id = runs.new_run_id(kind, label) + f"-{seed_shift}"
        doc = runs.run_doc(
            run_id=run_id,
            kind=kind,
            retriever=retriever,
            split=split,
            eval_split="test",
            dims=["768", "64"],
            n_queries=len(qids),
            metrics={d: aggregate(r) for d, r in results.items()},
            timing={},
            device="cpu",
        )
        runs.write_run(doc, results)
        return run_id

    base = {
        "type": "dense",
        "model": config.BASE_MODELS["bge-base"].hub_id,
        "revision": "x",
        "checkpoint": None,
        "train_run": None,
        "label": "bge-base (base)",
    }
    tuned = base | {"checkpoint": "outputs/t/checkpoint-1", "train_run": "t", "label": "tuned"}
    c = TestClient(app_mod.create_app())
    c.base_id = write("baseline", "bge-base", base, 0.0)  # type: ignore[attr-defined]
    c.tuned_id = write("eval", "tuned", tuned, 0.5)  # type: ignore[attr-defined]
    return c


def test_health_and_overview(client: TestClient) -> None:
    assert client.get("/healthz").json()["ok"] is True
    o = client.get("/api/overview").json()
    assert o["base"]["run_id"] == client.base_id  # type: ignore[attr-defined]
    assert o["tuned"]["run_id"] == client.tuned_id  # type: ignore[attr-defined]
    assert o["compare"]["n"] == o["base"]["n_queries"]
    assert {m["id"] for m in o["milestones"]} >= {"M1", "M2", "M3", "M4"}


def test_split_explorer(client: TestClient) -> None:
    s = client.get("/api/split").json()
    assert s["leak_check"]["ok"] and s["passages"] > 0
    qs = client.get("/api/split/queries", params={"split": "all", "q": "risks"}).json()
    assert qs["total"] == 1 and len(qs["items"][0]["golden"]) == 2
    q = client.get(f"/api/split/queries/{qs['items'][0]['qid']}").json()
    assert len(q["golden"]) == 2
    ps = client.get("/api/split/passages", params={"q": "CEO"}).json()
    p = client.get(f"/api/split/passages/{ps['items'][0]['pid']}").json()
    assert p["rows"] == 2 and len(p["questions"]) == 1


def test_runs_queries_and_compare(client: TestClient) -> None:
    tuned, base = client.tuned_id, client.base_id  # type: ignore[attr-defined]
    assert len(client.get("/api/runs").json()) == 2
    rq = client.get(f"/api/runs/{tuned}/queries", params={"vs": base, "dim": "768"}).json()
    assert rq["total"] >= 1 and "change" in rq["items"][0]
    qid = rq["items"][0]["qid"]
    one = client.get(f"/api/runs/{tuned}/queries/{qid}", params={"vs": base}).json()
    assert len(one["run"]["top"]) == min(10, one["run"]["golden_rank"] + 10) or one["run"]["top"]
    assert one["vs"]["run_id"] == base
    cmp = client.get("/api/compare", params={"a": tuned, "b": base}).json()
    assert cmp["fixed"] + cmp["broken"] + cmp["both_hit"] + cmp["both_miss"] == cmp["n"]
    assert client.get(f"/api/runs/{tuned}/queries", params={"filter": "fixed"}).status_code == 400
    assert client.get("/api/runs/nope").status_code == 404


def test_method_batch_uses_the_harness_scores(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from finance_rag_sft.harness import DenseRetriever

    def fake_scores(self: DenseRetriever, split: Any, questions: Any, dim: str) -> np.ndarray:
        return np.random.default_rng(int(dim)).random((len(questions), len(split.pids)))

    monkeypatch.setattr(DenseRetriever, "scores", fake_scores)
    v = client.get("/api/method/batch", params={"n": 3, "dim": "64"}).json()
    assert v["split"] == "train" and v["n"] == 3 and v["dim"] == "64" and v["scale"] == 20
    assert [m["retriever"] for m in v["models"]] == ["bge-base"]  # no tuned checkpoint on disk
    assert len(v["models"][0]["sims"]) == 3 and len(v["models"][0]["matryoshka"]) == 5
    assert client.get("/api/method/batch", params={"split": "test"}).status_code == 422
    assert client.get("/api/method/batch", params={"n": 40}).status_code == 422
