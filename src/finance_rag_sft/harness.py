"""One harness for every retriever: rank the whole corpus, record each question's golden ranks.

The same `score_matrix` turns any (questions x passages) score matrix into per-question
results, so BM25, the base model and a tuned checkpoint are measured identically. Ranks are
1-based over the whole corpus; ties break by corpus order (sorted passage IDs).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from finance_rag_sft.config import BASE_MODELS, CACHE_DIR, ROOT, SMALL_DIM, settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from finance_rag_sft.data import Split

K = 10
METRICS = ("recall@1", "recall@5", "recall@10", "ndcg@10", "mrr@10")


@dataclass
class QueryResult:
    qid: str
    golden_ranks: list[int]  # every golden passage's rank, ascending
    top: list[tuple[str, float]]  # the top K (pid, score)
    golden: list[tuple[str, int, float]] | None = None  # (pid, rank, score), by rank

    @property
    def golden_rank(self) -> int:
        return self.golden_ranks[0]


def score_matrix(
    scores: np.ndarray,
    pids: Sequence[str],
    qids: Sequence[str],
    golden: dict[str, list[str]],
    k: int = K,
) -> list[QueryResult]:
    index = {p: i for i, p in enumerate(pids)}
    order = np.argsort(-scores, axis=1, kind="stable")
    out: list[QueryResult] = []
    for row, qid in enumerate(qids):
        pos = np.empty(len(pids), dtype=np.int64)
        pos[order[row]] = np.arange(1, len(pids) + 1)
        gold = sorted(
            ((p, int(pos[index[p]]), float(scores[row, index[p]])) for p in golden[qid]),
            key=lambda t: t[1],
        )
        top = [(pids[j], float(scores[row, j])) for j in order[row, :k]]
        out.append(QueryResult(qid=qid, golden_ranks=[r for _, r, _ in gold], top=top, golden=gold))
    return out


def per_query(ranks: Sequence[int], k: int = K) -> dict[str, float]:
    """One question's metrics from its golden ranks (ascending). recall@k = any golden in top k."""
    first = ranks[0]
    dcg = sum(1.0 / math.log2(r + 1) for r in ranks if r <= k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(ranks), k)))
    return {
        "recall@1": float(first <= 1),
        "recall@5": float(first <= 5),
        "recall@10": float(first <= 10),
        "ndcg@10": dcg / idcg,
        "mrr@10": 1.0 / first if first <= k else 0.0,
    }


def aggregate(results: Sequence[QueryResult]) -> dict[str, float]:
    rows = [per_query(r.golden_ranks) for r in results]
    return {m: float(np.mean([r[m] for r in rows])) for m in METRICS}


# ---- retrievers ---------------------------------------------------------------------------


class Retriever(Protocol):
    label: str

    def describe(self) -> dict[str, Any]: ...

    def dims(self) -> list[str]: ...

    def scores(self, split: Split, questions: Sequence[str], dim: str) -> np.ndarray: ...


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return np.asarray(x / np.clip(n, 1e-12, None), dtype=np.float32)


def load_st(name_or_path: str, revision: str | None = None) -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name_or_path, revision=revision, device=settings().device)
    # Training pads to multiples of 64 on MPS; scoring always pads to the batch's longest, so
    # base and tuned are encoded identically.
    model[0].processing_kwargs = {}
    return model


class DenseRetriever:
    """A Sentence Transformers model: the hub base at a pinned revision, or a checkpoint."""

    def __init__(
        self,
        name_or_path: str,
        *,
        revision: str | None,
        native_dim: int,
        label: str,
        checkpoint: str | None = None,
        train_run: str | None = None,
        key: str | None = None,
    ) -> None:
        self.name_or_path = name_or_path
        self.revision = revision
        self.native_dim = native_dim
        self.label = label
        self.checkpoint = checkpoint
        self.train_run = train_run
        self.key = key or name_or_path
        self._model: SentenceTransformer | None = None
        self.timing: dict[str, float] = {}

    @classmethod
    def base(cls, key: str) -> DenseRetriever:
        m = BASE_MODELS[key]
        return cls(m.hub_id, revision=m.revision, native_dim=m.dim, label=m.label, key=key)

    @classmethod
    def tuned(cls, checkpoint: Path, train_run: str, epoch: int) -> DenseRetriever:
        return cls(
            str(checkpoint),
            revision=None,
            native_dim=BASE_MODELS["bge-base"].dim,
            label=f"bge-base tuned (epoch {epoch})",
            checkpoint=str(checkpoint),
            train_run=train_run,
            key=f"tuned:{train_run}",
        )

    def describe(self) -> dict[str, Any]:
        base = self.checkpoint is None
        return {
            "type": "dense",
            "model": self.name_or_path if base else BASE_MODELS["bge-base"].hub_id,
            "revision": self.revision if base else BASE_MODELS["bge-base"].revision,
            "checkpoint": self.checkpoint,
            "train_run": self.train_run,
            "label": self.label,
        }

    def dims(self) -> list[str]:
        return [str(self.native_dim), str(SMALL_DIM)]

    @property
    def path(self) -> str:
        """Where to load from: a hub ID, or a checkpoint resolved against the repo root."""
        if self.checkpoint and not Path(self.checkpoint).is_absolute():
            return str(ROOT / self.checkpoint)
        return self.name_or_path

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = load_st(self.path, self.revision)
        return self._model

    @property
    def device(self) -> str:
        return str(self.model.device).split(":")[0]

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        out = self.model.encode(
            list(texts), batch_size=64, convert_to_numpy=True, show_progress_bar=False
        )
        return np.asarray(out, dtype=np.float32)

    def _cache_key(self, split: Split) -> str:
        ident = f"{self.name_or_path}|{self.revision}|{split.version}|{split.fingerprint}"
        if self.checkpoint:
            weights = Path(self.path) / "model.safetensors"
            if weights.exists():
                ident += f"|{weights.stat().st_size}|{weights.stat().st_mtime_ns}"
        return hashlib.sha1(ident.encode()).hexdigest()[:16]

    def corpus_cache(self, split: Split) -> Path:
        return CACHE_DIR / "emb" / f"{self._cache_key(split)}.npy"

    def corpus_embeddings(self, split: Split) -> np.ndarray:
        path = self.corpus_cache(split)
        if path.exists():
            return np.asarray(np.load(path), dtype=np.float32)
        t0 = time.time()
        emb = self._encode([split.passage_text[p] for p in split.pids])
        self.timing["encode_corpus"] = round(time.time() - t0, 1)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, emb)
        path.with_suffix(".json").write_text(
            json.dumps(
                {"model": self.name_or_path, "revision": self.revision, "split": split.version}
            )
        )
        return emb

    def scores(self, split: Split, questions: Sequence[str], dim: str) -> np.ndarray:
        d = int(dim)
        corpus = _unit(self.corpus_embeddings(split)[:, :d])
        t0 = time.time()
        queries = _unit(self._encode(questions)[:, :d])
        self.timing[f"encode_queries_{dim}"] = round(time.time() - t0, 2)
        return np.asarray(queries @ corpus.T)


class BM25Retriever:
    """bm25s over the corpus, English stopwords and the Snowball stemmer, as vidore-v3-loop."""

    label = "BM25"
    key = "bm25"

    def __init__(self) -> None:
        self._index: tuple[str, Any, Any] | None = None  # (split fingerprint, retriever, stemmer)
        self.timing: dict[str, float] = {}
        self.device = "cpu"

    def describe(self) -> dict[str, Any]:
        return {
            "type": "bm25",
            "model": None,
            "revision": None,
            "checkpoint": None,
            "train_run": None,
            "label": self.label,
        }

    def dims(self) -> list[str]:
        return ["lexical"]

    def _built(self, split: Split) -> tuple[Any, Any]:
        if self._index is None or self._index[0] != split.fingerprint:
            import bm25s
            import Stemmer

            stemmer = Stemmer.Stemmer("english")
            tokens = bm25s.tokenize(
                [split.passage_text[p] for p in split.pids],
                stopwords="en",
                stemmer=stemmer,
                show_progress=False,
            )
            index = bm25s.BM25()
            index.index(tokens, show_progress=False)
            self._index = (split.fingerprint, index, stemmer)
        return self._index[1], self._index[2]

    def scores(self, split: Split, questions: Sequence[str], dim: str = "lexical") -> np.ndarray:
        import bm25s

        index, stemmer = self._built(split)
        tokenized = bm25s.tokenize(
            list(questions), stopwords="en", stemmer=stemmer, return_ids=False, show_progress=False
        )
        empty = np.zeros(len(split.pids), dtype=np.float32)  # every word a stopword
        rows = [
            np.asarray(index.get_scores(list(toks)), dtype=np.float32) if toks else empty
            for toks in tokenized
        ]
        return np.vstack(rows) if rows else np.zeros((0, len(split.pids)), dtype=np.float32)


def evaluate(
    retriever: DenseRetriever | BM25Retriever, split: Split, eval_split: str
) -> tuple[dict[str, dict[str, float]], dict[str, list[QueryResult]], dict[str, float]]:
    """Score `eval_split`'s questions at every dim. Returns (metrics, results, timing)."""
    qids = split.qids(eval_split)
    questions = [split.query_text[q] for q in qids]
    metrics: dict[str, dict[str, float]] = {}
    results: dict[str, list[QueryResult]] = {}
    for dim in retriever.dims():
        t0 = time.time()
        s = retriever.scores(split, questions, dim)
        results[dim] = score_matrix(s, split.pids, qids, split.golden)
        metrics[dim] = aggregate(results[dim])
        retriever.timing[f"score_{dim}"] = round(time.time() - t0, 2)
    return metrics, results, dict(retriever.timing)
