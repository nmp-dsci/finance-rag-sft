"""The Method page's worked example: what MultipleNegativesRankingLoss sees for one batch.

Training never stores this view, so it is recomputed here from real pairs and the harness's own
scores (the same truncation and renormalisation that scored test). The loss is Sentence
Transformers v6's MNRL with its defaults: query to document only, cosine similarity, scale 20:

    loss_i = -log( exp(20 * s_ii) / sum_j exp(20 * s_ij) )

MatryoshkaLoss sums that loss, weight 1 each, over the first 768, 512, 256, 128 and 64 dims.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from finance_rag_sft.data import Split

MNRL_SCALE = 20.0
MATRYOSHKA_DIMS = ("768", "512", "256", "128", "64")


class Scorer(Protocol):
    label: str

    def scores(self, split: Split, questions: Sequence[str], dim: str) -> np.ndarray: ...


def sample_pairs(split: Split, eval_split: str, n: int, seed: int) -> list[tuple[str, str]]:
    """n (qid, pid) pairs with no question or passage twice, as the NO_DUPLICATES sampler builds."""
    pairs = [(q, p) for q in split.qids(eval_split) for p in split.golden[q]]
    random.Random(seed).shuffle(pairs)
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for q, p in pairs:
        if q in seen or p in seen:
            continue
        out.append((q, p))
        seen |= {q, p}
        if len(out) == n:
            break
    return out


def in_batch(sims: np.ndarray, scale: float = MNRL_SCALE) -> tuple[np.ndarray, np.ndarray]:
    """Per question: the softmax probability on its own passage, and MNRL's loss (-log of it)."""
    logits = np.asarray(sims, dtype=np.float64) * scale
    log_z = np.logaddexp.reduce(logits, axis=1)
    log_p = np.diag(logits) - log_z
    return np.exp(log_p), -log_p


def _rank(row: np.ndarray, j: int) -> int:
    """1-based rank of column j in a score row, ties broken by corpus order (as score_matrix)."""
    return int((row > row[j]).sum() + (row[:j] == row[j]).sum() + 1)


def batch_view(
    split: Split,
    scorers: Sequence[tuple[str, Scorer]],
    eval_split: str = "train",
    n: int = 4,
    seed: int = 0,
    dim: str = "768",
) -> dict[str, Any]:
    pairs = sample_pairs(split, eval_split, n, seed)
    questions = [split.query_text[q] for q, _ in pairs]
    col = {p: i for i, p in enumerate(split.pids)}
    cols = [col[p] for _, p in pairs]
    models = []
    for rid, scorer in scorers:
        by_dim = {d: scorer.scores(split, questions, d) for d in MATRYOSHKA_DIMS}
        full = by_dim[dim]
        sims = full[:, cols]
        p_correct, loss = in_batch(sims)
        per_dim = {d: float(in_batch(by_dim[d][:, cols])[1].mean()) for d in MATRYOSHKA_DIMS}
        models.append(
            {
                "retriever": rid,
                "label": scorer.label,
                "sims": np.round(sims, 4).tolist(),
                "p_correct": np.round(p_correct, 4).tolist(),
                "loss": np.round(loss, 4).tolist(),
                "mean_loss": round(float(loss.mean()), 4),
                "corpus_rank": [_rank(full[i], c) for i, c in enumerate(cols)],
                "matryoshka": [{"dim": d, "mean_loss": round(v, 4)} for d, v in per_dim.items()],
                "matryoshka_loss": round(sum(per_dim.values()), 4),
            }
        )
    return {
        "split": eval_split,
        "n": len(pairs),
        "seed": seed,
        "dim": dim,
        "scale": MNRL_SCALE,
        "corpus": len(split.pids),
        "pairs": [
            {"qid": q, "question": split.query_text[q], "pid": p, "passage": split.passage_text[p]}
            for q, p in pairs
        ],
        "models": models,
    }
