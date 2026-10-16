"""Paired comparison of two runs over the same questions: McNemar on hits, bootstrap on deltas."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from finance_rag_sft.harness import K, per_query

DELTA_METRICS = ("recall@10", "ndcg@10", "mrr@10")
BUCKETS = (
    ("up 10+", 10, 10**9),
    ("up 4-9", 4, 9),
    ("up 1-3", 1, 3),
    ("same", 0, 0),
    ("down 1-3", -3, -1),
    ("down 4-9", -9, -4),
    ("down 10+", -(10**9), -10),
)


def mcnemar_exact(fixed: int, broken: int) -> float:
    """Two-sided exact McNemar: a binomial test of `fixed` among the discordant pairs."""
    from scipy.stats import binomtest

    n = fixed + broken
    return 1.0 if n == 0 else float(binomtest(fixed, n, 0.5).pvalue)


def compare_ranks(
    a: Mapping[str, Sequence[int]],
    b: Mapping[str, Sequence[int]],
    k: int = K,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """`a` and `b` map qid -> golden ranks (ascending). Deltas are a - b; "fixed" = a hits, b misses."""
    qids = sorted(a)
    if qids != sorted(b):
        raise ValueError("the two runs scored different questions")
    a_hit = np.array([a[q][0] <= k for q in qids])
    b_hit = np.array([b[q][0] <= k for q in qids])
    fixed = int(np.sum(a_hit & ~b_hit))
    broken = int(np.sum(~a_hit & b_hit))
    pa = [per_query(a[q], k) for q in qids]
    pb = [per_query(b[q], k) for q in qids]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(qids), size=(n_boot, len(qids)))
    delta: dict[str, dict[str, float]] = {}
    for m in DELTA_METRICS:
        diff = np.array([x[m] - y[m] for x, y in zip(pa, pb, strict=True)])
        boots = diff[idx].mean(axis=1)
        delta[m] = {
            "value": float(diff.mean()),
            "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)),
        }
    shift = [b[q][0] - a[q][0] for q in qids]  # positive: a ranks the golden passage higher
    rank_shift = [
        {"bucket": name, "count": sum(1 for s in shift if lo <= s <= hi)}
        for name, lo, hi in BUCKETS
    ]
    return {
        "n": len(qids),
        "fixed": fixed,
        "broken": broken,
        "both_hit": int(np.sum(a_hit & b_hit)),
        "both_miss": int(np.sum(~a_hit & ~b_hit)),
        "p_exact": mcnemar_exact(fixed, broken),
        "delta": delta,
        "rank_shift": rank_shift,
        "fixed_qids": [q for q, x, y in zip(qids, a_hit, b_hit, strict=True) if x and not y],
        "broken_qids": [q for q, x, y in zip(qids, a_hit, b_hit, strict=True) if y and not x],
    }
