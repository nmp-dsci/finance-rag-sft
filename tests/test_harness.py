from __future__ import annotations

import math

import numpy as np
import pytest

from finance_rag_sft.compare import compare_ranks, mcnemar_exact
from finance_rag_sft.harness import aggregate, per_query, score_matrix


def test_score_matrix_ranks_whole_corpus_and_breaks_ties_by_order() -> None:
    pids = ["p1", "p2", "p3", "p4"]
    scores = np.array([[0.1, 0.9, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]])
    golden = {"qa": ["p4"], "qb": ["p3", "p1"]}
    res = score_matrix(scores, pids, ["qa", "qb"], golden, k=2)
    assert res[0].golden_ranks == [3]  # p2, p3 (tie, earlier), p4
    assert [p for p, _ in res[0].top] == ["p2", "p3"]
    assert res[1].golden_ranks == [1, 3]


def test_per_query_metrics() -> None:
    m = per_query([3])
    assert m["recall@1"] == 0 and m["recall@5"] == 1 and m["recall@10"] == 1
    assert m["mrr@10"] == pytest.approx(1 / 3)
    assert m["ndcg@10"] == pytest.approx(1 / math.log2(4))
    assert per_query([11])["recall@10"] == 0 and per_query([11])["ndcg@10"] == 0
    two = per_query([1, 2])
    assert two["ndcg@10"] == pytest.approx(1.0)


def test_aggregate_matches_hand_average() -> None:
    pids = ["p1", "p2"]
    res = score_matrix(np.array([[1.0, 0.0], [1.0, 0.0]]), pids, ["a", "b"], {"a": ["p1"], "b": ["p2"]})
    assert aggregate(res)["recall@1"] == 0.5


def test_compare_counts_and_mcnemar() -> None:
    a = {"q1": [1], "q2": [2], "q3": [30], "q4": [5]}
    b = {"q1": [20], "q2": [15], "q3": [3], "q4": [5]}
    c = compare_ranks(a, b, n_boot=200)
    assert (c["fixed"], c["broken"], c["both_hit"], c["both_miss"]) == (2, 1, 1, 0)
    assert c["fixed_qids"] == ["q1", "q2"] and c["broken_qids"] == ["q3"]
    assert c["delta"]["recall@10"]["value"] == pytest.approx(0.25)
    assert sum(r["count"] for r in c["rank_shift"]) == 4
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) < 0.01


def test_compare_refuses_different_questions() -> None:
    with pytest.raises(ValueError):
        compare_ranks({"q1": [1]}, {"q2": [1]})
