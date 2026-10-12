from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from finance_rag_sft.data import Split, build_split, load_split_from
from finance_rag_sft.explain import MATRYOSHKA_DIMS, batch_view, in_batch, sample_pairs


@pytest.fixture
def split(rows: list[tuple[str, str]], tmp_path: Path) -> Split:
    build_split(rows, tmp_path / "v1")
    return load_split_from(rows, tmp_path / "v1")


class FakeScorer:
    """Deterministic scores: the question's own golden passages score highest."""

    label = "fake"

    def __init__(self, s: Split) -> None:
        self.s = s
        self.by_text = {t: q for q, t in s.query_text.items()}

    def scores(self, split: Split, questions: Sequence[str], dim: str) -> np.ndarray:
        rng = np.random.default_rng(int(dim))
        out = rng.uniform(-0.2, 0.5, size=(len(questions), len(split.pids)))
        for i, text in enumerate(questions):
            for p in split.golden[self.by_text[text]]:
                out[i, split.pids.index(p)] = 0.9
        return out


def test_sample_pairs_never_repeats_a_question_or_passage(split: Split) -> None:
    pairs = sample_pairs(split, "train", 8, seed=3)
    assert len(pairs) == 8
    assert len({q for q, _ in pairs}) == 8 and len({p for _, p in pairs}) == 8
    assert all(q in split.qids("train") and p in split.golden[q] for q, p in pairs)
    assert pairs == sample_pairs(split, "train", 8, seed=3)


def test_in_batch_matches_the_hand_formula() -> None:
    p, loss = in_batch(np.array([[1.0, 0.0], [0.5, 0.5]]))
    assert p[0] == pytest.approx(1 / (1 + math.exp(-20)))
    assert p[1] == pytest.approx(0.5)
    assert loss[1] == pytest.approx(math.log(2))


def test_in_batch_equals_sentence_transformers_mnrl() -> None:
    torch = pytest.importorskip("torch")
    losses = pytest.importorskip("sentence_transformers.sentence_transformer.losses")
    from sentence_transformers.util import cos_sim

    rng = np.random.default_rng(0)
    q, d = (torch.tensor(rng.normal(size=(6, 32)), dtype=torch.float32) for _ in range(2))

    class Stub(torch.nn.Module):  # MNRL only needs a module to hold; this test skips the encoder
        pass

    mnrl = losses.MultipleNegativesRankingLoss(Stub())
    ref = float(mnrl.compute_loss_from_embeddings([q, d], None))
    assert float(in_batch(cos_sim(q, d).numpy())[1].mean()) == pytest.approx(ref, rel=1e-5)


def test_batch_view_shapes_ranks_and_matryoshka_sum(split: Split) -> None:
    v = batch_view(split, [("fake", FakeScorer(split))], "train", n=4, seed=0, dim="64")
    m = v["models"][0]
    assert v["n"] == 4 and len(v["pairs"]) == 4 and len(m["sims"]) == 4 == len(m["sims"][0])
    assert m["corpus_rank"] == [1, 1, 1, 1]
    assert [r["dim"] for r in m["matryoshka"]] == list(MATRYOSHKA_DIMS)
    total = sum(r["mean_loss"] for r in m["matryoshka"])
    assert m["matryoshka_loss"] == pytest.approx(total, abs=1e-3)
    assert all(0 < p <= 1 for p in m["p_correct"])
