from __future__ import annotations

from pathlib import Path

from finance_rag_sft.data import (
    build_split,
    collect,
    leak_check,
    load_split_from,
    passage_groups,
    pid_of,
    qid_of,
)


def test_ids_ignore_whitespace() -> None:
    assert pid_of("a  b\n c") == pid_of("a b c")
    assert qid_of("x") != pid_of("x")


def test_collect_merges_duplicates(rows: list[tuple[str, str]]) -> None:
    data = collect(rows)
    assert data.rows == len(rows)
    assert len(data.golden[qid_of("What are the risks?")]) == 2  # two golden passages
    assert data.passage_rows[pid_of("Jane Doe has been CEO since 2019.")] == 2
    assert len(data.query_text) == len(rows) - 2  # two repeated questions merged


def test_groups_join_passages_through_a_shared_question(rows: list[tuple[str, str]]) -> None:
    data = collect(rows)
    groups = passage_groups(data.golden)
    a = pid_of("Supply chain disruption is a key risk.")
    b = pid_of("Interest rate changes are a key risk.")
    assert groups[a] == groups[b]


def test_split_is_leak_free_deterministic_and_reloads(
    rows: list[tuple[str, str]], tmp_path: Path
) -> None:
    m1 = build_split(rows, tmp_path / "v1")
    m2 = build_split(rows, tmp_path / "again")
    assert m1["fingerprint"] == m2["fingerprint"]
    split = load_split_from(rows, tmp_path / "v1")
    assert leak_check(split)["ok"]
    for q, ps in split.golden.items():
        assert {split.passage_split[p] for p in ps} == {split.query_split[q]}
    assert sum(m1["queries"][s] for s in ("train", "val", "test")) == m1["queries"]["total"]
    assert split.pairs("train") and all(len(p) == 2 for p in split.pairs("train"))
