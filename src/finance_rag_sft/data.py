"""The dataset at its pinned revision, and split v1.

Split v1 is dealt by *passage group*, not by row. A group is every passage reachable from
another through a shared question, so no question's golden passage can sit in two splits.
The split files hold IDs and hashes only; text is rebuilt from the pinned parquet, so the
repo never redistributes the dataset (its card declares no licence).
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

from finance_rag_sft.config import (
    DATASET,
    DATASET_FILE,
    DATASET_REVISION,
    SPLIT_VERSION,
    SPLITS_DIR,
    settings,
)

SPLITS = ("train", "val", "test")
FRACTIONS = (0.8, 0.1, 0.1)
SEED = 42


def normalise(text: str) -> str:
    """Collapse whitespace. The only normalisation: IDs hash this, and passages compare by it."""
    return " ".join(text.split())


def _id(prefix: str, text: str) -> str:
    return prefix + hashlib.sha1(normalise(text).encode()).hexdigest()[:12]


def pid_of(text: str) -> str:
    return _id("p", text)


def qid_of(text: str) -> str:
    return _id("q", text)


def fetch_dataset() -> Path:
    """The parquet at the pinned revision, from the Hugging Face cache (HF_HOME)."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=DATASET, repo_type="dataset", filename=DATASET_FILE, revision=DATASET_REVISION
        )
    )


def read_rows(path: Path | None = None) -> list[tuple[str, str]]:
    """(question, context) rows, in file order."""
    import pyarrow.parquet as pq

    table = pq.read_table(path or fetch_dataset()).to_pydict()
    return list(zip(table["question"], table["context"], strict=True))


@dataclass
class SplitData:
    """Everything a split decision needs, built from rows; the same code builds and loads."""

    passage_text: dict[str, str]
    query_text: dict[str, str]
    golden: dict[str, list[str]]  # qid -> sorted golden pids
    passage_rows: Counter[str]  # pid -> rows of the 7,000 that carried it
    query_rows: Counter[str]
    rows: int


def collect(rows: Sequence[tuple[str, str]]) -> SplitData:
    passage_text: dict[str, str] = {}
    query_text: dict[str, str] = {}
    golden: defaultdict[str, set[str]] = defaultdict(set)
    passage_rows: Counter[str] = Counter()
    query_rows: Counter[str] = Counter()
    for question, context in rows:
        q, p = qid_of(question), pid_of(context)
        passage_text.setdefault(p, normalise(context))
        query_text.setdefault(q, normalise(question))
        golden[q].add(p)
        passage_rows[p] += 1
        query_rows[q] += 1
    return SplitData(
        passage_text=passage_text,
        query_text=query_text,
        golden={q: sorted(ps) for q, ps in golden.items()},
        passage_rows=passage_rows,
        query_rows=query_rows,
        rows=len(rows),
    )


def passage_groups(golden: dict[str, list[str]]) -> dict[str, str]:
    """Union-find over passages joined by a shared question: pid -> group root pid."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for pids in golden.values():
        first = find(pids[0])
        for p in pids[1:]:
            root = find(p)
            if root != first:
                lo, hi = sorted((first, root))
                parent[hi] = lo
                first = lo
    return {p: find(p) for p in list(parent)}


def deal(
    data: SplitData, seed: int = SEED, fractions: tuple[float, float, float] = FRACTIONS
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Deal passage groups to train / val / test by cumulative question count.

    Returns (query split, passage split, passage group)."""
    group_of = passage_groups(data.golden)
    group_queries: defaultdict[str, list[str]] = defaultdict(list)
    for q, pids in data.golden.items():
        group_queries[group_of[pids[0]]].append(q)
    roots = sorted(group_queries)
    random.Random(seed).shuffle(roots)
    total = len(data.golden)
    cut_train, cut_val = fractions[0] * total, (fractions[0] + fractions[1]) * total
    group_split: dict[str, str] = {}
    seen = 0
    for root in roots:
        group_split[root] = "train" if seen < cut_train else "val" if seen < cut_val else "test"
        seen += len(group_queries[root])
    query_split = {q: group_split[group_of[pids[0]]] for q, pids in data.golden.items()}
    passage_split = {p: group_split[g] for p, g in group_of.items()}
    return query_split, passage_split, group_of


def _hist(values: Sequence[int], edges: Sequence[int]) -> list[list[int]]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        out.append([lo, hi, sum(1 for v in values if lo <= v < hi)])
    return out


def _length_stats(values: Sequence[int], edges: Sequence[int]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "max": ordered[-1],
        "hist": _hist(ordered, edges),
    }


def _fingerprint(lines: Iterable[str]) -> str:
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


def build_split(
    rows: Sequence[tuple[str, str]],
    out_dir: Path,
    truncated_passages: int | None = None,
    file_sha256: str | None = None,
) -> dict[str, Any]:
    """Write queries.jsonl, passages.jsonl and manifest.json; return the manifest."""
    data = collect(rows)
    query_split, passage_split, group_of = deal(data)
    q_lines = [
        json.dumps({"qid": q, "split": query_split[q], "golden": data.golden[q]}, sort_keys=True)
        for q in sorted(data.golden)
    ]
    p_lines = [
        json.dumps(
            {
                "pid": p,
                "split": passage_split[p],
                "group": group_of[p],
                "rows": data.passage_rows[p],
            },
            sort_keys=True,
        )
        for p in sorted(data.passage_text)
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "queries.jsonl").write_text("\n".join(q_lines) + "\n")
    (out_dir / "passages.jsonl").write_text("\n".join(p_lines) + "\n")

    counts = Counter(query_split.values())
    pairs = Counter(query_split[q] for q, ps in data.golden.items() for _ in ps)
    manifest: dict[str, Any] = {
        "version": out_dir.name,
        "dataset": DATASET,
        "revision": DATASET_REVISION,
        "file": DATASET_FILE,
        "file_sha256": file_sha256,
        "seed": SEED,
        "fractions": list(FRACTIONS),
        "dealt_by": "passage group (passages joined by a shared question), by cumulative question count",
        "normalisation": "whitespace collapsed; IDs = sha1(normalised text)[:12]",
        "rows": data.rows,
        "passages": len(data.passage_text),
        "queries": {s: counts[s] for s in SPLITS} | {"total": len(data.golden)},
        "pairs": {s: pairs[s] for s in SPLITS},
        "groups": len(set(group_of.values())),
        "duplicates": {
            "passages_repeated": sum(1 for v in data.passage_rows.values() if v > 1),
            "max_repeats": max(data.passage_rows.values()),
            "rows_sharing": sum(v for v in data.passage_rows.values() if v > 1),
            "questions_repeated": sum(1 for v in data.query_rows.values() if v > 1),
        },
        "multi_golden_queries": sum(1 for ps in data.golden.values() if len(ps) > 1),
        "truncated_passages": truncated_passages,
        "fingerprint": _fingerprint(q_lines + p_lines),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_sha": settings().code_sha,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


@dataclass
class Split:
    """A loaded split: IDs from data/splits/<v>/, text from the pinned parquet."""

    version: str
    manifest: dict[str, Any]
    pids: list[str]  # corpus order (sorted IDs): the row order of every embedding matrix
    passage_text: dict[str, str]
    passage_split: dict[str, str]
    passage_rows: dict[str, int]
    query_text: dict[str, str]
    query_split: dict[str, str]
    golden: dict[str, list[str]]
    questions_of: dict[str, list[str]] = field(default_factory=dict)  # pid -> qids

    def qids(self, split: str) -> list[str]:
        return sorted(q for q, s in self.query_split.items() if split in ("all", s))

    def pairs(self, split: str) -> list[tuple[str, str]]:
        """(question, passage) training pairs: one per golden passage."""
        return [
            (self.query_text[q], self.passage_text[p])
            for q in self.qids(split)
            for p in self.golden[q]
        ]

    @property
    def fingerprint(self) -> str:
        return str(self.manifest["fingerprint"])


def load_split_from(rows: Sequence[tuple[str, str]], split_dir: Path) -> Split:
    data = collect(rows)
    manifest = json.loads((split_dir / "manifest.json").read_text())
    query_split: dict[str, str] = {}
    golden: dict[str, list[str]] = {}
    for line in (split_dir / "queries.jsonl").read_text().splitlines():
        r = json.loads(line)
        query_split[r["qid"]] = r["split"]
        golden[r["qid"]] = r["golden"]
    passage_split: dict[str, str] = {}
    passage_rows: dict[str, int] = {}
    for line in (split_dir / "passages.jsonl").read_text().splitlines():
        r = json.loads(line)
        passage_split[r["pid"]] = r["split"]
        passage_rows[r["pid"]] = r["rows"]
    missing = (set(golden) - set(data.query_text)) | (set(passage_split) - set(data.passage_text))
    if missing:
        raise ValueError(
            f"split {split_dir.name}: {len(missing)} IDs not in the dataset; wrong revision?"
        )
    questions_of: defaultdict[str, list[str]] = defaultdict(list)
    for q, ps in golden.items():
        for p in ps:
            questions_of[p].append(q)
    pids = sorted(passage_split)
    return Split(
        version=split_dir.name,
        manifest=manifest,
        pids=pids,
        passage_text={p: data.passage_text[p] for p in pids},
        passage_split=passage_split,
        passage_rows=passage_rows,
        query_text={q: data.query_text[q] for q in golden},
        query_split=query_split,
        golden=golden,
        questions_of=dict(questions_of),
    )


@cache
def load_split(version: str = SPLIT_VERSION) -> Split:
    split_dir = SPLITS_DIR / version
    if not (split_dir / "manifest.json").exists():
        raise FileNotFoundError(f"{split_dir} has no manifest; run `finrag split` first")
    return load_split_from(read_rows(), split_dir)


def leak_check(split: Split) -> dict[str, Any]:
    """No passage may be golden for questions in two different splits."""
    seen: defaultdict[str, set[str]] = defaultdict(set)
    for q, ps in split.golden.items():
        for p in ps:
            seen[p].add(split.query_split[q])
    crossing = [s for s in seen.values() if len(s) > 1]
    tt = sum(1 for s in crossing if {"train", "test"} <= s)
    tv = sum(1 for s in crossing if {"train", "val"} <= s)
    vt = sum(1 for s in crossing if {"val", "test"} <= s)
    return {
        "ok": tt == tv == vt == 0,
        "train_test_overlap": tt,
        "train_val_overlap": tv,
        "val_test_overlap": vt,
    }


def split_stats(split: Split) -> dict[str, Any]:
    """The Data explorer's header: the manifest plus lengths and the leak check."""
    m = split.manifest
    pw = [len(t.split()) for t in split.passage_text.values()]
    qw = [len(t.split()) for t in split.query_text.values()]
    return {
        "version": split.version,
        "dataset": m["dataset"],
        "revision": m["revision"],
        "seed": m["seed"],
        "fingerprint": m["fingerprint"],
        "rows": m["rows"],
        "passages": m["passages"],
        "queries": m["queries"],
        "pairs": m["pairs"],
        "groups": m["groups"],
        "duplicates": m["duplicates"],
        "multi_golden_queries": m["multi_golden_queries"],
        "leak_check": leak_check(split),
        "truncated_passages": m["truncated_passages"],
        "lengths": {
            "passage_words": _length_stats(pw, [0, 10, 20, 30, 40, 60, 80, 120, 200, 800]),
            "question_words": _length_stats(qw, [0, 5, 10, 15, 20, 25, 30, 40]),
        },
    }
