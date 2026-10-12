# finance-rag-sft — Finance RAG retriever

Does fine-tuning an embedding model put the right passage in the top 10 more often?
This repo fine-tunes [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) on
finance question and passage pairs from company 10-K filings
([`philschmid/finanical-rag-embedding-dataset`](https://huggingface.co/datasets/philschmid/finanical-rag-embedding-dataset))
and scores the untouched model and the tuned one with the same harness, on the same frozen
test questions, against the same 6,757-passage corpus.

## Result

Test split v1: 693 questions, each ranking all 6,757 passages. Recall@10 = the share of
questions whose golden passage lands in the top 10.

| Retriever | Recall@10 | nDCG@10 | MRR@10 | Recall@1 |
|---|---:|---:|---:|---:|
| **bge-base, fine-tuned (epoch 2), 768 dims** | **0.926** | **0.830** | **0.799** | **0.724** |
| bge-base, fine-tuned, 64 dims | 0.896 | 0.795 | 0.763 | 0.690 |
| bge-base, 768 dims | 0.872 | 0.756 | 0.720 | 0.639 |
| BM25 (bm25s, stemmed) | 0.870 | 0.759 | 0.723 | 0.645 |
| bge-small, 384 dims | 0.856 | 0.733 | 0.695 | 0.606 |
| bge-base, 64 dims | 0.792 | 0.650 | 0.607 | 0.521 |
| bge-small, 64 dims | 0.761 | 0.626 | 0.584 | 0.498 |

Paired over the same 693 questions:

| Comparison | Fixed | Broken | Δ Recall@10 (95% CI) | Exact McNemar p |
|---|---:|---:|---|---:|
| tuned vs base, 768 dims | 42 | 4 | +5.5 pts (+3.6 to +7.4) | 5.1e-9 |
| tuned vs base, 64 dims | 84 | 12 | +10.4 pts (+7.8 to +13.1) | 1.8e-14 |
| tuned at **64** vs base at **768** | 35 | 18 | +2.5 pts (+0.4 to +4.5) | 0.027 |
| tuned 768 vs BM25 | 47 | 8 | +5.6 pts | 8.1e-8 |

The tuned model's 64-dimension vectors beat the base model's full 768: a 12x smaller index
that retrieves better. Training took 22 minutes on an M1 Pro (MPS). The checkpoint was chosen
on the val split, and test was scored once.

**Harness check.** On Phil Schmid's own setup (random 90/10 rows, all 7,000 rows as the
corpus, relevance = the question's own row) the untouched bge-base scores nDCG@10 0.748 against
his published 0.768. Across 20 split seeds the same setup ranges 0.728 to 0.773 (median 0.746),
so the gap is which questions his split drew, not the harness.

**Not yet measured:** whether the tuned model got worse outside this dataset (MTEB FiQA2018, M5).

## Why the split is by passage

The dataset has 7,000 rows but 6,757 unique passages: 141 passages repeat (one 15 times) and
48 questions have more than one golden passage. A random row split puts about 44 of 700 test
questions' passages into train. Split v1 deals **passage groups** (passages joined by a shared
question) 80/10/10 with seed 42, so no golden passage crosses train, val and test; a test
asserts it. The split files hold IDs and hashes only; text is rebuilt from the dataset pinned
at revision `e0b17819`, because the dataset card declares no licence.

## Run it

```bash
make setup            # uv sync + npm ci
make platform-up      # the central MLflow (nmp-central-ai) on :5000
make smoke            # zero-model run, logged
make baseline check   # M2: BM25, bge-small, bge-base; the harness check
make train            # M3: ~22 min on an M1 Pro; epoch chosen on val
make eval TRAIN_RUN=<train run id>   # M4: test, once
make compare A=<tuned eval run> B=<base run>
make viewer           # http://127.0.0.1:8084
```

### The viewer

React + Vite on the portfolio's house UI (hue 15), served by FastAPI on :8084:
Overview (the result and milestones), Data (browse train / val / test, questions and passages),
Runs (per-question golden ranks, `?vs=<run>` to compare), Question (top 10s side by side) and
Search (live retrieval with BM25, the base and the tuned model).

### Containers

One Dockerfile, two images: `train` (CUDA 12.6 on linux/amd64, CPU on linux/arm64) and
`viewer`. CI pushes both to `ghcr.io/nmp-dsci/finance-rag-sft` tagged by commit.
Docker on a Mac cannot reach the GPU, so the Mac trains natively and the image is how training
moves to a CUDA host such as an EC2 g6:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --rm train train --no-mlflow
# copy runs/<id> and outputs/<id> back, then on the Mac:
uv run finrag log-run runs/<id>
```

## Layout and decisions

See [AGENTS.md](AGENTS.md). The plan, with every decision and its reason, is
[.lavish/s00_finance-rag-retriever-plan.html](.lavish/s00_finance-rag-retriever-plan.html).
