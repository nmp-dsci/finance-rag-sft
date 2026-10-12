# AGENTS.md — finance-rag-sft

The source of truth for how this project is built and why. `CLAUDE.md` points here.
The plan lives in `.lavish/s00_finance-rag-retriever-plan.html` (never gitignored).

## 1 · What it is

A finance RAG retriever. It fine-tunes `BAAI/bge-base-en-v1.5` on question and passage pairs
from company 10-K filings (`philschmid/finanical-rag-embedding-dataset`) and asks one question:
**does the golden passage land in the top 10 more often than with the untouched base model?**
It follows use case UC1 of `ai-engineer-fit/.lavish/s63_fine-tuning-playbook.html`.

## 2 · Decisions, and the reasons

| Decision | Choice | Why |
|---|---|---|
| Data | the Hub dataset pinned at `e0b17819…` (`config.py`); split files hold IDs and hashes, text is rebuilt from the parquet | the card declares no licence, so the repo does not redistribute text |
| Split (D3) | v1: 80/10/10 by **passage group** (passages joined by a shared question), seed 42; 5,547 / 693 / 693 questions | a random row split leaks: ~44 of 700 test questions had their passage in train |
| Corpus (D2) | every question ranks all 6,757 unique passages | harder and closer to real RAG; training passages act as distractors |
| Metric | `recall@10` = any golden passage in the top 10 (headline); nDCG@10, MRR@10, recall@1/@5 beside it; at full size and 64 dims | the owner's question; 64 dims tests the Matryoshka claim |
| Base model (D1) | bge-base-en-v1.5 at a pinned revision; no query instruction for any model | matches the reference run; the same prompt for base and tuned |
| Baselines (D5, D10) | BM25 (bm25s, English stopwords, Snowball stemmer), bge-small, bge-base, each ranking alone; no hybrid | the owner's call: BM25 stays a floor |
| Training | MNRL inside MatryoshkaLoss [768…64], 4 epochs, lr 2e-5, batch 32, 10% warmup, NO_DUPLICATES; each epoch saved; **chosen on val**, test scored once | s63's recipe with the one change that keeps test honest |
| Mac GPU | pad to multiples of 64 and empty the MPS cache each step (train only) | measured 2026-10-12 on the M1 Pro: 15–20 s/step without, ~2.3 s with |
| Where it trains (D4) | the Mac natively first; the CUDA image on EC2 if the Mac fails | Docker on a Mac cannot reach the GPU |
| Comparison | exact McNemar on paired hits@10, paired bootstrap 95% CI on deltas | same questions, so a paired test |
| Tracking | central MLflow, experiment `finance-rag-sft/evals`; tags `project`, `git_sha`, `env`, `billing=none`; run folders under `runs/` are the record, MLflow replays them | PLATFORM.md; a run from another machine is logged later with `finrag log-run` |
| Model registry (D6) | the chosen checkpoint is registered as `finance-rag-sft-bge-base` in MLflow only; no Hub push until the dataset licence is checked | the licence is unknown |
| Viewer | React 18 + Vite + TypeScript + react-router, served by FastAPI (API :8084, dev UI :5176), vidore-v3-loop's layout, tau2-loop's URL grammar (`?vs=` is a lens) | one portfolio, one way of reading a run |
| Viewer styling (D8, D9) | imports the vendored `.lavish/house.css` + `project.css`; hue 15 from the registry; no hex in `frontend/src` | the colour has one source; sibling apps' hand-coded colours drifted |
| Containers (D11) | one Dockerfile: `train` (CUDA 12.6 amd64 / CPU arm64) and `viewer`; GHCR by commit SHA | the project moves to any CUDA host unchanged |

## 3 · Layout

```
src/finance_rag_sft/
  config.py        paths, pinned dataset and models, settings (the only reader of os.environ)
  data.py          pinned download, IDs, passage-group split, leak check, split stats
  harness.py       score_matrix (rank the whole corpus), metrics, BM25 and dense retrievers, embedding cache
  train.py         M3: MNRL inside Matryoshka, epochs scored on val, one chosen
  compare.py       exact McNemar, paired bootstrap, rank shifts
  runs.py          run folders: runs/<id>/run.json + queries.jsonl (the record)
  tracking.py      MLflow preflight, required tags, log_run (replays a folder), model registration
  cli.py           `finrag` — smoke, split, baseline, check, train, eval, compare, runs, log-run, serve, doctor
  serving/app.py   FastAPI: the contract in docs/api.md; serves frontend/dist
frontend/          the viewer (house UI, hue 15)
data/splits/v1/    manifest.json, queries.jsonl, passages.jsonl (committed; IDs only)
runs/              one folder per scored run (committed)
outputs/           checkpoints (gitignored; the chosen one is registered in MLflow)
configs/           train_bge_base.yaml
Dockerfile, compose.yaml, compose.gpu.yaml
```

## 4 · How to run

```bash
make setup                         # uv sync + npm ci
make smoke                         # zero-model run in central MLflow
make baseline && make check        # M2
make train                         # M3 on the Mac (~30 min of steps + val scoring)
make eval TRAIN_RUN=<id>           # M4: test once
make compare A=<tuned> B=<base>
make viewer                        # http://127.0.0.1:8084
```

On a CUDA host (D4's fallback): pull `ghcr.io/nmp-dsci/finance-rag-sft:train-<sha>`, run with
`compose.gpu.yaml` and `--no-mlflow`, copy `runs/<id>` and `outputs/<id>` back, then
`finrag log-run runs/<id>` on the Mac.
