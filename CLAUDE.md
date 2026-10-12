# CLAUDE.md — finance-rag-sft

> Read [`AGENTS.md`](./AGENTS.md) first: what this is, the decisions, the layout.
> This file is a pointer plus the rules that bite.

## Quick reference

- `make smoke` · `make baseline` · `make train` · `make eval TRAIN_RUN=<id>` · `make viewer`
- `uv run pytest -q` · `make lint`
- MLflow: central (`make platform-up` → `make -C ../nmp-central-ai up`) → http://localhost:5000

## Rules

- **The test split is scored, never optimised on.** Checkpoints are chosen on val;
  `finrag eval` refuses a second test run of the same checkpoint without `--force`.
- **Split v1 is frozen.** A change is a new version (`data/splits/v2/`), never an edit to v1.
- **The dataset is pinned** (`config.DATASET_REVISION`); load through `finance_rag_sft.data`.
- **Run folders are the record**; MLflow replays them. Never fall back to a local MLflow store.
- **Same treatment for base and tuned**: same harness, corpus, questions and (empty) query prompt.
- **Colour comes from the registry** (hue 15 in nmp-central-ai's `registry/projects.yaml`):
  no hex in `frontend/src`; never edit `.lavish/house.css`.
- `data/cache/` and `outputs/` are never committed. Never add `.lavish/` to `.gitignore`.
