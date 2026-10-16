# finance-rag-sft — every target is a thin wrapper over `uv run finrag …`.
.DEFAULT_GOAL := help
API_PORT ?= 8084
MLFLOW_TRACKING_URI ?= http://localhost:5000
GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo local)
IMAGE ?= ghcr.io/nmp-dsci/finance-rag-sft

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

setup: ## python deps (uv) and frontend deps (npm)
	uv sync
	cd frontend && npm ci

platform-up: ## start the central MLflow (nmp-central-ai on :5000)
	$(MAKE) -C ../nmp-central-ai up

platform-status: ## preflight: the central MLflow must answer /health
	@curl -fsS $(MLFLOW_TRACKING_URI)/health >/dev/null || (echo "central MLflow down at $(MLFLOW_TRACKING_URI): run make platform-up"; exit 1)

smoke: platform-status ## zero-model run logged to the central MLflow
	uv run finrag smoke

doctor: ## device, torch build, GPU driver, MLflow reachability
	uv run finrag doctor

split: ## M1: write split v1 from the pinned dataset (once; committed)
	uv run finrag split

baseline: platform-status ## M2: BM25, bge-small, bge-base on test, each ranking alone
	uv run finrag baseline

check: platform-status ## M2: the harness check against Phil Schmid's 0.768
	uv run finrag check

train: platform-status ## M3: fine-tune bge-base, choose the epoch on val (MAX_STEPS= for a smoke run)
	caffeinate -is uv run finrag train $(if $(MAX_STEPS),--max-steps $(MAX_STEPS))

eval: platform-status ## M4: score TRAIN_RUN's chosen checkpoint on test, once
	uv run finrag eval --train-run $(TRAIN_RUN)

compare: ## paired comparison: A against B at DIM (default 768)
	uv run finrag compare $(A) $(B) --dim $(or $(DIM),768)

runs: ## every run, newest first
	uv run finrag runs

dev: ## the API on :$(API_PORT), reloading (UI: cd frontend && npm run dev → :5176)
	uv run finrag serve --port $(API_PORT) --reload

viewer: ## build the frontend and serve it with the API on :$(API_PORT)
	cd frontend && npm run build
	uv run finrag serve --port $(API_PORT)

test: ## pytest (offline)
	uv run pytest -q

lint: ## ruff + mypy (+ frontend checks when node_modules exist)
	uv run ruff format --check src tests && uv run ruff check src tests && uv run mypy
	@if [ -d frontend/node_modules ]; then cd frontend && npm run typecheck && npm run lint:design && npm test; fi

fmt: ## ruff format + fix
	uv run ruff format src tests && uv run ruff check --fix src tests

docker-build: ## build the arm64 images locally (train + viewer); CI builds amd64 CUDA
	docker build --target train -t $(IMAGE):train-$(GIT_SHA) --build-arg GIT_SHA=$(GIT_SHA) .
	docker build --target viewer -t $(IMAGE):viewer-$(GIT_SHA) --build-arg GIT_SHA=$(GIT_SHA) .

docker-smoke: platform-status ## the smoke run inside the train image, on the nmp-central network
	GIT_SHA=$(GIT_SHA) docker compose run --rm train smoke

.PHONY: help setup platform-up platform-status smoke doctor split baseline check train eval compare runs dev viewer test lint fmt docker-build docker-smoke
