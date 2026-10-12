# syntax=docker/dockerfile:1.7
# finance-rag-sft — one Dockerfile, two images (D11):
#
#   train   `finrag` with torch: CUDA 12.6 on linux/amd64 (EC2 g6, any NVIDIA host),
#           CPU-only on linux/arm64 (smoke runs on the Mac; Docker cannot reach the Mac GPU)
#   viewer  the API and the built frontend on :8084
#
# Never in the image: model weights (HF cache volume at /cache/hf), checkpoints (/app/outputs),
# run folders written by a job (/app/runs, mounted), tokens (environment variables).
#
#   docker build --target train  -t finance-rag-sft:train  --build-arg GIT_SHA=$(git rev-parse --short HEAD) .
#   docker run --rm --gpus all -v $PWD/runs:/app/runs -v $PWD/outputs:/app/outputs \
#     -v hf-cache:/cache/hf finance-rag-sft:train train          # on a CUDA host
#   docker compose run --rm train smoke                          # on the Mac, CPU

# ── Python environment, shared by both images ───────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58 AS base
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/cache/hf FINRAG_ROOT=/app MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock README.md ./
# uv.lock picks torch per platform: +cu126 on amd64, +cpu on arm64 (tool.uv.sources)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY src/ src/
COPY configs/ configs/
COPY data/splits/ data/splits/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev
ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA}

# ── train: `finrag <command>`, default `train` ──────────────────────────────
FROM base AS train
ENTRYPOINT ["uv", "run", "--no-sync", "finrag"]
CMD ["train"]

# ── frontend bundle (the house UI comes from .lavish/, as in dev) ───────────
FROM node:22-slim@sha256:c3de60bf2f9dd0ac6370e6117950ff62d6e339527e7472301c9c78a017978392 AS ui
WORKDIR /src
COPY .lavish/house.css .lavish/project.css .lavish/
COPY .lavish/fonts/ .lavish/fonts/
COPY frontend/package.json frontend/package-lock.json frontend/
RUN cd frontend && npm ci
COPY frontend/ frontend/
RUN cd frontend && npm run build

# ── viewer: API + built frontend + the committed runs ───────────────────────
FROM base AS viewer
COPY --from=ui /src/frontend/dist frontend/dist
COPY runs/ runs/
EXPOSE 8084
ENTRYPOINT ["uv", "run", "--no-sync", "finrag"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8084"]
