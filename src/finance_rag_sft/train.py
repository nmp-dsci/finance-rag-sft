"""M3: fine-tune bge-base with MultipleNegativesRankingLoss inside MatryoshkaLoss (s63 UC1).

The recipe is s63's, with one change: the checkpoint is chosen on the **val** split by the
same harness that scores test, never on test. Each epoch is saved; each is scored on val.

Two settings exist only for Apple's GPU (MPS), measured on 2026-10-12 on an M1 Pro: padding
to multiples of 64 tokens keeps the number of tensor shapes small (MPS recompiles kernels per
shape; dynamic padding made steps 15-20 s), and emptying the MPS cache every step stops
memory growth into swap (steps went from ~9 s back to ~2.3 s). Neither changes the maths.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from finance_rag_sft import config, runs
from finance_rag_sft.data import Split, load_split
from finance_rag_sft.harness import DenseRetriever, evaluate, load_st


@dataclass
class TrainConfig:
    base_model: str = "bge-base"
    epochs: int = 4
    batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    matryoshka_dims: list[int] = field(default_factory=lambda: [768, 512, 256, 128, 64])
    seed: int = 42
    pad_to_multiple_of: int | None = 64
    max_steps: int | None = None
    select_dim: str = "768"
    select_metric: str = "recall@10"


def load_config(path: Path) -> TrainConfig:
    raw = yaml.safe_load(path.read_text()) or {}
    return TrainConfig(**raw)


def _free_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - best effort
        pass


def _callbacks(loss_curve: list[dict[str, float]], on_mps: bool) -> list[Any]:
    import torch
    from transformers import TrainerCallback

    class LossLog(TrainerCallback):  # type: ignore[misc]
        def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kw: Any) -> None:
            if logs and "loss" in logs:
                loss_curve.append({"step": int(state.global_step), "loss": float(logs["loss"])})

    class MpsCache(TrainerCallback):  # type: ignore[misc]
        def on_step_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
            torch.mps.empty_cache()

    return [LossLog(), MpsCache()] if on_mps else [LossLog()]


def _checkpoints(out: Path) -> list[tuple[Path, float, int]]:
    """(dir, epoch, step) for every saved checkpoint, by step."""
    found = []
    for d in out.glob("checkpoint-*"):
        state = d / "trainer_state.json"
        step = int(d.name.split("-")[-1])
        epoch = float(json.loads(state.read_text())["epoch"]) if state.exists() else 0.0
        found.append((d, epoch, step))
    return sorted(found, key=lambda t: t[2])


def train(cfg: TrainConfig, note: str = "", split: Split | None = None) -> dict[str, Any]:
    """Train, score every epoch on val, choose one. Returns the run doc (folder written)."""
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.base.sampler import BatchSamplers
    from sentence_transformers.sentence_transformer.losses import (
        MatryoshkaLoss,
        MultipleNegativesRankingLoss,
    )

    split = split or load_split()
    base = config.BASE_MODELS[cfg.base_model]
    run_id = runs.new_run_id("train", cfg.base_model)
    out = config.OUTPUTS_DIR / run_id
    pairs = split.pairs("train")

    model = load_st(base.hub_id, base.revision)
    device = str(model.device).split(":")[0]
    if cfg.pad_to_multiple_of:
        model[0].processing_kwargs = {"text": {"pad_to_multiple_of": cfg.pad_to_multiple_of}}
    dataset = Dataset.from_dict(
        {"anchor": [q for q, _ in pairs], "positive": [p for _, p in pairs]}
    )
    loss = MatryoshkaLoss(
        model, MultipleNegativesRankingLoss(model), matryoshka_dims=cfg.matryoshka_dims
    )
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out),
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps or -1,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_ratio,  # a float is a ratio in Transformers v5
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # in-batch negatives must not be positives
        eval_strategy="no",
        save_strategy="epoch",
        save_only_model=True,
        logging_steps=10,
        report_to="none",
        seed=cfg.seed,
        data_seed=cfg.seed,
    )
    loss_curve: list[dict[str, float]] = []
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=loss,
        callbacks=_callbacks(loss_curve, device == "mps"),
    )
    t0 = time.time()
    trainer.train()
    train_seconds = round(time.time() - t0, 1)
    if not _checkpoints(out):  # a max_steps run that ended inside an epoch
        final = out / f"checkpoint-{trainer.state.global_step}"
        trainer.save_model(str(final))
        (final / "trainer_state.json").write_text(json.dumps({"epoch": trainer.state.epoch}))
    del trainer, model, loss
    _free_memory()

    epochs: list[dict[str, Any]] = []
    for ckpt, epoch, step in _checkpoints(out):
        rel = str(ckpt.relative_to(config.ROOT))
        retriever = DenseRetriever.tuned(Path(rel), run_id, round(epoch))
        metrics, _, _ = evaluate(retriever, split, "val")
        nearest = [p["loss"] for p in loss_curve if p["step"] <= step]
        epochs.append(
            {
                "epoch": round(epoch, 2),
                "step": step,
                "checkpoint": rel,
                "val": metrics,
                "loss": nearest[-1] if nearest else None,
            }
        )
        del retriever
        _free_memory()
    best = max(
        epochs,
        key=lambda e: (
            e["val"][cfg.select_dim][cfg.select_metric],
            e["val"][cfg.select_dim]["ndcg@10"],
            -e["epoch"],
        ),
    )
    doc = runs.run_doc(
        run_id=run_id,
        kind="train",
        retriever=None,
        split=split,
        eval_split="val",
        dims=[],
        n_queries=len(split.qids("val")),
        metrics={},
        timing={"train": train_seconds},
        device=device,
        note=note,
        label=f"train {cfg.base_model}: MNRL + Matryoshka, {cfg.epochs} epochs",
        train={
            "config": asdict(cfg),
            "epochs": epochs,
            "chosen": {
                "epoch": best["epoch"],
                "checkpoint": best["checkpoint"],
                "by": f"val {cfg.select_metric} at {cfg.select_dim}",
            },
            "train_seconds": train_seconds,
            "n_pairs": len(pairs),
            "loss_curve": loss_curve,
        },
    )
    runs.write_run(doc)
    return doc
