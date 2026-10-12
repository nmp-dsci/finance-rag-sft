/**
 * The Method page's arithmetic, kept pure so it is tested: the recipe as the train run recorded
 * it (falling back to configs/train_bge_base.yaml when no run exists yet), the learning-rate
 * schedule Transformers applied, and the summaries the worked example prints.
 */
import type { MethodModel, Run } from './api';

/** configs/train_bge_base.yaml, for a page rendered before any train run exists. */
export const RECIPE = {
  base_model: 'bge-base',
  epochs: 4,
  batch_size: 32,
  learning_rate: 2e-5,
  warmup_ratio: 0.1,
  matryoshka_dims: [768, 512, 256, 128, 64],
  seed: 42,
  pad_to_multiple_of: 64,
};

export type Recipe = typeof RECIPE & {
  /** Optimiser steps in the whole run, and per epoch (the last batch is smaller, not dropped). */
  total_steps: number;
  steps_per_epoch: number;
  warmup_steps: number;
};

/** The recipe the run recorded; any key it lacks falls back to the config file's value. */
export function recipeOf(run: Run | null | undefined, nPairs?: number): Recipe {
  const cfg = run?.train?.config ?? {};
  const pick = <T>(k: keyof typeof RECIPE, ok: (v: unknown) => boolean): T =>
    (ok(cfg[k]) ? cfg[k] : RECIPE[k]) as T;
  const isNum = (v: unknown) => typeof v === 'number' && Number.isFinite(v);
  const r = {
    base_model: pick<string>('base_model', (v) => typeof v === 'string'),
    epochs: pick<number>('epochs', isNum),
    batch_size: pick<number>('batch_size', isNum),
    learning_rate: pick<number>('learning_rate', isNum),
    warmup_ratio: pick<number>('warmup_ratio', isNum),
    matryoshka_dims: pick<number[]>('matryoshka_dims', (v) => Array.isArray(v) && v.every(isNum)),
    seed: pick<number>('seed', isNum),
    pad_to_multiple_of: pick<number>('pad_to_multiple_of', isNum),
  };
  const epochs = run?.train?.epochs ?? [];
  const pairs = run?.train?.n_pairs ?? nPairs ?? 0;
  const perEpoch = epochs[0]?.step ?? Math.ceil(pairs / r.batch_size);
  const total = epochs.at(-1)?.step ?? perEpoch * r.epochs;
  return { ...r, total_steps: total, steps_per_epoch: perEpoch, warmup_steps: warmupSteps(r.warmup_ratio, total) };
}

/** Transformers v5 reads a warmup below 1 as a ratio of the total steps, rounded up. */
export function warmupSteps(ratio: number, total: number): number {
  return ratio >= 1 ? Math.round(ratio) : Math.ceil(ratio * total);
}

/** The linear schedule: 0 → peak over the warmup, then straight down to 0 at the last step. */
export function lrAt(step: number, peak: number, warmup: number, total: number): number {
  if (step <= 0) return 0;
  if (step < warmup) return (peak * step) / warmup;
  return Math.max(0, (peak * (total - step)) / Math.max(1, total - warmup));
}

/** Mean similarity on the diagonal (right passages) and off it (wrong ones). */
export function diagStats(m: number[][]): { diag: number; off: number } {
  let d = 0;
  let o = 0;
  let nOff = 0;
  m.forEach((row, i) =>
    row.forEach((v, j) => {
      if (i === j) d += v;
      else {
        o += v;
        nOff += 1;
      }
    }),
  );
  return { diag: m.length ? d / m.length : 0, off: nOff ? o / nOff : 0 };
}

/** Questions where the base model's own passage is outside the top k and the tuned one's is in it. */
export function rescued(base: MethodModel, tuned: MethodModel, k = 10): number[] {
  return base.corpus_rank.flatMap((r, i) => (r > k && (tuned.corpus_rank[i] ?? Infinity) <= k ? [i] : []));
}

/** Bytes for one float32 vector, and for a whole index of `n` of them. */
export function indexBytes(dims: number, n: number): { per: number; total: number } {
  return { per: dims * 4, total: dims * 4 * n };
}

/** 20761088 → "20.8 MB"; 3072 → "3,072 B". */
export function bytes(n: number): string {
  if (n < 10_000) return `${n.toLocaleString('en-AU')} B`;
  if (n < 1e6) return `${(n / 1e3).toFixed(1)} kB`;
  return `${(n / 1e6).toFixed(1)} MB`;
}

/** 2e-5 → "2e-5"; 0.00001971 → "1.97e-5". */
export function sci(x: number): string {
  if (x === 0) return '0';
  const [m, e] = x.toExponential(2).split('e');
  return `${Number(m)}e${Number(e)}`;
}
