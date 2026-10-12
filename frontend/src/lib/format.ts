/** Number and text rules, in one place so every page prints a metric the same way. */
import type { Change, DeltaKey, Hit, MetricKey, RunSummary } from './api';

export const DASH = '—';

/** A proportion as a percentage to 1 dp: 0.7123 → "71.2%". */
export function pct(x: number | null | undefined): string {
  return x == null || Number.isNaN(x) ? DASH : `${(x * 100).toFixed(1)}%`;
}

/** A score in [0, 1] to 3 dp: 0.63412 → "0.634". */
export function score3(x: number | null | undefined): string {
  return x == null || Number.isNaN(x) ? DASH : x.toFixed(3);
}

/** Recall is a share of questions (a percentage); nDCG and MRR are scores. */
export function metric(key: MetricKey | DeltaKey, x: number | null | undefined): string {
  return key.startsWith('recall') ? pct(x) : score3(x);
}

/** An exact McNemar p-value: "< 0.001", else 3 dp. */
export function pValue(p: number | null | undefined): string {
  if (p == null || Number.isNaN(p)) return DASH;
  if (p < 0.001) return '< 0.001';
  return p.toFixed(3);
}

/** "p = 0.032" or "p < 0.001". */
export function pText(p: number | null | undefined): string {
  const v = pValue(p);
  return v.startsWith('<') ? `p ${v}` : `p = ${v}`;
}

/** "+4.2", "−1.2", and "0.0" for a value that rounds to zero either side. */
const sign = (s: string, x: number) => (x > 0 ? `+${s}` : x < 0 ? s.replace('-', '−') : s.replace('-', ''));

/** A paired difference a − b: recall in percentage points (1 dp), scores to 3 dp, signed. */
export function delta(key: DeltaKey, x: number | null | undefined): string {
  if (x == null || Number.isNaN(x)) return DASH;
  if (key.startsWith('recall')) {
    const v = x * 100;
    return `${sign(v.toFixed(1), Number(v.toFixed(1)))} pp`;
  }
  return sign(x.toFixed(3), Number(x.toFixed(3)));
}

/** A 95% interval for a paired difference: "[+1.8, +6.5] pp" or "[−0.012, +0.040]". */
export function interval(key: DeltaKey, lo: number, hi: number): string {
  const strip = (s: string) => s.replace(/ pp$/, '');
  const unit = key.startsWith('recall') ? ' pp' : '';
  return `[${strip(delta(key, lo))}, ${strip(delta(key, hi))}]${unit}`;
}

/** Thousands separators: 6757 → "6,757". */
export function int(n: number | null | undefined): string {
  return n == null ? DASH : n.toLocaleString('en-AU');
}

/** ISO 8601 → "2026-10-12 17:00". */
export function when(iso: string | null | undefined): string {
  return iso ? iso.replace('T', ' ').slice(0, 16) : DASH;
}

/** Seconds → "71.2 s", "3 min 12 s", "1 h 04 min". */
export function seconds(s: number | null | undefined): string {
  if (s == null || Number.isNaN(s)) return DASH;
  if (s < 59.95) return `${s.toFixed(1)} s`;
  const pad = (n: number) => String(n).padStart(2, '0');
  const t = Math.round(s);
  if (t < 3600) return `${Math.floor(t / 60)} min ${pad(t % 60)} s`;
  const mins = Math.round(s / 60);
  return `${Math.floor(mins / 60)} h ${pad(mins % 60)} min`;
}

/** A dimension key as people say it: "768" → "768 dims", "lexical" → "lexical". */
export function dimLabel(dim: string): string {
  return /^\d+$/.test(dim) ? `${dim} dims` : dim;
}

/** Dimension keys, biggest first, lexical last. */
export function sortDims(dims: string[]): string[] {
  return [...dims].sort((a, b) => (Number(b) || -1) - (Number(a) || -1));
}

/** Runs that can sit beside this one under `?vs=`: same split and eval split, not itself, not training. */
export function comparable(run: RunSummary, runs: RunSummary[]): RunSummary[] {
  return runs.filter(
    (r) =>
      r.run_id !== run.run_id &&
      r.kind !== 'train' &&
      r.split === run.split &&
      r.eval_split != null &&
      r.eval_split === run.eval_split,
  );
}

/** This run against the other at k = 10, as the API's `change` names it. */
export function changeOf(rank: number, vsRank: number, k = 10): Change {
  const a = rank <= k;
  const b = vsRank <= k;
  if (a && !b) return 'fixed';
  if (!a && b) return 'broken';
  return a ? 'same_hit' : 'same_miss';
}

export const CHANGE_TEXT: Record<Change, string> = {
  fixed: '▲ fixed',
  broken: '▼ broken',
  same_hit: '● both hit',
  same_miss: '○ both miss',
};

export type RankRow = { kind: 'hit'; hit: Hit } | { kind: 'gap'; from: number; to: number };

/**
 * The top 10, then any golden passage that fell below it, with a dashed gap row standing for
 * the ranks in between: 1 … 10, ranks 11–13, 14 (golden).
 */
export function withGaps(top: Hit[], goldenBelow: Hit[]): RankRow[] {
  const rows: RankRow[] = top.map((hit) => ({ kind: 'hit', hit }));
  let last = top.length ? top[top.length - 1].rank : 0;
  const seen = new Set(top.map((h) => h.pid));
  for (const hit of [...goldenBelow].sort((x, y) => x.rank - y.rank)) {
    if (seen.has(hit.pid) || hit.rank <= last) continue;
    if (hit.rank > last + 1) rows.push({ kind: 'gap', from: last + 1, to: hit.rank - 1 });
    rows.push({ kind: 'hit', hit });
    last = hit.rank;
  }
  return rows;
}

/** "ranks 11–13" or "rank 11". */
export function gapText(from: number, to: number): string {
  return from === to ? `rank ${from}` : `ranks ${from}–${to}`;
}

/** Bars scaled to the largest value: [3, 6] → [0.5, 1]. All-zero input gives zeros. */
export function scale(values: number[]): number[] {
  const max = Math.max(0, ...values);
  return values.map((v) => (max > 0 ? v / max : 0));
}
