import { describe, expect, it } from 'vitest';
import type { Hit, RunSummary } from './api';
import {
  changeOf,
  comparable,
  delta,
  dimLabel,
  gapText,
  int,
  interval,
  metric,
  pText,
  pValue,
  pct,
  scale,
  score3,
  seconds,
  sortDims,
  when,
  withGaps,
} from './format';

describe('percentages and scores', () => {
  it('prints a proportion as a percentage to 1 dp', () => {
    expect(pct(0.7123)).toBe('71.2%');
    expect(pct(0.71249)).toBe('71.2%');
    expect(pct(1)).toBe('100.0%');
    expect(pct(0)).toBe('0.0%');
    expect(pct(null)).toBe('—');
    expect(pct(undefined)).toBe('—');
    expect(pct(Number.NaN)).toBe('—');
  });
  it('prints nDCG and MRR as scores to 3 dp, recall as a percentage', () => {
    expect(score3(0.63412)).toBe('0.634');
    expect(metric('ndcg@10', 0.6345)).toBe(score3(0.6345));
    expect(metric('mrr@10', 0.5)).toBe('0.500');
    expect(metric('recall@10', 0.853)).toBe('85.3%');
    expect(metric('recall@1', null)).toBe('—');
  });
});

describe('p-values', () => {
  it('floors at < 0.001, else 3 dp', () => {
    expect(pValue(0.00042)).toBe('< 0.001');
    expect(pValue(0.001)).toBe('0.001');
    expect(pValue(0.0496)).toBe('0.050');
    expect(pValue(0.049)).toBe('0.049');
    expect(pValue(1)).toBe('1.000');
    expect(pValue(null)).toBe('—');
  });
  it('reads as a sentence fragment', () => {
    expect(pText(0.00042)).toBe('p < 0.001');
    expect(pText(0.032)).toBe('p = 0.032');
  });
});

describe('paired differences', () => {
  it('prints recall deltas in signed percentage points, scores to 3 dp', () => {
    expect(delta('recall@10', 0.041)).toBe('+4.1 pp');
    expect(delta('recall@10', -0.012)).toBe('−1.2 pp');
    expect(delta('recall@10', 0)).toBe('0.0 pp');
    expect(delta('recall@10', -0.0001)).toBe('0.0 pp');
    expect(delta('ndcg@10', 0.0612)).toBe('+0.061');
    expect(delta('mrr@10', -0.004)).toBe('−0.004');
    expect(delta('mrr@10', null)).toBe('—');
  });
  it('prints a 95% interval with the unit once', () => {
    expect(interval('recall@10', 0.018, 0.065)).toBe('[+1.8, +6.5] pp');
    expect(interval('recall@10', -0.01, 0.02)).toBe('[−1.0, +2.0] pp');
    expect(interval('ndcg@10', -0.012, 0.04)).toBe('[−0.012, +0.040]');
  });
});

describe('counts, times and dims', () => {
  it('formats integers, dates and durations', () => {
    expect(int(6757)).toBe('6,757');
    expect(int(null)).toBe('—');
    expect(when('2026-10-12T17:00:00Z')).toBe('2026-10-12 17:00');
    expect(seconds(0.4)).toBe('0.4 s');
    expect(seconds(59.5)).toBe('59.5 s');
    expect(seconds(71.2)).toBe('1 min 11 s');
    expect(seconds(119.6)).toBe('2 min 00 s');
    expect(seconds(192)).toBe('3 min 12 s');
    expect(seconds(3840)).toBe('1 h 04 min');
  });
  it('names and orders dimension sizes', () => {
    expect(dimLabel('768')).toBe('768 dims');
    expect(dimLabel('lexical')).toBe('lexical');
    expect(sortDims(['64', '768', '384'])).toEqual(['768', '384', '64']);
    expect(sortDims(['lexical'])).toEqual(['lexical']);
  });
});

describe('compare helpers', () => {
  it('names the change at k = 10 the way the API does', () => {
    expect(changeOf(1, 14)).toBe('fixed');
    expect(changeOf(37, 6)).toBe('broken');
    expect(changeOf(10, 3)).toBe('same_hit');
    expect(changeOf(11, 40)).toBe('same_miss');
  });

  const run = (over: Partial<RunSummary>): RunSummary => ({
    run_id: 'r',
    kind: 'eval',
    created: '2026-10-12T00:00:00Z',
    label: 'r',
    retriever: null,
    split: 'v1',
    eval_split: 'test',
    dims: ['768'],
    n_queries: 700,
    metrics: {},
    mlflow_run_id: null,
    mlflow_url: null,
    note: '',
    ...over,
  });
  it('offers runs on the same split and eval split, never itself or a train run', () => {
    const me = run({ run_id: 'me' });
    const all = [
      me,
      run({ run_id: 'base', kind: 'baseline' }),
      run({ run_id: 'val', eval_split: 'val' }),
      run({ run_id: 'v2', split: 'v2' }),
      run({ run_id: 'train', kind: 'train', eval_split: null }),
    ];
    expect(comparable(me, all).map((r) => r.run_id)).toEqual(['base']);
  });
});

describe('ranked lists with gaps', () => {
  const h = (rank: number, pid = `p${rank}`, golden = false): Hit => ({ rank, pid, text: pid, score: 1 / rank, golden });
  const top = Array.from({ length: 10 }, (_, i) => h(i + 1));

  it('is the top 10 alone when every golden passage is inside it', () => {
    expect(withGaps(top, [])).toHaveLength(10);
  });
  it('adds a gap row down to each golden passage below the top 10', () => {
    const rows = withGaps(top, [h(14, 'g1', true), h(12, 'g2', true), h(15, 'g3', true)]);
    expect(rows.slice(10)).toEqual([
      { kind: 'gap', from: 11, to: 11 },
      { kind: 'hit', hit: h(12, 'g2', true) },
      { kind: 'gap', from: 13, to: 13 },
      { kind: 'hit', hit: h(14, 'g1', true) },
      { kind: 'hit', hit: h(15, 'g3', true) },
    ]);
  });
  it('ignores a golden hit already in the top 10', () => {
    expect(withGaps(top, [h(3)])).toHaveLength(10);
  });
  it('words a gap', () => {
    expect(gapText(11, 13)).toBe('ranks 11–13');
    expect(gapText(11, 11)).toBe('rank 11');
  });
});

describe('scale', () => {
  it('scales to the largest value', () => {
    expect(scale([3, 6, 0])).toEqual([0.5, 1, 0]);
    expect(scale([0, 0])).toEqual([0, 0]);
    expect(scale([])).toEqual([]);
  });
});
