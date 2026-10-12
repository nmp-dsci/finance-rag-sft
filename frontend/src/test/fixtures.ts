/**
 * API fixtures, shaped exactly as docs/api.md says. Question and passage texts are real rows from
 * philschmid/finanical-rag-embedding-dataset; every id, rank, score and metric is made up.
 */
import type {
  Compare,
  Health,
  Hit,
  MethodBatch,
  Metrics,
  Overview,
  PassageDetail,
  PassageList,
  QueryDetail,
  QueryList,
  RetrieverOption,
  Run,
  RunQueries,
  RunQuestion,
  RunSummary,
  SearchResponse,
  SplitStats,
} from '../lib/api';

export const SHA = 'a1b2c3d';

export const RUN_CHECK = '20261012-150000-check-bge-base';
export const RUN_BM25 = '20261012-160000-baseline-bm25';
export const RUN_SMALL = '20261012-163000-baseline-bge-small';
export const RUN_BASE = '20261012-170000-baseline-bge-base';
export const RUN_TRAIN = '20261013-090000-train-bge-base';
export const RUN_TUNED = '20261013-120000-eval-bge-base-tuned';

export const QID = 'q3f9a0c1b2d4e';
export const QID_2 = 'q7e21b0c9d8a1';
export const QID_3 = 'q0c4d5e6f7a8b';
export const PID = 'p5a6b7c8d9e0f';
export const PID_2 = 'p1f2e3d4c5b6a';
export const PID_3 = 'p9a8b7c6d5e4f';

export const Q_TEXT = "How much did AT&T's common stock dividends per share total in 2023?";
export const Q2_TEXT = 'What were the total capital expenditures for Alphabet Inc. in 2023?';
export const Q3_TEXT = 'What was the dividend declared per share by AT&T in 2023?';
export const P_TEXT = 'Dividends on common stock declared by our Board of Directors totaled $1.11 per share in 2023.';
export const P2_TEXT =
  'Net premiums written increased from $11,060 million in 2022 to $12,575 million in 2023, reflecting a 13.7% increase.';
export const P3_TEXT =
  'The results of tax audits and changes in tax laws can impact the tax provision recorded by a company.';

const m = (r1: number, r5: number, r10: number, ndcg: number, mrr: number): Metrics => ({
  'recall@1': r1,
  'recall@5': r5,
  'recall@10': r10,
  'ndcg@10': ndcg,
  'mrr@10': mrr,
});

export const health: Health = { ok: true, code_sha: SHA, mlflow: 'up' };

// ── runs ──────────────────────────────────────────────────────────────────
const bgeBase = {
  type: 'dense' as const,
  model: 'BAAI/bge-base-en-v1.5',
  revision: 'a5beb1e',
  checkpoint: null,
  train_run: null,
  label: 'bge-base (base)',
};

export const baseSummary: RunSummary = {
  run_id: RUN_BASE,
  kind: 'baseline',
  created: '2026-10-12T17:00:00Z',
  label: 'bge-base (base)',
  retriever: bgeBase,
  split: 'v1',
  eval_split: 'test',
  dims: ['768', '64'],
  n_queries: 700,
  metrics: { '768': m(0.512, 0.743, 0.812, 0.651, 0.603), '64': m(0.431, 0.662, 0.734, 0.571, 0.522) },
  mlflow_run_id: 'f00dbase',
  mlflow_url: 'http://localhost:5000/#/experiments/7/runs/f00dbase',
  note: '',
};

export const bm25Summary: RunSummary = {
  run_id: RUN_BM25,
  kind: 'baseline',
  created: '2026-10-12T16:00:00Z',
  label: 'BM25',
  retriever: { type: 'bm25', model: null, revision: null, checkpoint: null, train_run: null, label: 'BM25' },
  split: 'v1',
  eval_split: 'test',
  dims: ['lexical'],
  n_queries: 700,
  metrics: { lexical: m(0.402, 0.611, 0.688, 0.532, 0.488) },
  mlflow_run_id: null,
  mlflow_url: null,
  note: 'bm25s, default k1 and b',
};

export const smallSummary: RunSummary = {
  ...baseSummary,
  run_id: RUN_SMALL,
  created: '2026-10-12T16:30:00Z',
  label: 'bge-small (base)',
  retriever: { ...bgeBase, model: 'BAAI/bge-small-en-v1.5', label: 'bge-small (base)' },
  dims: ['384', '64'],
  metrics: { '384': m(0.47, 0.71, 0.779, 0.62, 0.571), '64': m(0.4, 0.63, 0.7, 0.55, 0.5) },
  mlflow_run_id: null,
  mlflow_url: null,
};

export const trainSummary: RunSummary = {
  run_id: RUN_TRAIN,
  kind: 'train',
  created: '2026-10-13T09:00:00Z',
  label: 'bge-base fine-tune, 3 epochs',
  retriever: null,
  split: 'v1',
  eval_split: null,
  dims: [],
  n_queries: 0,
  metrics: {},
  mlflow_run_id: 'f00dtrain',
  mlflow_url: 'http://localhost:5000/#/experiments/7/runs/f00dtrain',
  note: 'MNRL with Matryoshka 768/512/256/128/64',
};

export const tunedSummary: RunSummary = {
  run_id: RUN_TUNED,
  kind: 'eval',
  created: '2026-10-13T12:00:00Z',
  label: 'bge-base tuned (epoch 2)',
  retriever: {
    ...bgeBase,
    checkpoint: `outputs/${RUN_TRAIN}/checkpoint-350`,
    train_run: RUN_TRAIN,
    label: 'bge-base tuned (epoch 2)',
  },
  split: 'v1',
  eval_split: 'test',
  dims: ['768', '64'],
  n_queries: 700,
  metrics: { '768': m(0.581, 0.801, 0.853, 0.712, 0.668), '64': m(0.522, 0.751, 0.807, 0.66, 0.611) },
  mlflow_run_id: 'f00dtuned',
  mlflow_url: 'http://localhost:5000/#/experiments/7/runs/f00dtuned',
  note: '',
};

export const checkSummary: RunSummary = {
  ...baseSummary,
  run_id: RUN_CHECK,
  kind: 'check',
  created: '2026-10-12T15:00:00Z',
  label: 'harness check: bge-base on the blog split',
  dims: ['768'],
  metrics: { '768': m(0.6, 0.8, 0.861, 0.745, 0.702) },
  mlflow_run_id: null,
  mlflow_url: null,
  note: 'reproduces the published baseline',
};

/** Newest first, as the API returns them. */
export const runs: RunSummary[] = [tunedSummary, trainSummary, baseSummary, smallSummary, bm25Summary, checkSummary];

const runExtras = {
  code_sha: SHA,
  split_fingerprint: 'sha256:9c1e0f2a',
  n_passages: 6757,
  k: 100,
  timing: { encode_corpus: 71.2, score: 0.4 },
  device: 'mps',
};

export const runBase: Run = { ...baseSummary, ...runExtras };
export const runTuned: Run = { ...tunedSummary, ...runExtras };
export const runBm25: Run = { ...bm25Summary, ...runExtras, timing: { index: 2.1, score: 0.9 }, device: null };
export const runCheck: Run = {
  ...checkSummary,
  ...runExtras,
  check: { reference: 0.86, got: 0.861, metric: 'recall@10', note: 'Within 0.5 pp of the published number.' },
};
export const runTrain: Run = {
  ...trainSummary,
  ...runExtras,
  timing: { train: 1490.5 },
  train: {
    config: { epochs: 3, lr: 2e-5, batch_size: 32, loss: 'MatryoshkaLoss(MNRL)', dims: [768, 512, 256, 128, 64] },
    epochs: [
      { epoch: 1, step: 175, checkpoint: `outputs/${RUN_TRAIN}/checkpoint-175`, val: { '768': m(0.55, 0.78, 0.84, 0.69, 0.64), '64': m(0.5, 0.72, 0.79, 0.63, 0.59) }, loss: 0.4123 },
      { epoch: 2, step: 350, checkpoint: `outputs/${RUN_TRAIN}/checkpoint-350`, val: { '768': m(0.57, 0.8, 0.861, 0.71, 0.66), '64': m(0.52, 0.75, 0.81, 0.66, 0.61) }, loss: 0.2871 },
      { epoch: 3, step: 525, checkpoint: `outputs/${RUN_TRAIN}/checkpoint-525`, val: { '768': m(0.57, 0.79, 0.855, 0.705, 0.655), '64': m(0.51, 0.74, 0.8, 0.65, 0.6) }, loss: null },
    ],
    chosen: { epoch: 2, checkpoint: `outputs/${RUN_TRAIN}/checkpoint-350`, by: 'val recall@10 at 768' },
    train_seconds: 1490.5,
    n_pairs: 5512,
    loss_curve: [
      { step: 10, loss: 1.9 },
      { step: 175, loss: 0.41 },
      { step: 350, loss: 0.29 },
      { step: 525, loss: 0.12 },
    ],
  },
};

/** `/api/method/batch`, two pairs: Q2's passage is outside the base model's top 10. */
export const methodBatch: MethodBatch = {
  split: 'train',
  n: 2,
  seed: 0,
  dim: '768',
  scale: 20,
  corpus: 6757,
  pairs: [
    { qid: QID, question: Q_TEXT, pid: PID, passage: P_TEXT },
    { qid: QID_2, question: Q2_TEXT, pid: PID_2, passage: P2_TEXT },
  ],
  models: [
    {
      retriever: 'bge-base',
      label: 'bge-base (base)',
      sims: [
        [0.72, 0.55],
        [0.52, 0.76],
      ],
      p_correct: [0.9734, 0.9945],
      loss: [0.027, 0.0055],
      mean_loss: 0.0162,
      corpus_rank: [2, 11],
      matryoshka: ['768', '512', '256', '128', '64'].map((dim) => ({ dim, mean_loss: 0.02 })),
      matryoshka_loss: 0.1,
    },
    {
      retriever: `tuned:${RUN_TRAIN}`,
      label: 'bge-base tuned (epoch 2)',
      sims: [
        [0.7, 0.2],
        [0.25, 0.9],
      ],
      p_correct: [0.9999, 1],
      loss: [0.0001, 0],
      mean_loss: 0.0001,
      corpus_rank: [1, 1],
      matryoshka: ['768', '512', '256', '128', '64'].map((dim) => ({ dim, mean_loss: 0.0001 })),
      matryoshka_loss: 0.0005,
    },
  ],
};

export const runsById: Record<string, Run> = {
  [RUN_BASE]: runBase,
  [RUN_TUNED]: runTuned,
  [RUN_BM25]: runBm25,
  [RUN_CHECK]: runCheck,
  [RUN_TRAIN]: runTrain,
};

// ── compare and overview ──────────────────────────────────────────────────
export const compare: Compare = {
  a: tunedSummary,
  b: baseSummary,
  dim: '768',
  n: 700,
  fixed: 48,
  broken: 19,
  both_hit: 549,
  both_miss: 84,
  p_exact: 0.00042,
  delta: {
    'recall@10': { value: 0.041, ci_low: 0.018, ci_high: 0.065 },
    'ndcg@10': { value: 0.061, ci_low: 0.04, ci_high: 0.083 },
    'mrr@10': { value: 0.065, ci_low: 0.041, ci_high: 0.089 },
  },
  rank_shift: [
    { bucket: 'up 10+', count: 31 },
    { bucket: 'up 4-9', count: 57 },
    { bucket: 'up 1-3', count: 140 },
    { bucket: 'same', count: 301 },
    { bucket: 'down 1-3', count: 112 },
    { bucket: 'down 4-9', count: 38 },
    { bucket: 'down 10+', count: 21 },
  ],
};

export const overview: Overview = {
  project: 'finance-rag-sft',
  code_sha: SHA,
  split: { version: 'v1', passages: 6757, queries: { train: 5512, val: 691, test: 700 } },
  base: baseSummary,
  tuned: tunedSummary,
  compare,
  baselines: [baseSummary, smallSummary, bm25Summary],
  milestones: [
    { id: 'M1', title: 'Frozen split v1', status: 'done', detail: '6,757 passages, grouped by passage' },
    { id: 'M2', title: 'Baselines and harness check', status: 'done', detail: 'BM25, bge-small, bge-base on test' },
    { id: 'M3', title: 'Fine-tune bge-base', status: 'done', detail: 'epoch 2 chosen on val' },
    { id: 'M4', title: 'FiQA2018 regression check', status: 'pending', detail: '' },
  ],
};

/** Before any run: the split exists, nothing is scored. */
export const overviewEmpty: Overview = {
  project: 'finance-rag-sft',
  code_sha: SHA,
  split: null,
  base: null,
  tuned: null,
  compare: null,
  baselines: [],
  milestones: [
    { id: 'M1', title: 'Frozen split v1', status: 'pending', detail: '' },
    { id: 'M2', title: 'Baselines and harness check', status: 'pending', detail: '' },
    { id: 'M3', title: 'Fine-tune bge-base', status: 'pending', detail: '' },
  ],
};

// ── split ─────────────────────────────────────────────────────────────────
export const split: SplitStats = {
  version: 'v1',
  dataset: 'philschmid/finanical-rag-embedding-dataset',
  revision: 'e0b17819cf52d444066c99f4a176f5717e066300',
  seed: 42,
  fingerprint: 'sha256:9c1e0f2a',
  rows: 7000,
  passages: 6757,
  queries: { train: 5512, val: 691, test: 700, total: 6903 },
  pairs: { train: 5580, val: 699, test: 712 },
  groups: 6757,
  duplicates: { passages_repeated: 214, max_repeats: 4, rows_sharing: 457, questions_repeated: 97 },
  multi_golden_queries: 88,
  leak_check: { ok: true, train_test_overlap: 0, train_val_overlap: 0, val_test_overlap: 0 },
  truncated_passages: 3,
  lengths: {
    passage_words: {
      median: 38,
      p95: 82,
      max: 612,
      hist: [
        [0, 20, 610],
        [20, 40, 2950],
        [40, 60, 1980],
        [60, 80, 790],
        [80, 100, 280],
        [100, 120, 97],
        [120, 140, 50],
      ],
    },
    question_words: {
      median: 14,
      p95: 24,
      max: 51,
      hist: [
        [0, 5, 40],
        [5, 10, 1300],
        [10, 15, 2900],
        [15, 20, 1800],
        [20, 25, 650],
        [25, 30, 213],
      ],
    },
  },
};

export const queryList: QueryList = {
  total: 6903,
  items: [
    { qid: QID, split: 'test', text: Q_TEXT, golden: [PID] },
    { qid: QID_2, split: 'train', text: Q2_TEXT, golden: [PID_2, PID_3] },
  ],
};

export const queryDetail: QueryDetail = {
  qid: QID,
  split: 'test',
  text: Q_TEXT,
  golden: [{ pid: PID, text: P_TEXT, n_questions: 2 }],
  runs: [
    { run_id: RUN_BASE, label: 'bge-base (base)', dim: '768', golden_rank: 14 },
    { run_id: RUN_TUNED, label: 'bge-base tuned (epoch 2)', dim: '768', golden_rank: 1 },
  ],
};

export const passageList: PassageList = {
  total: 6757,
  items: [
    { pid: PID, split: 'test', text: P_TEXT, words: 16, n_questions: 2 },
    { pid: PID_2, split: 'train', text: P2_TEXT, words: 19, n_questions: 1 },
  ],
};

export const passageDetail: PassageDetail = {
  pid: PID,
  split: 'test',
  text: P_TEXT,
  words: 16,
  rows: 2,
  questions: [
    { qid: QID, split: 'test', text: Q_TEXT },
    { qid: QID_3, split: 'test', text: Q3_TEXT },
  ],
};

// ── per-question ──────────────────────────────────────────────────────────
export const runQueries: RunQueries = {
  total: 700,
  dim: '768',
  items: [
    { qid: QID, text: Q_TEXT, golden_rank: 1, hit: true },
    { qid: QID_2, text: Q2_TEXT, golden_rank: 37, hit: false },
  ],
};

export const runQueriesVs: RunQueries = {
  total: 700,
  dim: '768',
  items: [
    { qid: QID, text: Q_TEXT, golden_rank: 1, hit: true, vs_golden_rank: 14, change: 'fixed' },
    { qid: QID_2, text: Q2_TEXT, golden_rank: 37, hit: false, vs_golden_rank: 6, change: 'broken' },
  ],
};

const hit = (rank: number, pid: string, text: string, score: number, golden = false): Hit => ({
  rank,
  pid,
  text,
  score,
  golden,
});
const filler = (from: number, to: number, base: number): Hit[] =>
  Array.from({ length: to - from + 1 }, (_, i) =>
    hit(from + i, `p${(from + i).toString(16).padStart(12, '0')}`, `Another passage at rank ${from + i}.`, base - i * 0.01),
  );

export const tunedTop: Hit[] = [hit(1, PID, P_TEXT, 0.842, true), hit(2, PID_2, P2_TEXT, 0.661), hit(3, PID_3, P3_TEXT, 0.612), ...filler(4, 10, 0.6)];
export const baseTop: Hit[] = [hit(1, PID_3, P3_TEXT, 0.712), hit(2, PID_2, P2_TEXT, 0.691), ...filler(3, 10, 0.68)];

export const runQuestion: RunQuestion = {
  qid: QID,
  text: Q_TEXT,
  split: 'test',
  golden: [{ pid: PID, text: P_TEXT }],
  dim: '768',
  run: { run_id: RUN_TUNED, label: 'bge-base tuned (epoch 2)', golden_rank: 1, top: tunedTop, golden_hits: [] },
  vs: null,
};

export const runQuestionVs: RunQuestion = {
  ...runQuestion,
  vs: {
    run_id: RUN_BASE,
    label: 'bge-base (base)',
    golden_rank: 14,
    top: baseTop,
    golden_hits: [hit(14, PID, P_TEXT, 0.62, true)],
  },
};

// ── live retrieval ────────────────────────────────────────────────────────
export const retrievers: RetrieverOption[] = [
  { id: 'bm25', label: 'BM25', dims: ['lexical'], ready: true },
  { id: 'bge-small', label: 'bge-small', dims: ['384', '64'], ready: false },
  { id: 'bge-base', label: 'bge-base', dims: ['768', '64'], ready: true },
  { id: `tuned:${RUN_TRAIN}`, label: 'bge-base tuned (epoch 2)', dims: ['768', '64'], ready: true },
];

export const searchResult: SearchResponse = {
  query: Q_TEXT,
  matched_qid: QID,
  golden: [PID],
  results: [
    { retriever: 'bm25', label: 'BM25', dim: 'lexical', ms: 3.2, top: [hit(1, PID_2, P2_TEXT, 12.1), hit(2, PID, P_TEXT, 11.7, true)] },
    { retriever: 'bge-base', label: 'bge-base', dim: '768', ms: 18.4, top: baseTop },
    { retriever: `tuned:${RUN_TRAIN}`, label: 'bge-base tuned (epoch 2)', dim: '768', ms: 17.9, top: tunedTop },
  ],
};
