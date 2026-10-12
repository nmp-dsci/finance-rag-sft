/**
 * The API contract, typed. Every shape here is `docs/api.md` verbatim; the viewer reads
 * nothing else. Dimension sizes are strings ("768", "384", "64"); BM25's one key is "lexical".
 */
import { useEffect, useState } from 'react';

// ── shared types ──────────────────────────────────────────────────────────
export type MetricKey = 'recall@1' | 'recall@5' | 'recall@10' | 'ndcg@10' | 'mrr@10';
export type Metrics = Record<MetricKey, number>;

export type Retriever = {
  type: 'dense' | 'bm25';
  model: string | null;
  revision: string | null;
  checkpoint: string | null;
  train_run: string | null;
  label: string;
};

export type RunKind = 'smoke' | 'baseline' | 'check' | 'train' | 'eval';
export type EvalSplit = 'train' | 'val' | 'test';

export type RunSummary = {
  run_id: string;
  kind: RunKind;
  created: string;
  label: string;
  retriever: Retriever | null;
  split: string;
  eval_split: EvalSplit | null;
  dims: string[];
  n_queries: number;
  metrics: Record<string, Metrics>;
  mlflow_run_id: string | null;
  mlflow_url: string | null;
  note: string;
};

export type TrainEpoch = {
  epoch: number;
  step: number;
  checkpoint: string;
  val: Record<string, Metrics>;
  loss: number | null;
};

export type Run = RunSummary & {
  code_sha: string;
  split_fingerprint: string;
  n_passages: number;
  k: number;
  timing: Record<string, number>;
  device: string | null;
  train?: {
    config: Record<string, unknown>;
    epochs: TrainEpoch[];
    chosen: { epoch: number; checkpoint: string; by: string };
    train_seconds: number;
    n_pairs: number;
    loss_curve?: { step: number; loss: number }[];
  };
  check?: { reference: number; got: number; metric: string; note: string };
};

export type Hit = { rank: number; pid: string; text: string; score: number; golden: boolean };

// ── health and overview ───────────────────────────────────────────────────
export type Health = { ok: true; code_sha: string; mlflow: 'up' | 'down' };

export type DeltaKey = 'recall@10' | 'ndcg@10' | 'mrr@10';
export type Compare = {
  a: RunSummary;
  b: RunSummary;
  dim: string;
  n: number;
  fixed: number;
  broken: number;
  both_hit: number;
  both_miss: number;
  p_exact: number;
  delta: Record<DeltaKey, { value: number; ci_low: number; ci_high: number }>;
  rank_shift: { bucket: string; count: number }[];
};

export type Milestone = { id: string; title: string; status: 'done' | 'pending'; detail: string };

export type Overview = {
  project: 'finance-rag-sft';
  code_sha: string;
  split: { version: string; passages: number; queries: { train: number; val: number; test: number } } | null;
  base: RunSummary | null;
  tuned: RunSummary | null;
  compare: Compare | null;
  baselines: RunSummary[];
  milestones: Milestone[];
};

// ── split / data explorer ─────────────────────────────────────────────────
export type Hist = [number, number, number][];
export type LengthStats = { median: number; p95: number; max: number; hist: Hist };

export type SplitStats = {
  version: 'v1';
  dataset: string;
  revision: string;
  seed: number;
  fingerprint: string;
  rows: number;
  passages: number;
  queries: { train: number; val: number; test: number; total: number };
  pairs: { train: number; val: number; test: number };
  groups: number;
  duplicates: { passages_repeated: number; max_repeats: number; rows_sharing: number; questions_repeated: number };
  multi_golden_queries: number;
  leak_check: { ok: boolean; train_test_overlap: number; train_val_overlap: number; val_test_overlap: number };
  truncated_passages: number;
  lengths: { passage_words: LengthStats; question_words: LengthStats };
};

export type SplitName = 'train' | 'val' | 'test' | 'all';

export type QueryListItem = { qid: string; split: string; text: string; golden: string[] };
export type QueryList = { total: number; items: QueryListItem[] };

export type QueryDetail = {
  qid: string;
  split: string;
  text: string;
  golden: { pid: string; text: string; n_questions: number }[];
  runs: { run_id: string; label: string; dim: string; golden_rank: number }[];
};

export type PassageListItem = { pid: string; split: string; text: string; words: number; n_questions: number };
export type PassageList = { total: number; items: PassageListItem[] };

export type PassageDetail = {
  pid: string;
  split: string;
  text: string;
  words: number;
  rows: number;
  questions: { qid: string; split: string; text: string }[];
};

// ── runs ──────────────────────────────────────────────────────────────────
export type QueryFilter = 'all' | 'hit' | 'miss' | 'fixed' | 'broken';
export type Change = 'fixed' | 'broken' | 'same_hit' | 'same_miss';

export type RunQueryRow = {
  qid: string;
  text: string;
  golden_rank: number;
  hit: boolean;
  vs_golden_rank?: number;
  change?: Change;
};
export type RunQueries = { total: number; dim: string; items: RunQueryRow[] };

export type RankedList = { run_id: string; label: string; golden_rank: number; top: Hit[]; golden_hits: Hit[] };
export type RunQuestion = {
  qid: string;
  text: string;
  split: string;
  golden: { pid: string; text: string }[];
  dim: string;
  run: RankedList;
  vs: RankedList | null;
};

// ── live retrieval ────────────────────────────────────────────────────────
export type RetrieverOption = { id: string; label: string; dims: string[]; ready: boolean };
export type SearchRequest = { query: string; retrievers: string[]; dim: string; k?: number };
export type SearchResponse = {
  query: string;
  matched_qid: string | null;
  golden: string[];
  results: { retriever: string; label: string; dim: string; ms: number; top: Hit[] }[];
};

// ── method: the loss's view of one batch ─────────────────────────────────
export type MethodSplit = 'train' | 'val';
export type MethodModel = {
  retriever: string;
  label: string;
  sims: number[][]; // [question i][passage j], cosine at `dim`
  p_correct: number[]; // softmax probability on the question's own passage
  loss: number[]; // MNRL per question: −log p_correct
  mean_loss: number;
  corpus_rank: number[]; // the same passage's rank among every passage in the corpus
  matryoshka: { dim: string; mean_loss: number }[];
  matryoshka_loss: number; // the sum over the five sizes: what training logs as "loss"
};
export type MethodBatch = {
  split: MethodSplit;
  n: number;
  seed: number;
  dim: string;
  scale: number;
  corpus: number;
  pairs: { qid: string; question: string; pid: string; passage: string }[];
  models: MethodModel[];
};

// ── endpoints ─────────────────────────────────────────────────────────────
type Params = Record<string, string | number | null | undefined>;

/** `?a=1&b=2`, skipping empty values; '' when nothing is set. */
export function qs(params: Params): string {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v != null && v !== '') sp.set(k, String(v));
  const s = sp.toString();
  return s ? `?${s}` : '';
}

const seg = encodeURIComponent;

export const api = {
  health: () => '/healthz',
  overview: () => '/api/overview',
  split: () => '/api/split',
  queries: (p: { split?: SplitName; q?: string; offset?: number; limit?: number }) => `/api/split/queries${qs(p)}`,
  query: (qid: string) => `/api/split/queries/${seg(qid)}`,
  passages: (p: { split?: SplitName; q?: string; offset?: number; limit?: number }) => `/api/split/passages${qs(p)}`,
  passage: (pid: string) => `/api/split/passages/${seg(pid)}`,
  runs: () => '/api/runs',
  run: (runId: string) => `/api/runs/${seg(runId)}`,
  runQueries: (
    runId: string,
    p: { dim?: string; filter?: QueryFilter; q?: string; offset?: number; limit?: number; vs?: string | null },
  ) => `/api/runs/${seg(runId)}/queries${qs(p)}`,
  runQuestion: (runId: string, qid: string, p: { dim?: string; vs?: string | null }) =>
    `/api/runs/${seg(runId)}/queries/${seg(qid)}${qs(p)}`,
  compare: (a: string, b: string, dim: string) => `/api/compare${qs({ a, b, dim })}`,
  retrievers: () => '/api/retrievers',
  search: () => '/api/search',
  methodBatch: (p: { split?: MethodSplit; n?: number; seed?: number; dim?: string }) =>
    `/api/method/batch${qs(p)}`,
};

// ── fetching ──────────────────────────────────────────────────────────────
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function read<T>(r: Response, url: string): Promise<T> {
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new ApiError((j as { detail?: string }).detail ?? `${r.status} ${url}`, r.status);
  return j as T;
}

export async function get<T>(url: string): Promise<T> {
  return read<T>(await fetch(url), url);
}

export async function post<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  return read<T>(r, url);
}

export type Got<T> = { data: T | null; error: string | null; loading: boolean };

/** GET a URL whenever it changes; null skips the request. */
export function useGet<T>(url: string | null): Got<T> {
  const [state, set] = useState<Got<T>>({ data: null, error: null, loading: !!url });
  useEffect(() => {
    if (!url) {
      set({ data: null, error: null, loading: false });
      return;
    }
    let alive = true;
    set((s) => ({ data: s.data, error: null, loading: true }));
    get<T>(url)
      .then((d) => alive && set({ data: d, error: null, loading: false }))
      .catch((e: Error) => alive && set({ data: null, error: e.message, loading: false }));
    return () => {
      alive = false;
    };
  }, [url]);
  return state;
}
