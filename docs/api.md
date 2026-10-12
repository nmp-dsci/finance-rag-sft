# API contract — `finrag serve` (FastAPI, :8084)

The viewer (`frontend/`) reads only this API. The API reads only `data/splits/v1/`,
the pinned dataset in `data/cache/`, and run folders under `runs/`. It never queries
MLflow; it links to it (`mlflow_url`).

Conventions

- IDs: passages `p` + 12 hex (sha1 of the normalised text), questions `q` + 12 hex.
- Dimension sizes are strings in JSON keys: `"768"`, `"384"`, `"64"`. BM25 has one key, `"lexical"`.
- `recall@k` means **at least one golden passage in the top k** (hit@k). `ndcg@10` uses binary
  relevance over all golden passages; `mrr@10` uses the first golden passage.
- `golden_rank` is 1-based over the **whole** corpus (6,757 passages). `null` never occurs.
- Errors: HTTP 4xx with `{"detail": "..."}`.

## Types

```ts
type Metrics = { "recall@1": number; "recall@5": number; "recall@10": number; "ndcg@10": number; "mrr@10": number };
type Retriever = {
  type: "dense" | "bm25";
  model: string | null;          // "BAAI/bge-base-en-v1.5", null for bm25
  revision: string | null;
  checkpoint: string | null;     // "outputs/<train_run_id>/checkpoint-525" for tuned models
  train_run: string | null;      // the train run that produced the checkpoint
  label: string;                 // "bge-base (base)", "bge-base tuned (epoch 3)", "BM25"
};
type RunSummary = {
  run_id: string;                // "20261012-170000-baseline-bge-base"
  kind: "smoke" | "baseline" | "check" | "train" | "eval";
  created: string;               // ISO 8601
  label: string;
  retriever: Retriever | null;   // null for train runs
  split: string;                 // "v1"
  eval_split: "train" | "val" | "test" | null;
  dims: string[];                // ["768","64"] | ["lexical"] | []
  n_queries: number;
  metrics: Record<string, Metrics>;   // keyed by dim; {} for train runs
  mlflow_run_id: string | null;
  mlflow_url: string | null;
  note: string;
};
type Run = RunSummary & {
  code_sha: string;
  split_fingerprint: string;
  n_passages: number;
  k: number;
  timing: Record<string, number>;     // seconds, e.g. {"encode_corpus": 71.2, "score": 0.4}
  device: string | null;              // "mps" | "cuda" | "cpu"
  train?: {                           // kind === "train" only
    config: Record<string, unknown>;
    epochs: { epoch: number; step: number; checkpoint: string; val: Record<string, Metrics>; loss: number | null }[];
    chosen: { epoch: number; checkpoint: string; by: string };   // by: "val recall@10 at 768"
    train_seconds: number;
    n_pairs: number;
  };
  check?: { reference: number; got: number; metric: string; note: string };  // kind === "check"
};
type Hit = { rank: number; pid: string; text: string; score: number; golden: boolean };
```

## Health and overview

`GET /healthz` → `{ ok: true, code_sha: string, mlflow: "up" | "down" }`

`GET /api/overview` →
```ts
{
  project: "finance-rag-sft";
  code_sha: string;
  split: { version: string; passages: number; queries: { train: number; val: number; test: number } } | null;
  base: RunSummary | null;         // latest test-split eval of the untouched bge-base
  tuned: RunSummary | null;        // latest test-split eval of a tuned checkpoint
  compare: Compare | null;         // tuned vs base at "768", when both exist
  baselines: RunSummary[];         // every kind=baseline run on test, newest first
  milestones: { id: string; title: string; status: "done" | "pending"; detail: string }[];
}
```

## Split / data explorer

`GET /api/split` →
```ts
{
  version: "v1"; dataset: string; revision: string; seed: number; fingerprint: string;
  rows: number;                               // 7000
  passages: number;                           // 6757
  queries: { train: number; val: number; test: number; total: number };
  pairs: { train: number; val: number; test: number };   // (question, golden passage) pairs
  groups: number;                             // passage groups the split was dealt by
  duplicates: { passages_repeated: number; max_repeats: number; rows_sharing: number; questions_repeated: number };
  multi_golden_queries: number;
  leak_check: { ok: boolean; train_test_overlap: number; train_val_overlap: number; val_test_overlap: number };
  truncated_passages: number;                 // passages over 512 bge tokens
  lengths: {
    passage_words: { median: number; p95: number; max: number; hist: [number, number, number][] };   // [lo, hi, count]
    question_words: { median: number; p95: number; max: number; hist: [number, number, number][] };
  };
}
```

`GET /api/split/queries?split=train|val|test|all&q=<text>&offset=0&limit=50` →
`{ total: number; items: { qid: string; split: string; text: string; golden: string[] }[] }`

`GET /api/split/queries/{qid}` →
```ts
{
  qid: string; split: string; text: string;
  golden: { pid: string; text: string; n_questions: number }[];
  runs: { run_id: string; label: string; dim: string; golden_rank: number }[];   // test/val runs that scored it
}
```

`GET /api/split/passages?q=<text>&split=…&offset=0&limit=50` →
`{ total: number; items: { pid: string; split: string; text: string; words: number; n_questions: number }[] }`

`GET /api/split/passages/{pid}` →
`{ pid: string; split: string; text: string; words: number; rows: number; questions: { qid: string; split: string; text: string }[] }`
(`rows` = how many of the 7,000 dataset rows had this passage.)

## Runs

`GET /api/runs` → `RunSummary[]` (newest first)

`GET /api/runs/{run_id}` → `Run`

`GET /api/runs/{run_id}/queries?dim=768&filter=all|hit|miss|fixed|broken&q=&offset=0&limit=100&vs=<run_id>` →
```ts
{
  total: number;
  dim: string;
  items: {
    qid: string; text: string; golden_rank: number; hit: boolean;     // hit = golden_rank <= 10
    vs_golden_rank?: number;                                         // with vs=
    change?: "fixed" | "broken" | "same_hit" | "same_miss";          // relative to vs (this run vs the other)
  }[];
}
```
`fixed` / `broken` filters need `vs`. "fixed" = this run hits, `vs` misses.

`GET /api/runs/{run_id}/queries/{qid}?dim=768&vs=<run_id>` →
```ts
{
  qid: string; text: string; split: string;
  golden: { pid: string; text: string }[];
  dim: string;
  run: { run_id: string; label: string; golden_rank: number; top: Hit[]; golden_hits: Hit[] };  // golden_hits: golden passages outside the top 10, with their rank
  vs: { run_id: string; label: string; golden_rank: number; top: Hit[]; golden_hits: Hit[] } | null;
}
```

`GET /api/compare?a=<run_id>&b=<run_id>&dim=768` → `Compare`
```ts
type Compare = {
  a: RunSummary; b: RunSummary; dim: string; n: number;
  fixed: number; broken: number; both_hit: number; both_miss: number;   // fixed = a hits, b misses
  p_exact: number;                                                     // exact McNemar (binomial) on fixed vs broken
  delta: Record<"recall@10" | "ndcg@10" | "mrr@10", { value: number; ci_low: number; ci_high: number }>;  // a - b, paired bootstrap 95%
  rank_shift: { bucket: string; count: number }[];   // "up 10+", "up 4-9", "up 1-3", "same", "down 1-3", "down 4-9", "down 10+"
};
```
Both runs must share `split`, `eval_split` and the dim; otherwise 400.

## Live retrieval

`GET /api/retrievers` →
`{ id: string; label: string; dims: string[]; ready: boolean }[]`
e.g. `bm25` (`["lexical"]`), `bge-small` (`["384","64"]`), `bge-base` (`["768","64"]`),
`tuned:<train_run_id>` (`["768","64"]`). `ready` = corpus embeddings already cached.

`POST /api/search` body `{ query: string; retrievers: string[]; dim: string; k?: number }` →
```ts
{
  query: string;
  matched_qid: string | null;      // when the text equals a dataset question
  golden: string[];                // its golden passages ([] when unmatched)
  results: { retriever: string; label: string; dim: string; ms: number; top: Hit[] }[];
}
```
`dim` applies to dense retrievers; BM25 ignores it and reports `"lexical"`.
