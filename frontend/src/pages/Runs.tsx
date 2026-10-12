import { Link, useParams } from 'react-router-dom';
import {
  type Compare,
  type QueryFilter,
  type Run as RunData,
  type RunQueries,
  type RunSummary,
  api,
  useGet,
} from '../lib/api';
import { CompareBody } from '../lib/compare';
import { DASH, comparable, dimLabel, int, metric, score3, seconds, sortDims, when } from '../lib/format';
import {
  ChangeBadge,
  Choice,
  Empty,
  ErrorNote,
  HitBadge,
  KindBadge,
  Loading,
  MetricsTable,
  MlflowLink,
  Pager,
  RunsTable,
  SearchBox,
  SplitBadge,
} from '../lib/ui';
import { intLens, runPath, runQuestionPath, runsPath, useLens } from '../lib/url';

/** `/runs` — every run, newest first. */
export function Runs() {
  const { data: runs, error } = useGet<RunSummary[]>(api.runs());
  return (
    <>
      <p className="eyebrow">Runs · split v1</p>
      <h1>
        Every run, <em>newest first</em>
      </h1>
      <p className="lede">
        One run scores one retriever on one split, at every dimension size it has. Training runs pick a checkpoint on
        val; their eval runs score it on test. Recall@10 is the share of questions with a golden passage in the top 10.
      </p>
      {!runs ? (
        <Loading error={error} what="the runs" />
      ) : runs.length === 0 ? (
        <Empty>
          No runs yet. Runs appear here once the API finds run folders under <code>runs/</code> — M2 adds the
          baselines.
        </Empty>
      ) : (
        <RunsTable runs={runs} />
      )}
    </>
  );
}

const LIMIT = 100;
const FILTERS: QueryFilter[] = ['all', 'hit', 'miss', 'fixed', 'broken'];

/** `/runs/:runId` — one run: metrics per dim, training or check detail, every question; `?vs=` compares. */
export function Run() {
  const { runId = '' } = useParams();
  const [lens, setLens] = useLens();
  const { data: run, error } = useGet<RunData>(api.run(runId));
  const { data: runs } = useGet<RunSummary[]>(api.runs());
  if (!run || run.run_id !== runId) return <Loading error={error} what={`run ${runId}`} />;

  const dims = sortDims(run.dims.length ? run.dims : Object.keys(run.metrics));
  const asked = lens.get('dim');
  const dim = asked && dims.includes(asked) ? asked : (dims[0] ?? '');
  const vs = lens.get('vs') || null;
  const scored = dims.length > 0 && run.kind !== 'train';

  return (
    <>
      <p className="crumb">
        <Link to={runsPath()}>runs</Link> / {run.run_id}
        {vs && ` ?vs=${vs}`}
        {dim && ` · ${dimLabel(dim)}`}
      </p>
      <h1>{run.label}</h1>
      <div className="row badges">
        <KindBadge kind={run.kind} />
        <span className="badge">split {run.split}</span>
        {run.eval_split && (
          <>
            <span className="tag">scored on</span> <SplitBadge split={run.eval_split} />
          </>
        )}
        {run.device && <span className="badge">{run.device}</span>}
        <MlflowLink url={run.mlflow_url} />
      </div>
      {run.note && <p className="lede">{run.note}</p>}
      <RunFacts run={run} runs={runs ?? []} />

      {Object.keys(run.metrics).length > 0 && (
        <section>
          <span className="eyebrow">Metrics</span>
          <h2>Every metric, per dimension size</h2>
          <MetricsTable metrics={run.metrics} caption={`Metrics for ${run.label}`} />
          <p className="small">
            Recall@k: a golden passage in the top k (hit@k). nDCG@10 counts every golden passage; MRR@10 the first.
            Over {int(run.n_queries)} questions and {int(run.n_passages)} passages.
          </p>
        </section>
      )}

      {run.check && <CheckPanel check={run.check} />}
      {run.train && <TrainPanel run={run} runs={runs ?? []} />}

      {scored && (
        <section>
          <span className="eyebrow">Compare lens</span>
          <h2>{vs ? 'This run against another' : 'Compare with another run'}</h2>
          <CompareWith
            run={run}
            runs={runs}
            dim={dim}
            vs={vs}
            onPick={(next) => {
              const f = lens.get('filter');
              setLens({ vs: next, offset: null, filter: !next && (f === 'fixed' || f === 'broken') ? null : f });
            }}
          />
          {vs && <ComparePanel runId={run.run_id} vs={vs} dim={dim} />}
        </section>
      )}

      {scored && (
        <section>
          <span className="eyebrow">Questions</span>
          <h2>Every question's golden rank</h2>
          <QueryTable runId={run.run_id} dims={dims} dim={dim} vs={vs} />
        </section>
      )}
    </>
  );
}

function RunFacts({ run, runs }: { run: RunData; runs: RunSummary[] }) {
  const r = run.retriever;
  const timing = Object.entries(run.timing);
  const trainRun = r?.train_run ? runs.find((x) => x.run_id === r.train_run) : null;
  return (
    <details className="facts" open>
      <summary className="label">Setup</summary>
      <dl className="kv">
        {r && (
          <>
            <dt>retriever</dt>
            <dd>
              {r.label} <span className="tag">({r.type})</span>
            </dd>
            {r.model && (
              <>
                <dt>model</dt>
                <dd className="mono">
                  {r.model}
                  {r.revision && <span className="tag"> @ {r.revision}</span>}
                </dd>
              </>
            )}
            {r.checkpoint && (
              <>
                <dt>checkpoint</dt>
                <dd className="mono">{r.checkpoint}</dd>
              </>
            )}
            {r.train_run && (
              <>
                <dt>trained by</dt>
                <dd>
                  <Link to={runPath(r.train_run)}>{trainRun?.label ?? r.train_run}</Link>
                </dd>
              </>
            )}
          </>
        )}
        <dt>run id</dt>
        <dd className="mono">{run.run_id}</dd>
        <dt>created</dt>
        <dd>{when(run.created)}</dd>
        <dt>split</dt>
        <dd>
          {run.split} · <span className="mono">{run.split_fingerprint}</span>
        </dd>
        <dt>corpus</dt>
        <dd>
          {int(run.n_passages)} passages · {int(run.n_queries)} questions · top {run.k} kept
        </dd>
        <dt>code</dt>
        <dd className="mono">{run.code_sha}</dd>
        {timing.length > 0 && (
          <>
            <dt>timing</dt>
            <dd>{timing.map(([k, v]) => `${k.replaceAll('_', ' ')} ${seconds(v)}`).join(' · ')}</dd>
          </>
        )}
        {run.mlflow_run_id && (
          <>
            <dt>MLflow run</dt>
            <dd className="mono">
              {run.mlflow_url ? (
                <a href={run.mlflow_url} target="_blank" rel="noreferrer">
                  {run.mlflow_run_id} ↗
                </a>
              ) : (
                run.mlflow_run_id
              )}
            </dd>
          </>
        )}
      </dl>
    </details>
  );
}

function CheckPanel({ check }: { check: NonNullable<RunData['check']> }) {
  const diff = check.got - check.reference;
  const asPct = check.metric.startsWith('recall');
  const show = (x: number) => (asPct ? metric('recall@10', x) : score3(x));
  return (
    <section>
      <span className="eyebrow">Harness check</span>
      <h2>Reference against what we got</h2>
      <div className="verdict">
        <div className="stat">
          <div className="n">{show(check.reference)}</div>
          <div className="l">reference · {check.metric}</div>
        </div>
        <div className="stat accent">
          <div className="n">{show(check.got)}</div>
          <div className="l">got · {check.metric}</div>
        </div>
        <div className="stat">
          <div className="n">
            {diff >= 0 ? '+' : '−'}
            {asPct ? (Math.abs(diff) * 100).toFixed(1) : Math.abs(diff).toFixed(3)}
          </div>
          <div className="l">difference{asPct ? ' (pp)' : ''}</div>
        </div>
      </div>
      {check.note && <p className="callout">{check.note}</p>}
    </section>
  );
}

function TrainPanel({ run, runs }: { run: RunData; runs: RunSummary[] }) {
  const t = run.train!;
  const valDims = sortDims([...new Set(t.epochs.flatMap((e) => Object.keys(e.val)))]);
  const evals = runs.filter((r) => r.retriever?.train_run === run.run_id);
  const config = Object.entries(t.config);
  return (
    <section>
      <span className="eyebrow">Training</span>
      <h2>
        Epoch {t.chosen.epoch} chosen, by {t.chosen.by}
      </h2>
      <dl className="kv">
        <dt>pairs</dt>
        <dd>{int(t.n_pairs)} (question, golden passage) pairs</dd>
        <dt>time</dt>
        <dd>{seconds(t.train_seconds)}</dd>
        <dt>chosen</dt>
        <dd className="mono">{t.chosen.checkpoint}</dd>
      </dl>
      <h3>Val metrics after each epoch</h3>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th className="num">epoch</th>
              <th className="num">step</th>
              <th className="num">loss</th>
              {valDims.map((d) => (
                <th key={`r-${d}`} className="num">
                  R@10 · {d}
                </th>
              ))}
              {valDims.map((d) => (
                <th key={`n-${d}`} className="num">
                  nDCG@10 · {d}
                </th>
              ))}
              <th />
            </tr>
          </thead>
          <tbody>
            {t.epochs.map((e) => {
              const chosen = e.epoch === t.chosen.epoch;
              const c = chosen ? 'rec' : '';
              return (
                <tr key={e.epoch} className={chosen ? 'chosen' : ''}>
                  <td className={`n num ${c}`}>{e.epoch}</td>
                  <td className={`n num ${c}`}>{int(e.step)}</td>
                  <td className={`n num ${c}`}>{e.loss == null ? DASH : e.loss.toFixed(4)}</td>
                  {valDims.map((d) => (
                    <td key={`r-${d}`} className={`n num ${c}`}>
                      {metric('recall@10', e.val[d]?.['recall@10'])}
                    </td>
                  ))}
                  {valDims.map((d) => (
                    <td key={`n-${d}`} className={`n num ${c}`}>
                      {metric('ndcg@10', e.val[d]?.['ndcg@10'])}
                    </td>
                  ))}
                  <td className={c}>{chosen && <span className="badge ok">✓ chosen</span>}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {evals.length > 0 && (
        <p className="small">
          Scored by:{' '}
          {evals.map((r, i) => (
            <span key={r.run_id}>
              {i > 0 && ' · '}
              <Link to={runPath(r.run_id)}>{r.label}</Link> ({r.eval_split ?? 'no split'})
            </span>
          ))}
        </p>
      )}
      {config.length > 0 && (
        <details className="facts">
          <summary className="label">Training config</summary>
          <dl className="kv">
            {config.map(([k, v]) => (
              <Fact key={k} k={k} v={v} />
            ))}
          </dl>
        </details>
      )}
    </section>
  );
}

function Fact({ k, v }: { k: string; v: unknown }) {
  return (
    <>
      <dt>{k}</dt>
      <dd className="mono">{typeof v === 'object' && v !== null ? JSON.stringify(v) : String(v)}</dd>
    </>
  );
}

function CompareWith({
  run,
  runs,
  dim,
  vs,
  onPick,
}: {
  run: RunSummary;
  runs: RunSummary[] | null;
  dim: string;
  vs: string | null;
  onPick: (vs: string | null) => void;
}) {
  if (!runs) return <p className="small">Loading the runs to compare with…</p>;
  const options = comparable(run, runs);
  const known = !vs || options.some((r) => r.run_id === vs);
  return (
    <div className="row pick">
      <label htmlFor="vs-pick" className="tag">
        compare with…
      </label>
      <select id="vs-pick" value={vs ?? ''} onChange={(e) => onPick(e.target.value || null)}>
        <option value="">— no comparison —</option>
        {!known && vs && <option value={vs}>{vs}</option>}
        {options.map((r) => {
          const ok = r.dims.includes(dim);
          return (
            <option key={r.run_id} value={r.run_id} disabled={!ok}>
              {r.label} · {r.kind} · {when(r.created)}
              {ok ? '' : ` (no ${dimLabel(dim)})`}
            </option>
          );
        })}
      </select>
      {options.length === 0 && (
        <span className="small">No other run is scored on {run.eval_split ?? 'this'} split yet.</span>
      )}
      {vs && (
        <Link className="small" to={runPath(vs, { vs: run.run_id, dim })}>
          swap sides ⇄
        </Link>
      )}
    </div>
  );
}

function ComparePanel({ runId, vs, dim }: { runId: string; vs: string; dim: string }) {
  const { data: c, error } = useGet<Compare>(api.compare(runId, vs, dim));
  if (!c) return <Loading error={error} what="the compare" />;
  return (
    <>
      <p className="sub">
        <b>{c.a.label}</b> against <Link to={runPath(c.b.run_id, { dim })}>{c.b.label}</Link> · {dimLabel(c.dim)} ·{' '}
        {int(c.n)} questions. "Fixed" means this run puts a golden passage in the top 10 and the other does not.
      </p>
      <CompareBody c={c} />
    </>
  );
}

function QueryTable({ runId, dims, dim, vs }: { runId: string; dims: string[]; dim: string; vs: string | null }) {
  const [lens, setLens] = useLens();
  const allowed = vs ? FILTERS : FILTERS.slice(0, 3);
  const askedFilter = lens.get('filter') as QueryFilter | null;
  const filter: QueryFilter = askedFilter && allowed.includes(askedFilter) ? askedFilter : 'all';
  const q = lens.get('q') ?? '';
  const offset = intLens(lens, 'offset');
  const { data, error, loading } = useGet<RunQueries>(
    api.runQueries(runId, { dim, filter, q, offset, limit: LIMIT, vs }),
  );
  const carry = { ...Object.fromEntries(lens.entries()), dim };
  return (
    <>
      <div className="toolbar">
        {dims.length > 1 ? (
          <Choice
            label="dims"
            options={dims.map((d) => ({ value: d, label: d }))}
            value={dim}
            onChange={(d) => setLens({ dim: d, offset: null })}
          />
        ) : (
          <span className="tag">dims · {dimLabel(dim)}</span>
        )}
        <Choice
          label="show"
          options={allowed.map((f) => ({ value: f, label: f }))}
          value={filter}
          onChange={(f) => setLens({ filter: f === 'all' ? null : f, offset: null })}
        />
        <SearchBox
          label="Search questions"
          placeholder="search the questions…"
          value={q}
          onSearch={(v) => setLens({ q: v || null, offset: null })}
        />
      </div>
      {!data ? (
        <Loading error={error} what="the questions" />
      ) : data.items.length === 0 ? (
        <Empty>No questions match{q ? ` "${q}"` : ''} with this filter.</Empty>
      ) : (
        <>
          <div className={`tablewrap ${loading ? 'stale' : ''}`}>
            <table className="qtable">
              <thead>
                <tr>
                  <th>question</th>
                  <th className="num">golden rank</th>
                  <th>top 10</th>
                  {vs && <th className="num">vs rank</th>}
                  {vs && <th>change</th>}
                </tr>
              </thead>
              <tbody>
                {data.items.map((row) => (
                  <tr key={row.qid}>
                    <td className="qtext">
                      <Link to={runQuestionPath(runId, row.qid, carry)}>{row.text}</Link>
                      <span className="tag qid">{row.qid}</span>
                    </td>
                    <td className="n num">{int(row.golden_rank)}</td>
                    <td>
                      <HitBadge rank={row.golden_rank} />
                    </td>
                    {vs && <td className="n num">{row.vs_golden_rank == null ? DASH : int(row.vs_golden_rank)}</td>}
                    {vs && (
                      <td>
                        <ChangeBadge change={row.change} />
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Pager total={data.total} offset={offset} limit={LIMIT} onPage={(o) => setLens({ offset: o || null })} />
        </>
      )}
      {error && data && <ErrorNote error={error} />}
    </>
  );
}
