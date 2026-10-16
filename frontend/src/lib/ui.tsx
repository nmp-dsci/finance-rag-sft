/**
 * The pieces every page shares. Colour comes only from house.css tokens; every status also
 * carries a word (and a glyph), because hue 15 sits close to the fixed status red.
 */
import { type FormEvent, type ReactNode, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import type { Change, Hist, Metrics, MetricKey, RunKind, RunSummary } from './api';
import {
  CHANGE_TEXT,
  DASH,
  type RankRow,
  dimLabel,
  gapText,
  int,
  metric,
  scale,
  sortDims,
  when,
} from './format';
import { runPath } from './url';

// ── states ────────────────────────────────────────────────────────────────
export function ErrorNote({ error, what }: { error: string; what?: string }) {
  return (
    <div className="callout red" role="alert">
      <span className="badge bad">✕ error</span> {what ? `Could not load ${what}: ` : ''}
      <span className="mono">{error}</span>
    </div>
  );
}

export function Loading({ error, what }: { error: string | null; what?: string }) {
  if (error) return <ErrorNote error={error} what={what} />;
  return (
    <p className="busy" role="status" aria-live="polite">
      <span className="busy-dot" aria-hidden="true" /> Loading{what ? ` ${what}` : ''}…
    </p>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

// ── badges ────────────────────────────────────────────────────────────────
const KIND_CLASS: Record<RunKind, string> = {
  baseline: 'info',
  eval: 'ok',
  train: 'warn',
  check: '',
  smoke: '',
};
export function KindBadge({ kind }: { kind: RunKind }) {
  return <span className={`badge ${KIND_CLASS[kind]}`}>{kind}</span>;
}

export function SplitBadge({ split }: { split: string | null | undefined }) {
  if (!split) return <span className="badge">no split</span>;
  return <span className={`badge split-${split}`}>{split}</span>;
}

export function ChangeBadge({ change }: { change: Change | undefined }) {
  if (!change) return null;
  const cls = change === 'fixed' ? 'ok' : change === 'broken' ? 'bad' : '';
  return <span className={`badge ${cls}`}>{CHANGE_TEXT[change]}</span>;
}

/** Hit = the golden passage is in the top 10. */
export function HitBadge({ rank, k = 10 }: { rank: number; k?: number }) {
  return rank <= k ? <span className="badge ok">● hit</span> : <span className="badge">○ miss</span>;
}

export function MlflowLink({ url }: { url: string | null }) {
  if (!url) return <span className="faint">{DASH}</span>;
  return (
    <a href={url} target="_blank" rel="noreferrer">
      MLflow ↗
    </a>
  );
}

// ── controls ──────────────────────────────────────────────────────────────
/** A one-of-n choice as toggle chips: the pressed one carries a tick, not only a colour. */
export function Choice<T extends string>({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: { value: T; label: string; disabled?: boolean }[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div className="choice" role="group" aria-label={label}>
      <span className="tag">{label}</span>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          className="chipbtn"
          aria-pressed={o.value === value}
          disabled={o.disabled}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** A search box that writes to the lens after a pause, or at once on Enter. */
export function SearchBox({
  value,
  onSearch,
  placeholder,
  label,
}: {
  value: string;
  onSearch: (v: string) => void;
  placeholder: string;
  label: string;
}) {
  const [text, setText] = useState(value);
  const pushed = useRef(value);
  const cb = useRef(onSearch);
  cb.current = onSearch;
  useEffect(() => {
    setText(value);
    pushed.current = value;
  }, [value]);
  useEffect(() => {
    if (text === pushed.current) return;
    const t = setTimeout(() => {
      pushed.current = text;
      cb.current(text);
    }, 300);
    return () => clearTimeout(t);
  }, [text]);
  const submit = (e: FormEvent) => {
    e.preventDefault();
    pushed.current = text;
    cb.current(text);
  };
  return (
    <form className="searchbox" role="search" onSubmit={submit}>
      <label className="sr-only" htmlFor={`sb-${label}`}>
        {label}
      </label>
      <input
        id={`sb-${label}`}
        type="search"
        value={text}
        placeholder={placeholder}
        onChange={(e) => setText(e.target.value)}
      />
    </form>
  );
}

export function Pager({
  total,
  offset,
  limit,
  onPage,
}: {
  total: number;
  offset: number;
  limit: number;
  onPage: (offset: number) => void;
}) {
  if (total <= limit && offset === 0) return <p className="small">{int(total)} shown</p>;
  const end = Math.min(total, offset + limit);
  return (
    <nav className="pager" aria-label="pages">
      <button type="button" className="ghost" disabled={offset === 0} onClick={() => onPage(Math.max(0, offset - limit))}>
        ← previous
      </button>
      <span className="small">
        {total === 0 ? 0 : int(offset + 1)}–{int(end)} of {int(total)}
      </span>
      <button type="button" className="ghost" disabled={end >= total} onClick={() => onPage(offset + limit)}>
        next →
      </button>
    </nav>
  );
}

// ── text ──────────────────────────────────────────────────────────────────
/** A passage, clamped to a few lines with a toggle when it is long. */
export function Clamp({ text, open: startOpen = false }: { text: string; open?: boolean }) {
  const long = text.length > 260;
  const [open, setOpen] = useState(startOpen);
  return (
    <span className="clampwrap">
      <span className={long && !open ? 'clamp' : ''}>{text}</span>
      {long && (
        <button type="button" className="linkbtn" aria-expanded={open} onClick={() => setOpen((v) => !v)}>
          {open ? 'less' : 'more'}
        </button>
      )}
    </span>
  );
}

// ── charts ────────────────────────────────────────────────────────────────
/** A length histogram as inline SVG bars; the labels stay HTML so they never shrink. */
export function Histogram({
  title,
  hist,
  median,
  p95,
  max,
  unit,
}: {
  title: string;
  hist: Hist;
  median: number;
  p95: number;
  max: number;
  unit: string;
}) {
  const W = 300;
  const H = 90;
  const heights = scale(hist.map((b) => b[2]));
  const lo = hist.length ? hist[0][0] : 0;
  const hi = hist.length ? hist[hist.length - 1][1] : 0;
  const x = (v: number) => (hi > lo ? (Math.min(Math.max(v, lo), hi) - lo) / (hi - lo) : 0) * W;
  return (
    <figure className="fig hist">
      <div className="fig-title">{title}</div>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label={`${title}: median ${median}, p95 ${p95}, max ${max} ${unit}`}>
        {hist.map(([a, b, n], i) => {
          const h = Math.max(heights[i] * (H - 4), n > 0 ? 1 : 0);
          return (
            <rect key={`${a}-${b}`} className="hbar" x={x(a) + 0.5} y={H - h} width={Math.max(x(b) - x(a) - 1, 0.5)} height={h}>
              <title>
                {a}–{b} {unit}: {int(n)}
              </title>
            </rect>
          );
        })}
        <line className="hmark" x1={x(median)} x2={x(median)} y1={0} y2={H} vectorEffect="non-scaling-stroke" />
        <line className="hmark p95" x1={x(p95)} x2={x(p95)} y1={0} y2={H} vectorEffect="non-scaling-stroke" />
      </svg>
      <div className="haxis tag">
        <span>{int(lo)}</span>
        <span>{int(hi)} {unit}</span>
      </div>
      <figcaption>
        median <b>{int(median)}</b> (solid line) · p95 <b>{int(p95)}</b> (dashed) · max <b>{int(max)}</b> {unit}
      </figcaption>
    </figure>
  );
}

/**
 * A line over steps, as inline SVG; every label is HTML positioned in percent, so text never
 * stretches with the plot. `marks` are vertical lines (epoch ends); the `on` one is emphasised.
 */
export function LineChart({
  title,
  points,
  xMax,
  yMax,
  yTicks,
  marks = [],
  caption,
  alt = false,
}: {
  title: string;
  points: [number, number][];
  xMax: number;
  yMax: number;
  yTicks: [number, string][];
  marks?: { x: number; label: string; on?: boolean }[];
  caption: ReactNode;
  alt?: boolean;
}) {
  const W = 600;
  const H = 160;
  const x = (v: number) => (xMax > 0 ? (v / xMax) * W : 0);
  const y = (v: number) => H - (yMax > 0 ? Math.min(v / yMax, 1) : 0) * H;
  const d = points.map(([a, b], i) => `${i ? 'L' : 'M'}${x(a).toFixed(1)},${y(b).toFixed(1)}`).join(' ');
  const left = (v: number) => `${((v / Math.max(xMax, 1)) * 100).toFixed(2)}%`;
  return (
    <figure className="fig linechart">
      <div className="fig-title">{title}</div>
      <div className="plot">
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label={title}>
          {marks.map((m) => (
            <line key={m.x} className={`vmark ${m.on ? 'on' : ''}`} x1={x(m.x)} x2={x(m.x)} y1={0} y2={H} vectorEffect="non-scaling-stroke" />
          ))}
          <path className={`line ${alt ? 'alt' : ''}`} d={d} vectorEffect="non-scaling-stroke" />
        </svg>
        {yTicks.map(([v, label]) => (
          <span key={label} className="ytick tag" style={{ top: `${((y(v) / H) * 100).toFixed(2)}%` }}>
            {label}
          </span>
        ))}
        {marks.filter((m) => m.label).map((m) => (
          <span
            key={m.x}
            className={`mlabel tag ${m.on ? 'on' : ''} ${m.x > xMax * 0.92 ? 'end' : ''}`}
            style={{ left: left(m.x) }}
          >
            {m.label}
          </span>
        ))}
      </div>
      <div className="xaxis tag">
        <span style={{ left: '0%' }}>0</span>
        <span style={{ left: '100%' }}>{int(xMax)} steps</span>
      </div>
      <figcaption>{caption}</figcaption>
    </figure>
  );
}

/** Where the golden passage moved, this run against the other: up is better. */
export function RankShift({ rows }: { rows: { bucket: string; count: number }[] }) {
  const widths = scale(rows.map((r) => r.count));
  return (
    <figure className="fig">
      <div className="fig-title">Golden rank, this run against the other</div>
      <ul className="shift">
        {rows.map((r, i) => {
          const dir = r.bucket.startsWith('up') ? 'up' : r.bucket.startsWith('down') ? 'down' : 'same';
          return (
            <li key={r.bucket} className={dir}>
              <span className="tag">
                {dir === 'up' ? '▲' : dir === 'down' ? '▼' : '='} {r.bucket}
              </span>
              <span className="track">
                <span className="fill" style={{ width: `${(widths[i] * 100).toFixed(1)}%` }} />
              </span>
              <span className="mono n">{int(r.count)}</span>
            </li>
          );
        })}
      </ul>
      <figcaption>Questions by how far the golden passage moved. Up means this run ranks it higher.</figcaption>
    </figure>
  );
}

// ── ranked lists ──────────────────────────────────────────────────────────
/** A top 10 in the mock's shape: golden rows carry the accent border and a "golden" badge. */
export function RankList({ rows, golden }: { rows: RankRow[]; golden?: Set<string> }) {
  return (
    <ol className="rk">
      {rows.map((row) =>
        row.kind === 'gap' ? (
          <li key={`gap-${row.from}`} className="gap">
            <span>
              {gapText(row.from, row.to)} · golden passage not in the top 10
            </span>
          </li>
        ) : (
          <RankItem key={`${row.hit.rank}-${row.hit.pid}`} hit={row.hit} golden={row.hit.golden || !!golden?.has(row.hit.pid)} />
        ),
      )}
    </ol>
  );
}

function RankItem({ hit, golden }: { hit: { rank: number; pid: string; text: string; score: number }; golden: boolean }) {
  return (
    <li className={golden ? 'gold' : ''}>
      <span className="r">{hit.rank}</span>
      <span className="body">
        {golden && <span className="badge ok">★ golden</span>} <Clamp text={hit.text} open={golden} />
        <span className="pid tag">{hit.pid}</span>
      </span>
      <span className="sc">{hit.score.toFixed(3)}</span>
    </li>
  );
}

// ── tables ────────────────────────────────────────────────────────────────
export const METRIC_COLS: [MetricKey, string][] = [
  ['recall@1', 'R@1'],
  ['recall@5', 'R@5'],
  ['recall@10', 'R@10'],
  ['ndcg@10', 'nDCG@10'],
  ['mrr@10', 'MRR@10'],
];

/** Every metric for every dimension size: one row per dim. */
export function MetricsTable({ metrics, caption }: { metrics: Record<string, Metrics>; caption?: string }) {
  const dims = sortDims(Object.keys(metrics));
  if (!dims.length) return <Empty>No metrics on this run.</Empty>;
  return (
    <div className="tablewrap">
      <table>
        {caption && <caption className="sr-only">{caption}</caption>}
        <thead>
          <tr>
            <th>dims</th>
            {METRIC_COLS.map(([k, l]) => (
              <th key={k} className="num" title={k}>
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {dims.map((d) => (
            <tr key={d}>
              <td className="k">{dimLabel(d)}</td>
              {METRIC_COLS.map(([k]) => (
                <td key={k} className={`n num ${k === 'recall@10' ? 'rec' : ''}`}>
                  {metric(k, metrics[d]?.[k])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Runs, one row per (run, dim): the run's cells span its dims. */
export function RunsTable({ runs }: { runs: RunSummary[] }) {
  return (
    <div className="tablewrap">
      <table className="runs">
        <thead>
          <tr>
            <th>run</th>
            <th>kind</th>
            <th>label</th>
            <th>eval</th>
            <th>dims</th>
            <th className="num">R@10</th>
            <th className="num">nDCG@10</th>
            <th className="num">MRR@10</th>
            <th className="num">questions</th>
            <th>MLflow</th>
          </tr>
        </thead>
        <tbody>
          {runs.flatMap((r) => {
            const dims = sortDims(r.dims.length ? r.dims : Object.keys(r.metrics));
            const rows = dims.length ? dims : [null];
            const span = rows.length;
            return rows.map((d, i) => (
              <tr key={`${r.run_id}-${d ?? 'none'}`} className={i === 0 ? 'first' : 'cont'}>
                {i === 0 && (
                  <>
                    <td rowSpan={span} className="runcell">
                      <Link to={runPath(r.run_id)} className="nw">
                        {when(r.created)}
                      </Link>
                      <span className="tag runid">{r.run_id}</span>
                    </td>
                    <td rowSpan={span}>
                      <KindBadge kind={r.kind} />
                    </td>
                    <td rowSpan={span}>
                      {r.label}
                      {r.note && <div className="small">{r.note}</div>}
                    </td>
                    <td rowSpan={span}>
                      {r.eval_split ? <SplitBadge split={r.eval_split} /> : <span className="faint">{DASH}</span>}
                    </td>
                  </>
                )}
                <td className="k">{d ? dimLabel(d) : DASH}</td>
                <td className="n num rec">{d ? metric('recall@10', r.metrics[d]?.['recall@10']) : DASH}</td>
                <td className="n num">{d ? metric('ndcg@10', r.metrics[d]?.['ndcg@10']) : DASH}</td>
                <td className="n num">{d ? metric('mrr@10', r.metrics[d]?.['mrr@10']) : DASH}</td>
                {i === 0 && (
                  <>
                    <td rowSpan={span} className="n num">
                      {r.n_queries ? int(r.n_queries) : DASH}
                    </td>
                    <td rowSpan={span}>
                      <MlflowLink url={r.mlflow_url} />
                    </td>
                  </>
                )}
              </tr>
            ));
          })}
        </tbody>
      </table>
    </div>
  );
}
