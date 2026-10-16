import { Link } from 'react-router-dom';
import { type Compare, type Overview as OverviewData, type RunSummary, api, useGet } from '../lib/api';
import { dimLabel, int, pText, pct } from '../lib/format';
import { CompareBody } from '../lib/compare';
import { Empty, KindBadge, Loading, RunsTable } from '../lib/ui';
import { dataPath, runPath } from '../lib/url';

const HEADLINE_DIMS = ['768', '64'];

/** `/` — the answer first: Recall@10, base against tuned, at full and truncated size. */
export function Overview() {
  const { data, error } = useGet<OverviewData>(api.overview());
  if (!data) return <Loading error={error} what="the overview" />;
  const { base, tuned, compare, split } = data;
  return (
    <>
      <p className="eyebrow">
        Overview · split {split?.version ?? 'v1'} · test questions
      </p>
      <h1>
        Does fine-tuning put the golden passage in the top 10 <em>more often</em>?
      </h1>
      <p className="lede">
        <code>BAAI/bge-base-en-v1.5</code>, untouched and fine-tuned on finance (question, passage) pairs, scored with
        the same harness on the same frozen test split. Recall@10 is the share of questions whose golden passage lands in
        the top 10 of {split ? int(split.passages) : '6,757'} passages.
      </p>

      <div className="verdict headline" role="group" aria-label="Recall@10, base and tuned">
        {HEADLINE_DIMS.map((d) => (
          <Headline key={`base-${d}`} run={base} dim={d} who="base" />
        ))}
        {HEADLINE_DIMS.map((d) => (
          <Headline key={`tuned-${d}`} run={tuned} dim={d} who="tuned" />
        ))}
      </div>

      <section>
        <span className="eyebrow">Tuned against base</span>
        <h2>Which questions moved</h2>
        <CompareSummary compare={compare} base={base} tuned={tuned} />
      </section>

      <section>
        <span className="eyebrow">Baselines · test split</span>
        <h2>Every baseline, newest first</h2>
        <p className="sub">BM25, bge-small and bge-base before any training: the rows a tuned model has to beat.</p>
        {data.baselines.length ? (
          <RunsTable runs={data.baselines} />
        ) : (
          <Empty>No baseline runs yet — M2 scores BM25, bge-small and bge-base on the test split.</Empty>
        )}
      </section>

      <div className="split twocol">
        <section>
          <span className="eyebrow">Milestones</span>
          <h2>Where the project stands</h2>
          {data.milestones.length ? (
            <ol className="milestones">
              {data.milestones.map((m) => (
                <li key={m.id} className={m.status}>
                  <span className={`badge ${m.status === 'done' ? 'ok' : ''}`}>
                    {m.status === 'done' ? '✓ done' : '○ pending'}
                  </span>
                  <div>
                    <b>
                      {m.id} · {m.title}
                    </b>
                    {m.detail && <div className="small">{m.detail}</div>}
                  </div>
                </li>
              ))}
            </ol>
          ) : (
            <Empty>No milestones reported.</Empty>
          )}
        </section>
        <section>
          <span className="eyebrow">The split</span>
          <h2>Frozen split {split?.version ?? 'v1'}</h2>
          {split ? (
            <>
              <dl className="kv">
                <dt>passages</dt>
                <dd>{int(split.passages)}</dd>
                <dt>train</dt>
                <dd>{int(split.queries.train)} questions</dd>
                <dt>val</dt>
                <dd>{int(split.queries.val)} questions</dd>
                <dt>test</dt>
                <dd>{int(split.queries.test)} questions</dd>
              </dl>
              <p className="small">
                Grouped by passage, so no golden passage crosses splits. <Link to={dataPath()}>Explore the data →</Link>
              </p>
            </>
          ) : (
            <Empty>No split yet — M1 freezes split v1.</Empty>
          )}
        </section>
      </div>
    </>
  );
}

function Headline({ run, dim, who }: { run: RunSummary | null; dim: string; who: 'base' | 'tuned' }) {
  const v = run?.metrics[dim]?.['recall@10'];
  if (!run)
    return (
      <div className="stat empty-stat">
        <div className="n faint">—</div>
        <div className="l">
          {who === 'tuned' ? 'No tuned model yet — M3' : 'No base run yet — M2'} · {dimLabel(dim)}
        </div>
      </div>
    );
  return (
    <div className={`stat ${who === 'tuned' ? 'accent' : ''}`}>
      <div className="n">{v == null ? '—' : pct(v)}</div>
      <div className="l">
        Recall@10, {who} · {dimLabel(dim)}
      </div>
      <div className="tag">
        <Link to={runPath(run.run_id, { dim })}>{run.label}</Link>
        {v == null && ' · not scored at this size'}
      </div>
    </div>
  );
}

function CompareSummary({
  compare,
  base,
  tuned,
}: {
  compare: Compare | null;
  base: RunSummary | null;
  tuned: RunSummary | null;
}) {
  if (!compare) {
    return (
      <Empty>
        {!tuned
          ? 'No tuned model yet — M3. The compare appears once a tuned run is scored on test.'
          : !base
            ? 'No base run to compare against yet.'
            : 'No compare available for these two runs.'}
      </Empty>
    );
  }
  return (
    <>
      <p className="sub">
        <KindBadge kind={compare.a.kind} /> {compare.a.label} against <KindBadge kind={compare.b.kind} />{' '}
        {compare.b.label}, {dimLabel(compare.dim)}, {int(compare.n)} questions · {pText(compare.p_exact)} (exact
        McNemar on fixed against broken).
      </p>
      <CompareBody c={compare} />
      <p>
        <Link to={runPath(compare.a.run_id, { dim: compare.dim, vs: compare.b.run_id })}>
          Open every question, tuned against base →
        </Link>
      </p>
    </>
  );
}
