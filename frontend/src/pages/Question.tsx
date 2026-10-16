import type { ReactNode } from 'react';
import { Link, useParams } from 'react-router-dom';
import { type RankedList, type Run, type RunQuestion as RunQuestionData, api, useGet } from '../lib/api';
import { CHANGE_TEXT, changeOf, dimLabel, int, sortDims, withGaps } from '../lib/format';
import { ChangeBadge, Choice, Clamp, Loading, RankList, SplitBadge } from '../lib/ui';
import { dataPassagePath, dataQuestionPath, runPath, runsPath, useLens } from '../lib/url';

/**
 * `/runs/:runId/:qid` — one question under one run: its golden passage(s) and the run's top 10.
 * With `?vs=<run>` the other run's top 10 sits beside it (the plan's mock: base left, tuned right).
 */
export function RunQuestion() {
  const { runId = '', qid = '' } = useParams();
  const [lens, setLens] = useLens();
  const vs = lens.get('vs') || null;
  const { data: run, error: runError } = useGet<Run>(api.run(runId));
  const dims = run ? sortDims(run.dims.length ? run.dims : Object.keys(run.metrics)) : [];
  // Links from the Run page carry `dim`; a bare address waits for the run to learn its first dim.
  const dim = lens.get('dim') || dims[0] || null;
  const { data: q, error } = useGet<RunQuestionData>(dim ? api.runQuestion(runId, qid, { dim, vs }) : null);
  const back = Object.fromEntries(lens.entries());

  if (!q || q.qid !== qid) return <Loading error={error ?? runError} what={`question ${qid}`} />;
  const change = q.vs ? changeOf(q.run.golden_rank, q.vs.golden_rank) : undefined;
  const golden = new Set(q.golden.map((g) => g.pid));

  return (
    <>
      <p className="crumb">
        <Link to={runsPath()}>runs</Link> / <Link to={runPath(runId, back)}>{q.run.label}</Link> / {q.qid}
        {q.vs && ` ?vs=${q.vs.run_id}`} · split {q.split} · {dimLabel(q.dim)}
      </p>
      <h1 className="qt">{q.text}</h1>
      <div className="row badges">
        <SplitBadge split={q.split} />
        <Link className="small" to={dataQuestionPath(q.qid)}>
          this question in the data →
        </Link>
        {dims.length > 1 && (
          <Choice
            label="dims"
            options={dims.map((d) => ({ value: d, label: d }))}
            value={q.dim}
            onChange={(d) => setLens({ dim: d })}
          />
        )}
      </div>

      <div className="kpis">
        {q.vs && (
          <div className="stat">
            <div className="n">{int(q.vs.golden_rank)}</div>
            <div className="l">golden rank, {q.vs.label}</div>
          </div>
        )}
        <div className={`stat ${q.run.golden_rank <= 10 ? 'accent' : ''}`}>
          <div className="n">{int(q.run.golden_rank)}</div>
          <div className="l">golden rank, {q.run.label}</div>
        </div>
        {change ? (
          <div className={`stat ${change === 'fixed' ? 'accent' : change === 'broken' ? 'red' : ''}`}>
            <div className="n">{CHANGE_TEXT[change].split(' ').slice(1).join(' ')}</div>
            <div className="l">
              {change === 'fixed'
                ? 'miss → hit at k=10'
                : change === 'broken'
                  ? 'hit → miss at k=10'
                  : change === 'same_hit'
                    ? 'in the top 10 under both'
                    : 'outside the top 10 under both'}
            </div>
          </div>
        ) : (
          <div className="stat">
            <div className="n">{q.run.golden_rank <= 10 ? 'hit' : 'miss'}</div>
            <div className="l">golden passage {q.run.golden_rank <= 10 ? 'in' : 'not in'} the top 10</div>
          </div>
        )}
      </div>

      <h4>Golden passage{q.golden.length === 1 ? '' : `s (${q.golden.length})`}</h4>
      <ul className="golden">
        {q.golden.map((g) => (
          <li key={g.pid} className="card gold">
            <Clamp text={g.text} open />
            <Link className="tag pid" to={dataPassagePath(g.pid)}>
              {g.pid}
            </Link>
          </li>
        ))}
      </ul>

      <div className="rank-cols">
        {q.vs && <Column list={q.vs} golden={golden} tag={<span className="badge">vs</span>} />}
        <Column list={q.run} golden={golden} tag={q.vs ? <ChangeBadge change={change} /> : null} />
      </div>
    </>
  );
}

function Column({ list, golden, tag }: { list: RankedList; golden: Set<string>; tag: ReactNode }) {
  return (
    <div>
      <h4>
        <Link to={runPath(list.run_id)}>{list.label}</Link> {tag}
      </h4>
      <RankList rows={withGaps(list.top, list.golden_hits)} golden={golden} />
    </div>
  );
}
