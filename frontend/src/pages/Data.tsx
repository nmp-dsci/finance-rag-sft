import { Link, Outlet, useMatch, useOutletContext, useParams } from 'react-router-dom';
import {
  type PassageDetail,
  type PassageList,
  type QueryDetail,
  type QueryList,
  type SplitName,
  type SplitStats,
  api,
  useGet,
} from '../lib/api';
import { dimLabel, int } from '../lib/format';
import { Choice, Clamp, Empty, ErrorNote, HitBadge, Histogram, Loading, Pager, SearchBox, SplitBadge } from '../lib/ui';
import {
  type Lens,
  dataPassagePath,
  dataPath,
  dataQuestionPath,
  intLens,
  runQuestionPath,
  useLens,
} from '../lib/url';

const LIMIT = 50;
const SPLITS: SplitName[] = ['train', 'val', 'test', 'all'];
type View = 'questions' | 'passages';
type DetailCtx = { inline: boolean; lens: Lens };

/**
 * `/data` — browse train, val and test. Picking a question (`/data/q/:qid`) or a passage
 * (`/data/p/:pid`) opens it inside the list, under the row you picked; the lens (view, split,
 * q, offset) stays in the query string, so the list stays put.
 */
export function Data() {
  const [lens, setLens] = useLens();
  const qMatch = useMatch('/data/q/:qid');
  const pMatch = useMatch('/data/p/:pid');
  const askedView = lens.get('view');
  const view: View =
    askedView === 'questions' || askedView === 'passages' ? askedView : pMatch ? 'passages' : 'questions';
  const askedSplit = lens.get('split') as SplitName | null;
  const split: SplitName = askedSplit && SPLITS.includes(askedSplit) ? askedSplit : 'all';
  const q = lens.get('q') ?? '';
  const offset = intLens(lens, 'offset');
  const params = { split, q, offset, limit: LIMIT };
  const stats = useGet<SplitStats>(api.split());
  const questions = useGet<QueryList>(view === 'questions' ? api.queries(params) : null);
  const passages = useGet<PassageList>(view === 'passages' ? api.passages(params) : null);
  const carry: Lens = Object.fromEntries(lens.entries());
  const openQ = qMatch?.params.qid ?? null;
  const openP = pMatch?.params.pid ?? null;
  const inList =
    (openQ && view === 'questions' && questions.data?.items.some((i) => i.qid === openQ)) ||
    (openP && view === 'passages' && passages.data?.items.some((i) => i.pid === openP));
  const list = view === 'questions' ? questions : passages;

  return (
    <>
      <p className="eyebrow">Data · split {stats.data?.version ?? 'v1'}</p>
      <h1>
        The questions and passages, <em>train, val and test</em>
      </h1>
      <p className="lede">
        Every (question, golden passage) pair, split by passage group so no passage is in two splits. Train pairs teach
        the model; val picks the epoch; test is scored.
      </p>
      {stats.data ? <SplitKpis s={stats.data} /> : stats.error && <ErrorNote error={stats.error} what="the split" />}

      <div className="toolbar">
        <Choice
          label="browse"
          options={[
            { value: 'questions', label: 'questions' },
            { value: 'passages', label: 'passages' },
          ]}
          value={view}
          onChange={(v) => setLens({ view: v === 'questions' ? null : v, offset: null })}
        />
        <Choice
          label="split"
          options={SPLITS.map((s) => ({ value: s, label: s }))}
          value={split}
          onChange={(s) => setLens({ split: s === 'all' ? null : s, offset: null })}
        />
        <SearchBox
          label={`Search ${view}`}
          placeholder={`search the ${view}…`}
          value={q}
          onSearch={(v) => setLens({ q: v || null, offset: null })}
        />
      </div>

      {(openQ || openP) && !inList && (
        <div className="detail-pane">
          <Outlet context={{ inline: false, lens: carry } satisfies DetailCtx} />
        </div>
      )}

      {!list.data ? (
        <Loading error={list.error} what={`the ${view}`} />
      ) : list.data.items.length === 0 ? (
        <Empty>
          No {view} {split === 'all' ? '' : `in ${split} `}
          {q ? `match "${q}"` : 'yet'}.
        </Empty>
      ) : (
        <>
          <p className="small">
            {int(list.data.total)} {view}
            {split === 'all' ? '' : ` in ${split}`}
            {q ? ` matching "${q}"` : ''}
          </p>
          <ul className={`dlist ${list.loading ? 'stale' : ''}`}>
            {view === 'questions'
              ? questions.data?.items.map((it) => {
                  const open = it.qid === openQ;
                  return (
                    <li key={it.qid} className={open ? 'open' : ''}>
                      <Link
                        className="item"
                        to={open ? dataPath(carry) : dataQuestionPath(it.qid, carry)}
                        aria-expanded={open}
                      >
                        <span className="text">{it.text}</span>
                        <span className="meta">
                          <SplitBadge split={it.split} />
                          <span className="tag">
                            {it.golden.length} golden{it.golden.length === 1 ? '' : ' passages'}
                          </span>
                          <span className="tag">{it.qid}</span>
                        </span>
                      </Link>
                      {open && <Outlet context={{ inline: true, lens: carry } satisfies DetailCtx} />}
                    </li>
                  );
                })
              : passages.data?.items.map((it) => {
                  const open = it.pid === openP;
                  return (
                    <li key={it.pid} className={open ? 'open' : ''}>
                      <Link
                        className="item"
                        to={open ? dataPath(carry) : dataPassagePath(it.pid, carry)}
                        aria-expanded={open}
                      >
                        <span className={`text ${open ? '' : 'clamp'}`}>{it.text}</span>
                        <span className="meta">
                          <SplitBadge split={it.split} />
                          <span className="tag">{int(it.words)} words</span>
                          <span className="tag">
                            {it.n_questions} question{it.n_questions === 1 ? '' : 's'}
                          </span>
                          <span className="tag">{it.pid}</span>
                        </span>
                      </Link>
                      {open && <Outlet context={{ inline: true, lens: carry } satisfies DetailCtx} />}
                    </li>
                  );
                })}
          </ul>
          <Pager
            total={list.data.total}
            offset={offset}
            limit={LIMIT}
            onPage={(o) => setLens({ offset: o || null })}
          />
        </>
      )}

      {stats.data && <SplitDetail s={stats.data} />}
    </>
  );
}

function useDetailCtx(): DetailCtx {
  return useOutletContext<DetailCtx | undefined>() ?? { inline: false, lens: {} };
}

/** `/data/q/:qid` — a question, its golden passage(s), who else points at them, and how runs ranked it. */
export function DataQuestion() {
  const { qid = '' } = useParams();
  const { inline, lens } = useDetailCtx();
  const { data, error } = useGet<QueryDetail>(api.query(qid));
  if (!data || data.qid !== qid) return <Loading error={error} what={`question ${qid}`} />;
  return (
    <article className="detail" aria-label={`question ${qid}`}>
      {!inline && <h2 className="qt">{data.text}</h2>}
      <div className="row badges">
        <SplitBadge split={data.split} />
        <span className="tag">{data.qid}</span>
        <Link className="small" to={dataPath(lens)}>
          close ✕
        </Link>
      </div>
      <h4>Golden passage{data.golden.length === 1 ? '' : `s (${data.golden.length})`}</h4>
      <ul className="golden">
        {data.golden.map((g) => (
          <GoldenPassage key={g.pid} g={g} qid={data.qid} lens={lens} />
        ))}
      </ul>
      <h4>Runs that scored it</h4>
      {data.runs.length ? (
        <div className="tablewrap">
          <table>
            <thead>
              <tr>
                <th>run</th>
                <th>dims</th>
                <th className="num">golden rank</th>
                <th>top 10</th>
              </tr>
            </thead>
            <tbody>
              {data.runs.map((r) => (
                <tr key={`${r.run_id}-${r.dim}`}>
                  <td>
                    <Link to={runQuestionPath(r.run_id, data.qid, { dim: r.dim })}>{r.label}</Link>
                  </td>
                  <td className="k">{dimLabel(r.dim)}</td>
                  <td className="n num">{int(r.golden_rank)}</td>
                  <td>
                    <HitBadge rank={r.golden_rank} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <Empty>
          No run has scored this question yet
          {data.split === 'train' ? ' — train questions are for training, never scored.' : '.'}
        </Empty>
      )}
    </article>
  );
}

function GoldenPassage({
  g,
  qid,
  lens,
}: {
  g: QueryDetail['golden'][number];
  qid: string;
  lens: Lens;
}) {
  const others = g.n_questions - 1;
  const { data } = useGet<PassageDetail>(others > 0 ? api.passage(g.pid) : null);
  const list = data?.questions.filter((x) => x.qid !== qid) ?? [];
  return (
    <li className="card gold">
      <Clamp text={g.text} open />
      <div className="row meta">
        <Link className="tag" to={dataPassagePath(g.pid, { ...lens, view: 'passages' })}>
          {g.pid}
        </Link>
        <span className="tag">
          {others > 0
            ? `${others} other question${others === 1 ? ' points' : 's point'} here`
            : 'no other question points here'}
        </span>
      </div>
      {list.length > 0 && (
        <ul className="others">
          {list.map((o) => (
            <li key={o.qid}>
              <SplitBadge split={o.split} /> <Link to={dataQuestionPath(o.qid, lens)}>{o.text}</Link>
            </li>
          ))}
        </ul>
      )}
    </li>
  );
}

/** `/data/p/:pid` — a passage and every question that points at it. */
export function DataPassage() {
  const { pid = '' } = useParams();
  const { inline, lens } = useDetailCtx();
  const { data, error } = useGet<PassageDetail>(api.passage(pid));
  if (!data || data.pid !== pid) return <Loading error={error} what={`passage ${pid}`} />;
  return (
    <article className="detail" aria-label={`passage ${pid}`}>
      {!inline && (
        <div className="card gold">
          <Clamp text={data.text} open />
        </div>
      )}
      <div className="row badges">
        <SplitBadge split={data.split} />
        <span className="tag">{int(data.words)} words</span>
        <span className="tag">
          in {int(data.rows)} dataset row{data.rows === 1 ? '' : 's'}
        </span>
        <span className="tag">{data.pid}</span>
        <Link className="small" to={dataPath(lens)}>
          close ✕
        </Link>
      </div>
      <h4>
        {data.questions.length} question{data.questions.length === 1 ? '' : 's'} point here
      </h4>
      <ul className="others">
        {data.questions.map((o) => (
          <li key={o.qid}>
            <SplitBadge split={o.split} />{' '}
            <Link to={dataQuestionPath(o.qid, { ...lens, view: null })}>{o.text}</Link>
          </li>
        ))}
      </ul>
    </article>
  );
}

function SplitKpis({ s }: { s: SplitStats }) {
  return (
    <div className="kpis">
      <div className="stat">
        <div className="n">{int(s.passages)}</div>
        <div className="l">passages</div>
      </div>
      <div className="stat">
        <div className="n">{int(s.queries.train)}</div>
        <div className="l">train questions</div>
      </div>
      <div className="stat">
        <div className="n">{int(s.queries.val)}</div>
        <div className="l">val questions</div>
      </div>
      <div className="stat">
        <div className="n">{int(s.queries.test)}</div>
        <div className="l">test questions</div>
      </div>
      <div className={`stat ${s.leak_check.ok ? '' : 'red'}`}>
        <div className="n">
          {s.leak_check.ok ? <span className="badge ok">✓ ok</span> : <span className="badge bad">✕ leak</span>}
        </div>
        <div className="l">leak check: no passage in two splits</div>
      </div>
    </div>
  );
}

function SplitDetail({ s }: { s: SplitStats }) {
  const lc = s.leak_check;
  return (
    <section>
      <span className="eyebrow">Split {s.version}, in detail</span>
      <h2>How the split was dealt</h2>
      <p className="sub">
        {int(s.rows)} dataset rows became {int(s.passages)} passages and {int(s.queries.total)} questions, dealt into
        splits by {int(s.groups)} passage groups (seed {s.seed}).
      </p>
      <div className="cards">
        <div className="card">
          <h3>Source</h3>
          <dl className="kv">
            <dt>dataset</dt>
            <dd className="mono">{s.dataset}</dd>
            <dt>revision</dt>
            <dd className="mono">{s.revision}</dd>
            <dt>fingerprint</dt>
            <dd className="mono">{s.fingerprint}</dd>
            <dt>pairs</dt>
            <dd>
              {int(s.pairs.train)} train · {int(s.pairs.val)} val · {int(s.pairs.test)} test
            </dd>
          </dl>
        </div>
        <div className="card">
          <h3>Duplicates</h3>
          <dl className="kv">
            <dt>passages repeated</dt>
            <dd>
              {int(s.duplicates.passages_repeated)} (up to {int(s.duplicates.max_repeats)}×)
            </dd>
            <dt>rows sharing one</dt>
            <dd>{int(s.duplicates.rows_sharing)}</dd>
            <dt>questions repeated</dt>
            <dd>{int(s.duplicates.questions_repeated)}</dd>
            <dt>multi-golden</dt>
            <dd>{int(s.multi_golden_queries)} questions with more than one golden passage</dd>
          </dl>
        </div>
        <div className="card">
          <h3>
            Leak check{' '}
            {lc.ok ? <span className="badge ok">✓ ok</span> : <span className="badge bad">✕ failed</span>}
          </h3>
          <dl className="kv">
            <dt>train ∩ test</dt>
            <dd>{int(lc.train_test_overlap)}</dd>
            <dt>train ∩ val</dt>
            <dd>{int(lc.train_val_overlap)}</dd>
            <dt>val ∩ test</dt>
            <dd>{int(lc.val_test_overlap)}</dd>
          </dl>
          <p className="small">Passages shared between splits; all three must be 0.</p>
        </div>
        <div className="card">
          <h3>Truncation</h3>
          <p>
            <b>{int(s.truncated_passages)}</b> passage{s.truncated_passages === 1 ? '' : 's'} run past bge's 512-token
            window; the model sees only the start.
          </p>
        </div>
      </div>
      <div className="split hists">
        <Histogram title="Passage length" unit="words" {...s.lengths.passage_words} />
        <Histogram title="Question length" unit="words" {...s.lengths.question_words} />
      </div>
    </section>
  );
}
