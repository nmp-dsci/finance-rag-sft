import { type FormEvent, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { type RetrieverOption, type SearchResponse, api, post, useGet } from '../lib/api';
import { dimLabel } from '../lib/format';
import { ErrorNote, Loading, RankList } from '../lib/ui';
import { dataQuestionPath, useLens } from '../lib/url';

/** Real questions from the dataset, so the golden passage lights up. */
export const EXAMPLES = [
  "How much did AT&T's common stock dividends per share total in 2023?",
  'What were the total capital expenditures for Alphabet Inc. in 2023?',
  "What factors primarily drove the decrease in Veklury's sales in 2023?",
  'What is the proportion of Nike Brand footwear manufactured in Vietnam, Indonesia, and China for fiscal 2023?',
];

const DIMS = ['768', '64'];

/** Retrievers in the lens (`?r=bm25,bge-base`), else every one with cached embeddings, else all. */
export function pickRetrievers(lensValue: string | null, options: RetrieverOption[]): string[] {
  const known = new Set(options.map((o) => o.id));
  if (lensValue != null) return lensValue.split(',').filter((id) => known.has(id));
  const ready = options.filter((o) => o.ready).map((o) => o.id);
  return ready.length ? ready : options.map((o) => o.id);
}

/**
 * `/search` — live retrieval. The lens is the state: `?q=` the query, `?r=` the retrievers,
 * `?dim=` the size. A query in the address runs on load, so a search can be linked.
 */
export function Search() {
  const [lens, setLens] = useLens();
  const { data: options, error: optError } = useGet<RetrieverOption[]>(api.retrievers());
  const q = lens.get('q') ?? '';
  const dim = DIMS.includes(lens.get('dim') ?? '') ? lens.get('dim')! : '768';
  const chosen = options ? pickRetrievers(lens.get('r'), options) : [];
  const chosenKey = chosen.join(',');
  const [text, setText] = useState(q);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => setText(q), [q]);

  useEffect(() => {
    if (!q.trim() || !options || !chosenKey) {
      setResult(null);
      return;
    }
    let alive = true;
    setBusy(true);
    setError(null);
    post<SearchResponse>(api.search(), { query: q, retrievers: chosenKey.split(','), dim, k: 10 })
      .then((r) => {
        if (alive) setResult(r);
      })
      .catch((e: Error) => {
        if (!alive) return;
        setError(e.message);
        setResult(null);
      })
      .finally(() => {
        if (alive) setBusy(false);
      });
    return () => {
      alive = false;
    };
  }, [q, chosenKey, dim, options]);

  const run = (query: string) => setLens({ q: query.trim() || null });
  const submit = (e: FormEvent) => {
    e.preventDefault();
    run(text);
  };
  const toggle = (id: string) => {
    const next = chosen.includes(id) ? chosen.filter((x) => x !== id) : [...chosen, id];
    // keep the options' order, so the columns do not shuffle
    const ordered = (options ?? []).map((o) => o.id).filter((x) => next.includes(x));
    setLens({ r: ordered.join(',') });
  };
  const golden = new Set(result?.golden ?? []);

  return (
    <>
      <p className="eyebrow">Search · live retrieval over the corpus</p>
      <h1>
        Ask the corpus <em>live</em>
      </h1>
      <p className="lede">
        Type any question; each retriever ranks every passage and returns its top 10. If the text is a dataset
        question, its golden passage is marked.
      </p>

      <form className="card searchform" onSubmit={submit}>
        <label htmlFor="query" className="label">
          Question
        </label>
        <textarea
          id="query"
          rows={2}
          value={text}
          placeholder="e.g. What were the total capital expenditures for Alphabet Inc. in 2023?"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              run(text);
            }
          }}
        />
        <div className="examples">
          <span className="tag">try</span>
          {EXAMPLES.map((ex) => (
            <button key={ex} type="button" className="chipbtn example" onClick={() => run(ex)}>
              {ex}
            </button>
          ))}
        </div>
        {!options ? (
          <Loading error={optError} what="the retrievers" />
        ) : (
          <div className="choice" role="group" aria-label="retrievers">
            <span className="tag">retrievers</span>
            {options.map((o) => (
              <button
                key={o.id}
                type="button"
                className="chipbtn"
                aria-pressed={chosen.includes(o.id)}
                title={o.ready ? `dims: ${o.dims.join(', ')}` : 'corpus embeddings not cached: the first query encodes the corpus'}
                onClick={() => toggle(o.id)}
              >
                {o.label}
                {!o.ready && <span className="notready"> · not cached</span>}
              </button>
            ))}
          </div>
        )}
        <div className="row formfoot">
          <label htmlFor="dim" className="tag">
            dims
          </label>
          <select id="dim" value={dim} onChange={(e) => setLens({ dim: e.target.value === '768' ? null : e.target.value })}>
            {DIMS.map((d) => (
              <option key={d} value={d}>
                {dimLabel(d)}
              </option>
            ))}
          </select>
          <span className="small">Applies to dense retrievers; BM25 ignores it.</span>
          <button type="submit" disabled={!text.trim() || !chosen.length}>
            Search
          </button>
        </div>
      </form>

      {error && <ErrorNote error={error} what="the search" />}
      {busy && !result && <Loading error={null} what="results" />}
      {result && (
        <section className={`results ${busy ? 'stale' : ''}`} aria-label="results">
          {result.matched_qid ? (
            <div className="callout">
              <span className="badge ok">★ dataset question</span> This is question{' '}
              <Link to={dataQuestionPath(result.matched_qid)}>{result.matched_qid}</Link>; its golden passage
              {result.golden.length === 1 ? ' is' : 's are'} marked below.
            </div>
          ) : (
            <p className="small">Not a dataset question, so there is no golden passage to mark.</p>
          )}
          <div className="rank-cols">
            {result.results.map((r) => {
              const otherDim = r.dim !== 'lexical' && r.dim !== dim;
              return (
                <div key={r.retriever}>
                  <h4>
                    {r.label} <span className="badge">{dimLabel(r.dim)}</span>
                    {otherDim && <span className="badge warn">asked {dim}, got {r.dim}</span>}
                    <span className="tag">{r.ms.toFixed(0)} ms</span>
                  </h4>
                  <RankList rows={r.top.map((hit) => ({ kind: 'hit', hit }))} golden={golden} />
                </div>
              );
            })}
          </div>
        </section>
      )}
    </>
  );
}
