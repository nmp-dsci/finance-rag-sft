import type { CSSProperties } from 'react';
import { Link } from 'react-router-dom';
import { type MethodBatch, type MethodModel, type MethodSplit, api, useGet } from '../lib/api';
import { dimLabel, int } from '../lib/format';
import { diagStats, rescued } from '../lib/method';
import { Choice, Clamp, Loading } from '../lib/ui';
import { dataQuestionPath, intLens, useLens } from '../lib/url';

const SIZES = ['768', '512', '256', '128', '64'];
const NS = [2, 4, 6, 8];

/**
 * One batch as MultipleNegativesRankingLoss sees it, computed live by `/api/method/batch` from
 * real pairs and the harness's own scores. The lens: `?batch=val&dim=64&n=6&seed=3`.
 */
export function BatchExample({ batchSize }: { batchSize: number }) {
  const [lens, setLens] = useLens();
  const split: MethodSplit = lens.get('batch') === 'val' ? 'val' : 'train';
  const dim = SIZES.includes(lens.get('dim') ?? '') ? lens.get('dim')! : '768';
  const n = NS.includes(intLens(lens, 'n', 4)) ? intLens(lens, 'n', 4) : 4;
  const seed = intLens(lens, 'seed', 0);
  const { data, error, loading } = useGet<MethodBatch>(api.methodBatch({ split, n, seed, dim }));
  return (
    <div className="batch-example">
      <div className="toolbar">
        <Choice<MethodSplit>
          label="pairs from"
          value={split}
          options={[
            { value: 'train', label: 'train' },
            { value: 'val', label: 'val (never trained on)' },
          ]}
          onChange={(v) => setLens({ batch: v === 'train' ? null : v })}
        />
        <Choice
          label="size"
          value={dim}
          options={SIZES.map((d) => ({ value: d, label: d }))}
          onChange={(v) => setLens({ dim: v === '768' ? null : v })}
        />
        <Choice
          label="pairs"
          value={String(n)}
          options={NS.map((v) => ({ value: String(v), label: String(v) }))}
          onChange={(v) => setLens({ n: v === '4' ? null : v })}
        />
        <button type="button" className="ghost" onClick={() => setLens({ seed: seed + 1 })}>
          ↻ another batch
        </button>
        <span className="tag">seed {seed}</span>
      </div>
      {!data ? (
        <Loading error={error} what="the batch (the first load starts both models, a few seconds)" />
      ) : (
        <div className={loading ? 'stale' : ''}>
          <ol className="pairs">
            {data.pairs.map((p, i) => (
              <li key={p.qid}>
                <span className="badge info">Q{i + 1}</span>
                <span>
                  <Link to={dataQuestionPath(p.qid)}>{p.question}</Link>
                </span>
                <span className="badge">P{i + 1}</span>
                <span className="small">
                  <Clamp text={p.passage} />
                </span>
              </li>
            ))}
          </ol>
          <div className="rank-cols">
            {data.models.map((m) => (
              <Matrix key={m.retriever} m={m} n={data.n} />
            ))}
          </div>
          <Notice data={data} batchSize={batchSize} />
        </div>
      )}
    </div>
  );
}

function Matrix({ m, n }: { m: MethodModel; n: number }) {
  const idx = Array.from({ length: n }, (_, i) => i);
  return (
    <div>
      <h4>
        {m.label} <span className="tag">mean loss {m.mean_loss.toFixed(4)}</span>
      </h4>
      <div className="tablewrap">
        <table className="matrix">
          <caption className="sr-only">
            Cosine similarity of each question (rows) with each passage in the batch (columns), {m.label}
          </caption>
          <thead>
            <tr>
              <th />
              {idx.map((j) => (
                <th key={j} className="num">
                  P{j + 1}
                </th>
              ))}
              <th className="num" title="softmax probability on the question's own passage">
                p(own)
              </th>
              <th className="num" title="MNRL for this row: −log p(own)">
                loss
              </th>
              <th className="num" title="the own passage's rank among every passage in the corpus">
                corpus rank
              </th>
            </tr>
          </thead>
          <tbody>
            {m.sims.map((row, i) => {
              const best = row.indexOf(Math.max(...row));
              return (
                <tr key={i}>
                  <th scope="row">Q{i + 1}</th>
                  {row.map((v, j) => (
                    <td
                      key={j}
                      className={`cell ${i === j ? 'diag' : ''} ${j === best ? 'best' : ''}`}
                      style={{ '--w': Math.max(0, Math.min(1, v)).toFixed(3) } as CSSProperties}
                      title={`cos(Q${i + 1}, P${j + 1}) = ${v.toFixed(4)}${i === j ? ' (the right passage)' : ''}`}
                    >
                      <span>{v.toFixed(3)}</span>
                    </td>
                  ))}
                  <td className="n num">{m.p_correct[i].toFixed(4)}</td>
                  <td className="n num">{m.loss[i].toFixed(4)}</td>
                  <td className="n num">
                    {m.corpus_rank[i] <= 10 ? (
                      <span className="badge ok">● {m.corpus_rank[i]}</span>
                    ) : (
                      <span className="badge warn">○ {int(m.corpus_rank[i])}</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="small">
        Matryoshka adds this loss at every size:{' '}
        {m.matryoshka.map((r, i) => (
          <span key={r.dim} className="nw">
            {i > 0 && ' + '}
            <span className="mono">{r.mean_loss.toFixed(4)}</span> ({r.dim})
          </span>
        ))}{' '}
        = <b className="mono">{m.matryoshka_loss.toFixed(4)}</b>, the number training logs.
      </p>
    </div>
  );
}

/** What this batch shows, in words, from its own numbers. */
function Notice({ data, batchSize }: { data: MethodBatch; batchSize: number }) {
  const [base, tuned] = data.models;
  const b = base ? diagStats(base.sims) : null;
  const t = tuned ? diagStats(tuned.sims) : null;
  const saved = base && tuned ? rescued(base, tuned) : [];
  return (
    <div className="callout">
      <ul className="tight">
        <li>
          Each row is a {data.n}-way choice; a real training batch is a {batchSize}-way choice, with{' '}
          {batchSize - 1} wrong passages per question, not {data.n - 1}. The darker the cell, the higher the cosine
          (0 to 1). The outlined diagonal holds the right passage, and an underline marks each row's highest score.
        </li>
        {b && t && (
          <li>
            Base model: right passages average <b className="mono">{b.diag.toFixed(3)}</b>, wrong ones{' '}
            <b className="mono">{b.off.toFixed(3)}</b>. Tuned: <b className="mono">{t.diag.toFixed(3)}</b> and{' '}
            <b className="mono">{t.off.toFixed(3)}</b>. Fine-tuning widened the gap by pushing wrong passages away as
            well as pulling right ones close. Absolute cosines change between models, so only the gap within one model
            matters.
          </li>
        )}
        {base &&
          tuned &&
          saved.map((i) => (
            <li key={i}>
              Q{i + 1}: inside this batch the base model gives its own passage p = {base.p_correct[i].toFixed(3)}, yet
              over the whole corpus of {int(data.corpus)} passages it ranks <b>{int(base.corpus_rank[i])}</b>, outside
              the top 10. The tuned model ranks it <b>{int(tuned.corpus_rank[i])}</b>. Random in-batch negatives are
              mostly easy; the passages that compete are elsewhere in the corpus.
            </li>
          ))}
        <li>
          {data.split === 'train'
            ? 'These pairs come from the train split, so the tuned model has seen them. Expect near-certain rows. Switch to val to see questions it never saw.'
            : 'These pairs come from val. Neither model trained on them; val only chose the epoch.'}{' '}
          Size {dimLabel(data.dim)}: each vector is cut to its first {data.dim} numbers and rescaled to length 1, as
          the loss and the harness do.
        </li>
      </ul>
    </div>
  );
}
