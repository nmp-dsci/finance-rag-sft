/** The compare lens's numbers, shared by the Overview (tuned vs base) and a Run page under `?vs=`. */
import type { Compare, DeltaKey } from './api';
import { delta, int, interval, pValue } from './format';
import { RankShift } from './ui';

const DELTA_ROWS: [DeltaKey, string][] = [
  ['recall@10', 'Recall@10'],
  ['ndcg@10', 'nDCG@10'],
  ['mrr@10', 'MRR@10'],
];

export function CompareBody({ c }: { c: Compare }) {
  return (
    <>
      <div className="kpis compare-stats">
        <div className="stat accent">
          <div className="n">{int(c.fixed)}</div>
          <div className="l">
            <span className="badge ok">▲ fixed</span> this run hits, the other misses
          </div>
        </div>
        <div className="stat red">
          <div className="n">{int(c.broken)}</div>
          <div className="l">
            <span className="badge bad">▼ broken</span> this run misses, the other hits
          </div>
        </div>
        <div className="stat">
          <div className="n">{int(c.both_hit)}</div>
          <div className="l">● both hit</div>
        </div>
        <div className="stat">
          <div className="n">{int(c.both_miss)}</div>
          <div className="l">○ both miss</div>
        </div>
        <div className="stat">
          <div className="n">{pValue(c.p_exact)}</div>
          <div className="l">p, exact McNemar on {int(c.fixed + c.broken)} discordant questions</div>
        </div>
      </div>
      <div className="split compare-detail">
        <div className="tablewrap">
          <table>
            <caption className="sr-only">Paired differences with 95% bootstrap intervals</caption>
            <thead>
              <tr>
                <th>metric</th>
                <th className="num">Δ this − other</th>
                <th className="num">95% CI</th>
              </tr>
            </thead>
            <tbody>
              {DELTA_ROWS.map(([k, label]) => {
                const d = c.delta[k];
                return (
                  <tr key={k}>
                    <td className="nw">{label}</td>
                    <td className="n num">{d ? delta(k, d.value) : '—'}</td>
                    <td className="n num">{d ? interval(k, d.ci_low, d.ci_high) : '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <RankShift rows={c.rank_shift} />
      </div>
      <p className="small">
        Δ is paired over the same {int(c.n)} questions; the interval is a paired bootstrap at 95%. An interval that
        crosses zero is not a difference this split can see.
      </p>
    </>
  );
}
