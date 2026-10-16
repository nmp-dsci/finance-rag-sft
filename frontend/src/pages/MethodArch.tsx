import type { CSSProperties, ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { int } from '../lib/format';
import { MODEL, bertParams } from '../lib/method';
import { dataQuestionPath, runPath } from '../lib/url';

/** A real train question and its golden passage; the tokens are the tokenizer's at MODEL.revision. */
const EXAMPLE = {
  qid: 'q12daf0167ef3',
  question: 'How is adjusted debt to EBITDAR calculated?',
  passage:
    'Adjusted debt to EBITDAR is defined as the sum of total debt, finance lease obligations and annual rents times six; divided by …',
  tokens: ['[CLS]', 'how', 'is', 'adjusted', 'debt', 'to', 'e', '##bit', '##dar', 'calculated', '?', '[SEP]'],
};

/**
 * How far training moved each block: ‖W_tuned − W_base‖ ÷ ‖W_base‖, the epoch-2 checkpoint of
 * MOVED.run against the base revision, measured on 2026-10-13. `wordRows`: rows of the word table
 * that changed at all (a word piece that never occurs in training gets no gradient).
 */
const MOVED = {
  run: '20261012-164453-train-bge-base',
  word: 0.00222,
  wordRows: 8685,
  otherEmbeddings: 0.00596,
  layers: [0.00693, 0.00676, 0.00663, 0.00669, 0.00673, 0.00685, 0.00693, 0.0069, 0.00676, 0.0063, 0.00657, 0.00996],
  allLayers: 0.00705,
  all: 0.00607,
};

const pc = (x: number, dp = 1) => `${(x * 100).toFixed(dp)}%`;
const short = (n: number) => `${(n / 1e6).toFixed(1)}M`;
const special = (t: string) => t.startsWith('[');

function Token({ t, className = '' }: { t: string; className?: string }) {
  const kind = special(t) ? 'special' : t.startsWith('##') ? 'piece' : '';
  return <span className={`tok ${kind} ${className}`}>{t}</span>;
}

/** One stage of the encoder: the shape that flows in on the edge above it, then the box. */
function Stage(props: { n: number; name: string; into?: string; weights: ReactNode; className?: string; children: ReactNode }) {
  return (
    <li className={`stage ${props.className ?? ''}`}>
      {props.into && (
        <div className="edge">
          <span className="shape">{props.into}</span>
        </div>
      )}
      <div className="box">
        <div className="head">
          <b>
            <span className="tag">{props.n}</span> {props.name}
          </b>
          <span className="tag w">{props.weights}</span>
        </div>
        {props.children}
      </div>
    </li>
  );
}

/**
 * §3's figure: the bi-encoder around the model, then one real question through every stage of
 * bge-base, then where the weights are and how far training moved them.
 */
export function Architecture({ dims, trainRun }: { dims: number[]; trainRun: string | null }) {
  const p = bertParams(MODEL);
  const n = EXAMPLE.tokens.length;
  const headDim = MODEL.hidden / MODEL.heads;
  const share = (w: number) => pc(w / p.total);
  const lo = Math.min(...MOVED.layers.slice(0, -1));
  const hi = Math.max(...MOVED.layers.slice(0, -1));
  const cuts = [...dims].sort((a, b) => b - a);
  return (
    <figure className="arch">
      <h3>The system: one encoder, used twice</h3>
      <p className="small">
        A <b>bi-encoder</b>. The question and the passage never meet inside the model: each becomes a vector on its own,
        and only the vectors are compared. That is what lets every passage be embedded once, ahead of time. A
        cross-encoder (a re-ranker, <a href="#alternatives">§14</a>) reads the pair together, which is more accurate but
        runs once per passage for every question.
      </p>
      <div className="bienc">
        <div className="in q">
          <span className="tag">question · embedded when asked</span>
          <Link to={dataQuestionPath(EXAMPLE.qid)}>{EXAMPLE.question}</Link>
        </div>
        <div className="in p">
          <span className="tag">passage · embedded once, into the index</span>
          <span>{EXAMPLE.passage}</span>
        </div>
        <span className="arr a1" aria-hidden="true">→</span>
        <span className="arr b1" aria-hidden="true">→</span>
        <div className="enc">
          <b>bge-base</b>
          <span className="tag">the same {short(p.total)} weights for both</span>
        </div>
        <span className="arr a2" aria-hidden="true">→</span>
        <span className="arr b2" aria-hidden="true">→</span>
        <div className="vec q">
          <b>q</b>
          <span className="tag">{MODEL.hidden} numbers, length 1</span>
        </div>
        <div className="vec p">
          <b>p</b>
          <span className="tag">{MODEL.hidden} numbers, length 1</span>
        </div>
        <span className="arr a3" aria-hidden="true">→</span>
        <span className="arr b3" aria-hidden="true">→</span>
        <div className="score">
          <b>score = q · p</b>
          <span className="tag">cosine similarity; the 10 highest-scoring passages are retrieved</span>
        </div>
      </div>

      <h3>Inside the encoder: one question, stage by stage</h3>
      <p className="small">
        The shape on each arrow is what flows into the next stage: {n} tokens for this question, each carried as{' '}
        {MODEL.hidden} numbers. Only stages 2 and 3 have weights, and only they change in training.
      </p>
      <ol className="stack">
        <Stage n={1} name="Tokenizer" weights="no weights">
          <p>
            Lowercases the text, splits it into word pieces from a fixed vocabulary of {int(MODEL.vocab)}, and adds{' '}
            <code>[CLS]</code> at the front and <code>[SEP]</code> at the end. "EBITDAR" is not in the vocabulary, so it
            becomes three pieces (a <code>##</code> piece continues the word before it). At most {MODEL.maxTokens}{' '}
            tokens; the rest is cut off.
          </p>
          <p className="said">“{EXAMPLE.question}”</p>
          <div className="toks" aria-label={`${n} tokens`}>
            {EXAMPLE.tokens.map((t, i) => (
              <Token key={i} t={t} />
            ))}
          </div>
        </Stage>

        <Stage
          n={2}
          name="Embeddings"
          into={`${n} token ids`}
          weights={`${int(p.embeddings)} weights · ${share(p.embeddings)}`}
        >
          <p>
            Each token id picks one row of a {int(MODEL.vocab)} × {MODEL.hidden} table. A position vector (where the
            token sits) and a segment vector are added, then LayerNorm rescales the sum. Every token is now{' '}
            {MODEL.hidden} numbers, but without context yet: "debt" gets this same vector in every sentence.
          </p>
          <div className="eqn">
            <span className="term">
              word <span className="tag">{int(MODEL.vocab)} × {MODEL.hidden}</span>
            </span>
            <span className="op">+</span>
            <span className="term">
              position <span className="tag">{MODEL.maxTokens} × {MODEL.hidden}</span>
            </span>
            <span className="op">+</span>
            <span className="term">
              segment <span className="tag">{MODEL.segments} × {MODEL.hidden}</span>
            </span>
            <span className="op">→</span>
            <span className="term">LayerNorm</span>
          </div>
        </Stage>

        <Stage
          n={3}
          name={`Encoder layer, × ${MODEL.layers}`}
          into={`${n} × ${MODEL.hidden}`}
          weights={`${int(p.layers)} weights · ${share(p.layers)} (${int(p.layer)} per layer)`}
          className="layers"
        >
          <div className="enc-layer">
            <div className="part">
              <div className="head">
                <b>Self-attention, {MODEL.heads} heads</b>
                <span className="tag">{int(p.attention)}</span>
              </div>
              <p>
                Every token looks at every other token, before and after it, and takes a weighted mix of their
                vectors. {MODEL.heads} heads do this in parallel, each on {headDim} numbers, and their results are
                joined back into {MODEL.hidden}. This is where "debt" picks up "adjusted" and "EBITDAR" around it.
              </p>
              <div className="heads" aria-label={`${MODEL.heads} heads of ${headDim} numbers`}>
                {Array.from({ length: MODEL.heads }, (_, i) => (
                  <span key={i}>{i + 1}</span>
                ))}
                <span className="tag">
                  {MODEL.heads} × {headDim} = {MODEL.hidden}
                </span>
              </div>
            </div>
            <div className="addnorm">+ add the input back (a residual), then LayerNorm</div>
            <div className="part">
              <div className="head">
                <b>Feed-forward</b>
                <span className="tag">{int(p.feedForward)}</span>
              </div>
              <p>
                Each token on its own: widen to {int(MODEL.ffn)} numbers, apply GELU, narrow back to {MODEL.hidden}.
              </p>
              <div className="widths" aria-label={`${MODEL.hidden} to ${MODEL.ffn} to ${MODEL.hidden}`}>
                {[MODEL.hidden, MODEL.ffn, MODEL.hidden].map((w, i) => (
                  <span key={i} style={{ '--f': (w / MODEL.ffn).toFixed(3) } as CSSProperties}>
                    {int(w)}
                    {i === 1 && ' · GELU'}
                  </span>
                ))}
              </div>
            </div>
            <div className="addnorm">+ add the input back, then LayerNorm</div>
          </div>
          <p className="small">
            Layer 1's output is layer 2's input, and so on up to {MODEL.layers}. The shape never changes: {n} ×{' '}
            {MODEL.hidden} in, {n} × {MODEL.hidden} out, but each vector now reflects the whole text.
          </p>
        </Stage>

        <Stage n={4} name="[CLS] pooling" into={`${n} × ${MODEL.hidden}, every token in context`} weights="no weights">
          <p>
            Keep the first row, the final vector of <code>[CLS]</code>, and drop the other {n - 1}. <code>[CLS]</code>{' '}
            stands for no word; through attention it has read every token, and BGE was trained to use it as the summary
            of the text. It is not an average over the tokens.
          </p>
          <div className="toks" aria-label="only [CLS] is kept">
            {EXAMPLE.tokens.map((t, i) => (
              <Token key={i} t={t} className={i === 0 ? 'kept' : 'dropped'} />
            ))}
          </div>
          <p className="small skipped">
            <b>Skipped:</b> BERT's own pooler, a {MODEL.hidden} × {MODEL.hidden} layer with tanh on top of{' '}
            <code>[CLS]</code> ({int(p.pooler)} weights). It is in the checkpoint, but CLS pooling reads the vector
            beneath it, so no gradient reaches it and training left it exactly as it was.
          </p>
        </Stage>

        <Stage n={5} name="Normalise" into={String(MODEL.hidden)} weights="no weights">
          <p>
            Divide by the vector's length so it has length 1. Then cosine similarity is a plain dot product, which a
            vector index computes fastest.
          </p>
        </Stage>

        <Stage n={6} name="Matryoshka cut (optional)" into={`${MODEL.hidden}, length 1`} weights="no weights">
          <p>
            For a smaller index, keep the first {cuts.slice(1).join(', ')} numbers and normalise again. The Matryoshka
            loss (<a href="#matryoshka">§7</a>) trained the leading numbers to carry the most, so a cut costs little.
          </p>
          <ul className="nest" aria-label="the sizes, as a share of the full vector">
            {cuts.map((d) => (
              <li key={d}>
                <span className="tag">{d}</span>
                <span className="bar" style={{ '--f': (d / MODEL.hidden).toFixed(3) } as CSSProperties} />
              </li>
            ))}
          </ul>
        </Stage>

        <li className="stage out">
          <div className="edge">
            <span className="shape">the embedding: {cuts.join(' / ')} numbers</span>
          </div>
        </li>
      </ol>

      <h3>Where the {int(p.total)} weights are, and how far training moved them</h3>
      <div
        className="wbar"
        role="img"
        aria-label={`Share of the weights: embeddings ${share(p.embeddings)}, ${MODEL.layers} layers ${share(p.layers)}, pooler ${share(p.pooler)}`}
      >
        <span className="seg emb" style={{ flexGrow: p.embeddings }} />
        {Array.from({ length: MODEL.layers }, (_, i) => (
          <span key={i} className="seg layer" style={{ flexGrow: p.layer }} />
        ))}
        <span className="seg pooler" style={{ flexGrow: p.pooler }} />
      </div>
      <div className="tablewrap">
        <table className="weights">
          <caption className="sr-only">Weights by part of the model, and how far training moved each part</caption>
          <thead>
            <tr>
              <th>part</th>
              <th className="num">weights</th>
              <th className="num">share</th>
              <th className="num">moved by training</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <span className="key emb" /> word embeddings ({int(MODEL.vocab)} × {MODEL.hidden})
              </td>
              <td className="num">{int(p.word)}</td>
              <td className="num">{share(p.word)}</td>
              <td className="num">
                {pc(MOVED.word, 2)}
                <div className="tag">
                  {int(MOVED.wordRows)} of {int(MODEL.vocab)} rows changed at all
                </div>
              </td>
            </tr>
            <tr>
              <td>
                <span className="key emb" /> position and segment embeddings, LayerNorm
              </td>
              <td className="num">{int(p.embeddings - p.word)}</td>
              <td className="num">{share(p.embeddings - p.word)}</td>
              <td className="num">{pc(MOVED.otherEmbeddings, 2)}</td>
            </tr>
            <tr>
              <td>
                <span className="key layer" /> {MODEL.layers} encoder layers
              </td>
              <td className="num">{int(p.layers)}</td>
              <td className="num">{share(p.layers)}</td>
              <td className="num">
                {pc(MOVED.allLayers, 2)}
                <div className="tag">
                  layers 1–{MODEL.layers - 1}: {pc(lo, 2)}–{pc(hi, 2)} each; layer {MODEL.layers}: {pc(MOVED.layers.at(-1)!, 2)}
                </div>
              </td>
            </tr>
            <tr>
              <td>
                <span className="key pooler" /> BERT's pooler (not used)
              </td>
              <td className="num">{int(p.pooler)}</td>
              <td className="num">{share(p.pooler)}</td>
              <td className="num">0 (no gradient)</td>
            </tr>
          </tbody>
          <tfoot>
            <tr>
              <td>the whole model</td>
              <td className="num">{int(p.total)}</td>
              <td className="num">100%</td>
              <td className="num">{pc(MOVED.all, 2)}</td>
            </tr>
          </tfoot>
        </table>
      </div>
      <figcaption className="small">
        <b>Moved</b> is the size of the change in a part's weights divided by the size of the weights (‖tuned − base‖ ÷
        ‖base‖). Fine-tuning nudges: the whole model moved {pc(MOVED.all, 2)}. Of the layers, the last, nearest the
        output, moved most. The word table moved least, because a word piece that never appears in the training text gets no
        gradient: {int(MODEL.vocab - MOVED.wordRows)} of its rows are untouched. Measured on the epoch-2 checkpoint of{' '}
        <Link to={runPath(MOVED.run)} className="mono">
          {MOVED.run}
        </Link>
        {trainRun && trainRun !== MOVED.run && ', a different run from the one this page describes'}.
      </figcaption>
    </figure>
  );
}
