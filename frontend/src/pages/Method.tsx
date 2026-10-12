import { Fragment, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import {
  type Overview as OverviewData,
  type Run,
  type RunSummary,
  type SplitStats,
  api,
  useGet,
} from '../lib/api';
import { DASH, delta, int, interval, metric, pText, pct, seconds } from '../lib/format';
import { type Recipe, bytes, indexBytes, recipeOf, sci } from '../lib/method';
import { LineChart, Loading } from '../lib/ui';
import { dataPath, overviewPath, runPath, searchPath } from '../lib/url';
import { BatchExample } from './MethodBatch';

/** Verified from the chosen checkpoint (config.json, 1_Pooling, tokenizer) on 2026-10-12. */
const MODEL = {
  hub: 'BAAI/bge-base-en-v1.5',
  revision: 'a5beb1e3',
  params: 109_482_240,
  layers: 12,
  heads: 12,
  hidden: 768,
  ffn: 3072,
  vocab: 30_522,
  maxTokens: 512,
};

/** Measured on the M1 Pro on 2026-10-12 (train.py's docstring) and in the check run. */
const MEASURED = {
  stepBefore: '15–20 s',
  stepDrift: '~9 s',
  stepAfter: '~2.3 s',
  phil768: [0.768, 0.825],
  phil64: [0.644, 0.789],
  philGpu: '3 min 26 s on one A10G',
  check: 0.748,
  checkRange: [0.728, 0.773],
  tuned64VsBase768: { delta: '+2.5 pts', ci: '+0.4 to +4.5', p: '0.027' },
  gradNorm: 'about 10 to 32',
  leak: '44 of 700',
};

/** The sections, in order; the number is the § the text cross-references. */
const TOC: [string, number | null, string][] = [
  ['short', null, 'The short version'],
  ['idea', 1, 'What fine-tuning changes'],
  ['pipeline', 2, 'The pipeline'],
  ['model', 3, 'The model'],
  ['data', 4, 'The training data'],
  ['loss', 5, 'The loss'],
  ['example', 6, 'One batch, live'],
  ['matryoshka', 7, 'Matryoshka'],
  ['batching', 8, 'Batching'],
  ['optimiser', 9, 'Optimiser and schedule'],
  ['selection', 10, 'Choosing the epoch'],
  ['mac', 11, 'Speed on the Mac'],
  ['scoring', 12, 'Scoring'],
  ['evidence', 13, 'Is the gain real?'],
  ['alternatives', 14, 'Alternatives'],
  ['limits', 15, 'Limits'],
  ['explain', 16, 'Explaining it'],
  ['glossary', 17, 'Glossary'],
  ['reproduce', 18, 'Reproduce it'],
];

const r10 = (r: RunSummary | null | undefined, dim: string) => r?.metrics[dim]?.['recall@10'];

/**
 * `/method` — how the tuned model was made: every setting, why, and what else was possible.
 * Numbers come from the train run the overview's tuned eval points at, split v1 and the
 * overview's compare; constants measured once are in MODEL and MEASURED above.
 */
export function Method() {
  const { data: ov, error } = useGet<OverviewData>(api.overview());
  const { data: split } = useGet<SplitStats>(api.split());
  const trainId = ov?.tuned?.retriever?.train_run ?? null;
  const { data: train } = useGet<Run>(trainId ? api.run(trainId) : null);
  if (!ov) return <Loading error={error} what="the method" />;
  const rec = recipeOf(train, split?.pairs.train);
  const t = train?.train;
  const c = ov.compare;
  const { base, tuned } = ov;
  const bm25 = ov.baselines.find((r) => r.retriever?.type === 'bm25');
  const queries = split?.queries ?? ov.split?.queries;
  const passages = split?.passages ?? ov.split?.passages;
  const nPairs = t?.n_pairs ?? split?.pairs.train;
  const chosen = t?.chosen;
  const f: Facts = { rec, nPairs, queries, passages, base, tuned, chosen: chosen?.epoch, secs: t?.train_seconds };
  return (
    <>
      <p className="eyebrow">Method · how the tuned model was made</p>
      <h1>
        How bge-base was fine-tuned, and <em>why this way</em>
      </h1>
      <p className="lede">
        Every setting behind the tuned model, the reason for it, and what else could have been done. The numbers come
        from the train run's record{' '}
        {trainId ? (
          <Link to={runPath(trainId)} className="mono">
            {trainId}
          </Link>
        ) : (
          '(none yet)'
        )}{' '}
        and split v1. The worked example in §6 is computed live from both models.
      </p>

      <div className="verdict headline" role="group" aria-label="The run in four numbers">
        <div className="stat">
          <div className="n">{int(nPairs)}</div>
          <div className="l">training pairs (question, golden passage)</div>
        </div>
        <div className="stat">
          <div className="n">{int(rec.total_steps)}</div>
          <div className="l">
            steps: {rec.epochs} epochs × {int(rec.steps_per_epoch)}, batch {rec.batch_size}
          </div>
        </div>
        <div className="stat">
          <div className="n">{t ? seconds(t.train_seconds) : DASH}</div>
          <div className="l">training time on {deviceName(train?.device)}</div>
        </div>
        <div className="stat accent">
          <div className="n">
            {pct(r10(base, '768'))} → {pct(r10(tuned, '768'))}
          </div>
          <div className="l">Recall@10 on test, 768 dims, base → tuned</div>
        </div>
      </div>

      <nav className="toc" aria-label="On this page">
        <span className="tag">On this page</span>
        <ol>
          {TOC.map(([id, n, label]) => (
            <li key={id}>
              <span className="tag">{n ?? '·'}</span> <a href={`#${id}`}>{label}</a>
            </li>
          ))}
        </ol>
      </nav>

      <Short f={f} fixed={c?.fixed} broken={c?.broken} />
      <Idea />
      <Pipeline f={f} />
      <Model f={f} />
      <Data f={f} split={split} />
      <Loss rec={rec} />
      <section id="example">
        <span className="eyebrow">6 · One batch, computed live</span>
        <h2>What the loss sees, with real pairs</h2>
        <p className="sub">
          Real pairs, scored by the base and the tuned model with the harness's own code. The loss column is the formula
          above, and a test checks it against Sentence Transformers' MNRL to six decimal places.
        </p>
        <BatchExample batchSize={rec.batch_size} />
      </section>
      <Matryoshka f={f} />
      <Batching f={f} split={split} />
      <Optimiser rec={rec} train={train} />
      <Selection rec={rec} train={train} valQ={queries?.val} />
      <Mac rec={rec} train={train} />
      <Scoring f={f} />
      <Evidence ov={ov} />
      <Alternatives f={f} bm25={bm25} />
      <Limits />
      <Explain f={f} ov={ov} />
      <Glossary />
      <Reproduce trainId={trainId} />
    </>
  );
}

type Facts = {
  rec: Recipe;
  nPairs: number | undefined;
  queries: { train: number; val: number; test: number } | undefined;
  passages: number | undefined;
  base: RunSummary | null;
  tuned: RunSummary | null;
  chosen: number | undefined;
  secs: number | undefined;
};

function deviceName(d: string | null | undefined): string {
  if (d === 'mps') return 'the Mac GPU (MPS)';
  if (d === 'cuda') return 'a CUDA GPU';
  return d ?? DASH;
}

function Sec({ id, n, eyebrow, title, children }: { id: string; n: number; eyebrow: string; title: ReactNode; children: ReactNode }) {
  return (
    <section id={id}>
      <span className="eyebrow">
        {n} · {eyebrow}
      </span>
      <h2>{title}</h2>
      {children}
    </section>
  );
}

const Code = ({ children }: { children: ReactNode }) => <code>{children}</code>;
const Src = ({ children }: { children: ReactNode }) => <span className="tag src">{children}</span>;

// ── 0 · the short version ────────────────────────────────────────────────
function Short({ f, fixed, broken }: { f: Facts; fixed?: number; broken?: number }) {
  const { rec } = f;
  return (
    <section id="short">
      <span className="eyebrow">The short version</span>
      <h2>In one paragraph</h2>
      <div className="callout big">
        <p>
          The base model already turns text into vectors so that a question lands near the passages that answer it.
          Fine-tuning showed it {int(f.nPairs)} finance (question, passage) pairs, each question with the 10-K passage
          that answers it, in batches of {rec.batch_size}. For every question the loss rewards the model when the question's own passage scores
          above the other {rec.batch_size - 1} passages in its batch, and penalises it when it doesn't. After{' '}
          {rec.epochs} passes over the data, the epoch that scored best on {int(f.queries?.val)} held-out val questions
          was kept (epoch {f.chosen ?? DASH}). Then the {int(f.queries?.test)} test questions were scored, once. The
          golden passage now lands in the top 10 for <b>{pct(r10(f.tuned, '768'))}</b> of test questions, up from{' '}
          <b>{pct(r10(f.base, '768'))}</b>: {int(fixed)} questions fixed, {int(broken)} broken.
        </p>
      </div>
    </section>
  );
}

// ── 1 · the idea ─────────────────────────────────────────────────────────
function Idea() {
  return (
    <Sec id="idea" n={1} eyebrow="The idea" title="What fine-tuning changes, and what it doesn't">
      <div className="split">
        <div>
          <h3>Retrieval is a nearest-neighbour search</h3>
          <p>
            The model turns every passage into a list of 768 numbers (a vector) once, and each question into a vector
            when it is asked. Passages are ranked by <b>cosine similarity</b>, the angle between the question's vector and
            the passage's. Recall@10 asks one thing: is the golden passage among the ten nearest?
          </p>
          <h3>Fine-tuning changes the weights, not the vectors</h3>
          <p>
            Training adjusts the model's {int(MODEL.params)} weights. Nothing is edited in an index. After training,
            every passage is embedded again with the new weights. That is why the harness caches corpus embeddings per
            model, and why fine-tuning a production retriever means rebuilding its index.
          </p>
        </div>
        <div>
          <h3>Why it can help</h3>
          <p>
            bge-base learned from large general-purpose collections of text pairs. 10-K filings are narrow: every company
            has a risk-factor paragraph, a revenue table and a dividend note, phrased alike. The base model sees those
            passages as close to each other. In-domain pairs teach it which differences matter here: which company,
            which year, which line item.
          </p>
          <h3>Why it can hurt</h3>
          <p>
            A model pulled towards one dataset can get worse at others. This project hasn't measured that yet: the
            FiQA2018 check (M5) is pending. See <a href="#limits">§15</a>.
          </p>
        </div>
      </div>
    </Sec>
  );
}

// ── 2 · the pipeline ─────────────────────────────────────────────────────
function Pipeline({ f }: { f: Facts }) {
  const { rec } = f;
  const steps: [string, ReactNode, string][] = [
    ['Pin the data', <>The Hub dataset <Code>philschmid/finanical-rag-embedding-dataset</Code> at revision <Code>e0b17819</Code>: 7,000 (question, passage) rows.</>, 'config.py · data.fetch_dataset'],
    ['Freeze split v1', <>Deduplicate to {int(f.passages)} passages, group passages that share a question, and deal whole groups 80/10/10 with seed 42.</>, 'data.build_split → data/splits/v1/'],
    ['Build training pairs', <>One pair per golden passage of each train question: {int(f.nPairs)} pairs. Val and test questions are never used here.</>, 'Split.pairs("train") · data.py:259'],
    ['Load the base model', <><Code>{MODEL.hub}</Code> at revision <Code>{MODEL.revision}</Code>, padding set to multiples of {rec.pad_to_multiple_of} tokens.</>, 'train.py:107'],
    ['Wrap the loss', <>MultipleNegativesRankingLoss inside MatryoshkaLoss at {rec.matryoshka_dims.join(', ')} dims.</>, 'train.py:114'],
    ['Train', <>{rec.epochs} epochs, {int(rec.total_steps)} steps, AdamW at {sci(rec.learning_rate)} with warmup, batches with no duplicate text. A checkpoint is saved at the end of every epoch.</>, 'train.py:117–142'],
    ['Score every checkpoint on val', <>The same harness that scores test: Recall@1/5/10, nDCG@10 and MRR@10 at 768 and 64 dims.</>, 'train.py:151'],
    ['Choose one', <>Highest val Recall@10 at 768 dims; ties go to nDCG@10, then the earlier epoch. Epoch {f.chosen ?? DASH} won.</>, 'train.py:168'],
    ['Write the record', <><Code>runs/&lt;id&gt;/run.json</Code>: the config, val metrics per epoch, the loss curve, timings and the device.</>, 'runs.write_run'],
    ['Score test, once', <><Code>finrag eval</Code> scores the chosen checkpoint on test. A second test run of the same checkpoint is refused without <Code>--force</Code>.</>, 'cli.py:229'],
    ['Compare, log and register', <>Paired comparison with the base model; the run is replayed into the central MLflow and the checkpoint is registered as <Code>finance-rag-sft-bge-base</Code>.</>, 'compare.py · tracking.py'],
  ];
  return (
    <Sec id="pipeline" n={2} eyebrow="End to end" title="The pipeline, step by step">
      <p className="sub">
        Each step reads only what an earlier step wrote, so any step can be rerun on its own. The tags are where the
        code is in <Code>src/finance_rag_sft/</Code>.
      </p>
      <div className="steps">
        {steps.map(([title, body, src]) => (
          <div key={title} className="step">
            <b>{title}</b>
            {body}
            <span className="tag">{src}</span>
          </div>
        ))}
      </div>
    </Sec>
  );
}

// ── 3 · the model ────────────────────────────────────────────────────────
function Model({ f }: { f: Facts }) {
  const flow: [string, string][] = [
    ['text', 'question or passage'],
    ['tokens', `lowercased WordPiece, ≤ ${MODEL.maxTokens}`],
    ['encoder', `${MODEL.layers} transformer layers`],
    ['[CLS] vector', `${MODEL.hidden} numbers`],
    ['normalise', 'length 1, so cosine = dot product'],
    ['embedding', `${f.rec.matryoshka_dims.join(' / ')} dims`],
  ];
  return (
    <Sec id="model" n={3} eyebrow="The model" title={<>The model being tuned: <code>bge-base-en-v1.5</code></>}>
      <ol className="flow" aria-label="How a text becomes an embedding">
        {flow.map(([k, v]) => (
          <li key={k}>
            <b>{k}</b>
            <span className="tag">{v}</span>
          </li>
        ))}
      </ol>
      <div className="split">
        <dl className="kv">
          <dt>model</dt>
          <dd>
            <Code>{MODEL.hub}</Code>, pinned at revision <Code>{MODEL.revision}</Code>. MIT licence.
          </dd>
          <dt>architecture</dt>
          <dd>
            A BERT-base encoder: {MODEL.layers} layers, {MODEL.heads} attention heads, hidden size {MODEL.hidden},
            feed-forward {int(MODEL.ffn)}
          </dd>
          <dt>parameters</dt>
          <dd>{int(MODEL.params)}, every one trained (full fine-tune)</dd>
          <dt>tokenizer</dt>
          <dd>WordPiece, {int(MODEL.vocab)}-token vocabulary, lowercases everything (an uncased model)</dd>
          <dt>max length</dt>
          <dd>{MODEL.maxTokens} tokens; anything longer is cut off</dd>
          <dt>pooling</dt>
          <dd>the final hidden state of the [CLS] token, not an average over tokens</dd>
          <dt>output</dt>
          <dd>768 float32 numbers (3,072 bytes), scaled to length 1</dd>
          <dt>query prompt</dt>
          <dd>none, for base and tuned alike</dd>
          <dt>precision</dt>
          <dd>float32 throughout</dd>
        </dl>
        <div>
          <h3>Why this model (decision D1)</h3>
          <ul className="tight">
            <li>
              <b>Checkable.</b> It is the model in the reference recipe (s63 UC1, Phil Schmid's notebook), so the harness
              can be checked against a published number before any training. It was: <a href="#scoring">§12</a>.
            </li>
            <li>
              <b>The right size for a laptop.</b> 110M parameters train in about 20 minutes on an M1 Pro. A large
              model (335M) would take roughly three times as long.
            </li>
            <li>
              <b>Strong for its size</b> on MTEB retrieval, with a permissive licence and a pinned revision.
            </li>
            <li>
              <b>A same-family control.</b> bge-small (33M parameters, 384 dims) is the baseline that shows what model
              size alone buys.
            </li>
          </ul>
          <h3>Why no query prompt</h3>
          <p className="small">
            bge v1.5 accepts an optional instruction in front of queries ("Represent this sentence for searching
            relevant passages: "). The reference recipe uses none, and v1.5 was built to work without one. Base and tuned
            get the same empty prompt, so the prompt is not a difference between them.
          </p>
        </div>
      </div>
    </Sec>
  );
}

// ── 4 · the data ─────────────────────────────────────────────────────────
function Data({ f, split }: { f: Facts; split: SplitStats | null }) {
  const q = f.queries;
  const extra = f.nPairs != null && q ? f.nPairs - q.train : null;
  return (
    <Sec id="data" n={4} eyebrow="The training data" title="What the model learned from, and what it never saw">
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>split</th>
              <th className="num">questions</th>
              <th className="num">pairs</th>
              <th>used for</th>
            </tr>
          </thead>
          <tbody>
            <tr className="chosen">
              <td className="k">train</td>
              <td className="n num">{int(q?.train)}</td>
              <td className="n num">{int(split?.pairs.train ?? f.nPairs)}</td>
              <td>the only data the model trained on</td>
            </tr>
            <tr>
              <td className="k">val</td>
              <td className="n num">{int(q?.val)}</td>
              <td className="n num">{int(split?.pairs.val)}</td>
              <td>choosing the epoch, nothing else</td>
            </tr>
            <tr>
              <td className="k">test</td>
              <td className="n num">{int(q?.test)}</td>
              <td className="n num">{int(split?.pairs.test)}</td>
              <td>scored once, after the choice was made</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div className="split">
        <div>
          <h3>Pairs, not labels</h3>
          <p>
            Each training example is a (question, golden passage) pair. There are more pairs than train questions (
            {extra != null ? int(extra) : DASH} more) because some questions have more than one golden passage (
            {int(split?.multi_golden_queries)} across the dataset). There are no labelled negatives: the passages a
            question must rank below are other questions' passages in the same batch (§5).
          </p>
          <h3>Lengths</h3>
          <p>
            Questions have a median of {int(split?.lengths.question_words.median)} words. Passages have a median of{' '}
            {int(split?.lengths.passage_words.median)} words (p95 {int(split?.lengths.passage_words.p95)}, max{' '}
            {int(split?.lengths.passage_words.max)}). Most are single sentences or table rows from several companies'
            2023 10-K filings. {int(split?.truncated_passages)} passages run past 512 tokens and are cut off.
          </p>
        </div>
        <div>
          <h3>Why the split is by passage group</h3>
          <p>
            {int(split?.duplicates.passages_repeated)} passages appear more than once in the 7,000 rows (one{' '}
            {int(split?.duplicates.max_repeats)} times). A random row split puts about {MEASURED.leak} test questions'
            passages into train. The model would be tested on passages it was trained to retrieve, and the score would
            be inflated. Split v1 joins passages that share a question into {int(split?.groups)} groups and deals whole
            groups, so no golden passage crosses train, val and test.
          </p>
          <p className="small">
            Leak check:{' '}
            {split?.leak_check.ok ? (
              <span className="badge ok">✓ zero overlap</span>
            ) : (
              <span className="badge warn">○ not loaded</span>
            )}{' '}
            · training passages stay in the corpus at test time as distractors · <Link to={dataPath()}>browse the data →</Link>
          </p>
        </div>
      </div>
    </Sec>
  );
}

// ── 5 · the loss ─────────────────────────────────────────────────────────
function Loss({ rec }: { rec: Recipe }) {
  const B = rec.batch_size;
  return (
    <Sec id="loss" n={5} eyebrow="The objective" title="The loss: MultipleNegativesRankingLoss (MNRL)">
      <div className="split">
        <div>
          <p>
            Each training step takes a batch of B = {B} pairs (q<sub>1</sub>, p<sub>1</sub>) … (q<sub>B</sub>, p
            <sub>B</sub>). The model embeds all {2 * B} texts. Each question is scored against every passage in the
            batch, giving a {B} × {B} matrix of cosine similarities. The diagonal holds the right answers; every
            off-diagonal entry is a wrong one. The loss is a {B}-way classification for each row: <b>pick your own
            passage</b>.
          </p>
          <pre className="formula" aria-label="The MNRL formula">
            {`s_ij   = cos(q_i, p_j)        −1 … 1

loss_i = −log  exp(20·s_ii)
              ─────────────────
              Σ_j exp(20·s_ij)

loss   = mean of loss_i over the batch`}
          </pre>
          <p className="small">
            Sentence Transformers v6's defaults, as used: cosine similarity, scale 20, query → passage only.{' '}
            <Src>train.py:114 · explain.py</Src>
          </p>
        </div>
        <div>
          <h3>The parts</h3>
          <dl className="kv">
            <dt>in-batch negatives</dt>
            <dd>
              The other {B - 1} passages in the batch are free negatives that nobody labelled. Batches are reshuffled
              every epoch, so over {rec.epochs} epochs a question meets about {rec.epochs * (B - 1)} different wrong
              passages.
            </dd>
            <dt>scale 20</dt>
            <dd>
              Cosines sit between −1 and 1, so a softmax over them would be nearly flat. Multiplying by 20 (a
              temperature of 0.05) sharpens it. A passage that is only slightly closer takes most of the probability,
              and the loss keeps pushing until the right passage is clearly ahead.
            </dd>
            <dt>the gradient</dt>
            <dd>
              It raises s<sub>ii</sub> and lowers each wrong s<sub>ij</sub> in proportion to the probability that
              passage took. Wrong passages that already score high (hard negatives) get most of the push; easy ones get
              almost none.
            </dd>
            <dt>direction</dt>
            <dd>
              Question → passage only. A symmetric variant also asks each passage to pick its question (see
              alternatives).
            </dd>
          </dl>
          <h3>Why this loss</h3>
          <ul className="tight">
            <li>It needs only positive pairs, which is exactly what the dataset has.</li>
            <li>
              Its shape is the task's shape: rank one right passage above many wrong ones. Recall@10 measures the same
              thing over the whole corpus.
            </li>
            <li>
              It is the standard choice for (question, answer) data in Sentence Transformers, and the same contrastive
              family bge itself was trained with.
            </li>
          </ul>
        </div>
      </div>
    </Sec>
  );
}

// ── 7 · Matryoshka ───────────────────────────────────────────────────────
function Matryoshka({ f }: { f: Facts }) {
  const n = f.passages ?? 0;
  const rows: [string, number][] = [
    ['768', 768],
    ['64', 64],
  ];
  return (
    <Sec id="matryoshka" n={7} eyebrow="Matryoshka representation learning" title="One model, five sizes">
      <div className="split">
        <div>
          <p>
            MatryoshkaLoss wraps MNRL. Each step runs the encoder <b>once</b>. It then takes the first{' '}
            {f.rec.matryoshka_dims.join(', ')} numbers of every embedding, rescales each cut-down vector to length 1,
            computes MNRL at each size and adds the five losses, each with weight 1.
          </p>
          <p>
            The gradient therefore pushes the most useful information into the first dimensions, like nested dolls. A
            64-number prefix becomes a usable embedding on its own. The base model was never trained this way, so cutting
            its vectors throws away information spread across all 768.
          </p>
        </div>
        <div>
          <p>
            The tuned model at <b>64</b> dims beats the base model at <b>768</b>: {MEASURED.tuned64VsBase768.delta}{' '}
            Recall@10 (paired 95% CI {MEASURED.tuned64VsBase768.ci}, p = {MEASURED.tuned64VsBase768.p}). The index is
            12× smaller and retrieves better. The reference run saw the same thing in nDCG@10: {MEASURED.phil64[1]} at
            64 against {MEASURED.phil768[0]} at 768.
          </p>
          <p className="small">
            The logged training loss is the five-way sum. That is why it starts near 2 although each size's part starts
            near 0.4. The encoder runs once per step, so the five extra loss computations add little time.
          </p>
        </div>
      </div>
      <h3>What the small size buys</h3>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>size</th>
              <th className="num">bytes / passage</th>
              <th className="num">index, {int(n)} passages</th>
              <th className="num">base R@10</th>
              <th className="num">tuned R@10</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([d, k]) => {
              const b = indexBytes(k, n);
              return (
                <tr key={d}>
                  <td className="k">{d} dims</td>
                  <td className="n num">{bytes(b.per)}</td>
                  <td className="n num">{bytes(b.total)}</td>
                  <td className="n num">{pct(r10(f.base, d))}</td>
                  <td className="n num rec">{pct(r10(f.tuned, d))}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="small">Only 768 and 64 are scored on test. 512, 256 and 128 are trained but not reported.</p>
    </Sec>
  );
}

// ── 8 · batching ─────────────────────────────────────────────────────────
function Batching({ f, split }: { f: Facts; split: SplitStats | null }) {
  const { rec } = f;
  const last = f.nPairs != null ? f.nPairs - (rec.steps_per_epoch - 1) * rec.batch_size : null;
  return (
    <Sec id="batching" n={8} eyebrow="Batching" title="Batches with no duplicate text">
      <div className="split">
        <div>
          <p>
            MNRL treats every off-diagonal passage as wrong. Suppose the same passage text sat twice in one batch. The
            loss would then push a question away from its own correct answer. In this dataset{' '}
            {int(split?.duplicates.passages_repeated)} passages repeat (one {int(split?.duplicates.max_repeats)} times),
            and a question with two golden passages forms two pairs with the same question text.
          </p>
          <p>
            The <Code>NO_DUPLICATES</Code> batch sampler fills each batch so that no text appears twice, in either
            column. These false negatives never reach the loss. <Src>train.py:124</Src>
          </p>
        </div>
        <dl className="kv">
          <dt>batch size</dt>
          <dd>{rec.batch_size} pairs, so {rec.batch_size - 1} negatives per question per step</dd>
          <dt>steps / epoch</dt>
          <dd>
            {int(f.nPairs)} pairs ÷ {rec.batch_size} = {int(rec.steps_per_epoch)} steps (the last batch holds{' '}
            {last != null && last > 0 ? int(last) : DASH}; it is kept, not dropped)
          </dd>
          <dt>shuffling</dt>
          <dd>reshuffled every epoch, seeded ({rec.seed}) so a rerun draws the same batches</dd>
          <dt>why 32</dt>
          <dd>
            The recipe's value. Larger batches mean more negatives per step and usually help this loss, but memory grows
            with the batch because all 2B texts are encoded together. CachedMNRL removes that limit (alternatives).
          </dd>
        </dl>
      </div>
    </Sec>
  );
}

// ── 9 · optimiser ────────────────────────────────────────────────────────
function Optimiser({ rec, train }: { rec: Recipe; train: Run | null }) {
  const t = train?.train;
  const curve = t?.loss_curve ?? [];
  const yMax = Math.max(0.5, Math.ceil(Math.max(...curve.map((p) => p.loss), 0) * 2) / 2);
  const epochMarks = (t?.epochs ?? []).map((e) => ({ x: e.step, label: `epoch ${e.epoch}`, on: e.epoch === t?.chosen.epoch }));
  const total = rec.total_steps;
  const w = rec.warmup_steps;
  return (
    <Sec id="optimiser" n={9} eyebrow="Optimiser, schedule and precision" title="How the weights were updated">
      <div className="split">
        <dl className="kv">
          <dt>optimiser</dt>
          <dd>AdamW, PyTorch's fused implementation (the Transformers v5 default)</dd>
          <dt>learning rate</dt>
          <dd>peak {sci(rec.learning_rate)}</dd>
          <dt>warmup</dt>
          <dd>
            linear from 0 over the first {int(w)} steps ({Math.round(rec.warmup_ratio * 100)}% of {int(total)})
          </dd>
          <dt>decay</dt>
          <dd>linear, down to 0 at step {int(total)}</dd>
          <dt>Adam β / ε</dt>
          <dd>0.9, 0.999 / 1e-8</dd>
          <dt>weight decay</dt>
          <dd>0</dd>
          <dt>gradient clipping</dt>
          <dd>
            total norm capped at 1.0. The first 90 steps logged norms of {MEASURED.gradNorm} before clipping, so the
            cap was active from the start.
          </dd>
          <dt>precision</dt>
          <dd>float32, no fp16 or bf16</dd>
          <dt>trainable</dt>
          <dd>all {int(MODEL.params)} parameters; nothing frozen, no adapters</dd>
          <dt>epochs · seed</dt>
          <dd>
            {rec.epochs} · {rec.seed} (the seed fixes shuffling; every weight starts from the pre-trained model)
          </dd>
          <dt>eval while training</dt>
          <dd>off. Each epoch is saved and scored afterwards by the harness (<a href="#selection">§10</a>).</dd>
        </dl>
        <div>
          <h3>Why these values</h3>
          <p>
            They are the reference recipe's, unchanged on purpose. The project asks whether fine-tuning helps, so the
            recipe stays fixed and checkable rather than tuned to this test set.
          </p>
          <ul className="tight">
            <li>
              <b>A small learning rate</b> nudges a pre-trained model. Large steps overwrite what it already knows
              (catastrophic forgetting).
            </li>
            <li>
              <b>Warmup</b> keeps early steps small while Adam's running averages of the gradient are still unreliable.
            </li>
            <li>
              <b>Linear decay</b> makes the last steps small, so the final weights settle rather than bounce.
            </li>
            <li>
              <b>Clipping</b> caps any one step, so one odd batch cannot move the weights far.
            </li>
          </ul>
        </div>
      </div>
      <div className="split charts">
        {curve.length > 0 ? (
          <LineChart
            title="Training loss (the five-size Matryoshka sum), logged every 10 steps"
            points={curve.map((p) => [p.step, p.loss])}
            xMax={total}
            yMax={yMax}
            yTicks={[
              [0, '0'],
              [yMax / 2, (yMax / 2).toFixed(1)],
              [yMax, yMax.toFixed(1)],
            ]}
            marks={epochMarks}
            caption={
              <>
                Falls fast in epoch 1, then slowly; the solid line is the chosen checkpoint (epoch {t?.chosen.epoch}).
                Each point is the mean of the last 10 steps' losses.
              </>
            }
          />
        ) : (
          <div className="empty">No loss curve recorded on this run.</div>
        )}
        <LineChart
          title="Learning rate by step"
          points={[
            [0, 0],
            [w, rec.learning_rate],
            [total, 0],
          ]}
          xMax={total}
          yMax={rec.learning_rate}
          yTicks={[
            [0, '0'],
            [rec.learning_rate, sci(rec.learning_rate)],
          ]}
          marks={[{ x: w, label: `warmup ${w}` }, ...epochMarks.map((m) => ({ ...m, label: '' }))]}
          caption={
            <>
              Up over {int(w)} steps, then straight down to 0. Dashed lines are epoch ends; the solid one is the chosen
              checkpoint.
            </>
          }
          alt
        />
      </div>
    </Sec>
  );
}

// ── 10 · selection ───────────────────────────────────────────────────────
function Selection({ rec, train, valQ }: { rec: Recipe; train: Run | null; valQ?: number }) {
  const t = train?.train;
  const eps = t?.epochs ?? [];
  const r = eps.map((e) => e.val['768']?.['recall@10'] ?? 0);
  const spread = r.length ? Math.max(...r) - Math.min(...r) : null;
  const curve = t?.loss_curve ?? [];
  return (
    <Sec id="selection" n={10} eyebrow="Choosing the checkpoint" title="Choose on val, score test once">
      {eps.length > 0 ? (
        <div className="tablewrap">
          <table>
            <thead>
              <tr>
                <th className="num">epoch</th>
                <th className="num">step</th>
                <th className="num">train loss</th>
                <th className="num">val R@10 · 768</th>
                <th className="num">val R@10 · 64</th>
                <th className="num">val nDCG@10 · 768</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {eps.map((e) => {
                const on = e.epoch === t?.chosen.epoch;
                return (
                  <tr key={e.epoch} className={on ? 'chosen' : ''}>
                    <td className="n num">{e.epoch}</td>
                    <td className="n num">{int(e.step)}</td>
                    <td className="n num">{e.loss == null ? DASH : e.loss.toFixed(3)}</td>
                    <td className={`n num ${on ? 'rec' : ''}`}>{metric('recall@10', e.val['768']?.['recall@10'])}</td>
                    <td className="n num">{metric('recall@10', e.val['64']?.['recall@10'])}</td>
                    <td className="n num">{metric('ndcg@10', e.val['768']?.['ndcg@10'])}</td>
                    <td>{on && <span className="badge ok">✓ chosen</span>}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="empty">No train run yet: M3 writes one row per epoch here.</div>
      )}
      <div className="split">
        <div>
          <h3>Loss keeps falling; val doesn't</h3>
          <p>
            The training loss went from {curve[0] ? curve[0].loss.toFixed(2) : DASH} at step {curve[0]?.step ?? DASH}{' '}
            to {curve.at(-1) ? curve.at(-1)!.loss.toFixed(2) : DASH} at step {curve.at(-1)?.step ?? DASH}. Val Recall@10
            was flat after epoch 1. In the later epochs the model got better at its training pairs without getting
            better at new questions: it was starting to memorise. That is the reason to choose on held-out data.
          </p>
          <p className="small">
            The "train loss" column is the logged value nearest each checkpoint, an average over 10 steps, so it is
            noisy from one row to the next.
          </p>
        </div>
        <div>
          <h3>The rule, set before training</h3>
          <p>
            Highest val Recall@10 at 768 dims; ties go to val nDCG@10, then to the earlier epoch.{' '}
            <Src>train.py:168</Src> The epochs differ by{' '}
            {spread != null ? `${(spread * 100).toFixed(1)} points` : DASH}, about{' '}
            {spread != null && valQ ? int(Math.round(spread * valQ)) : DASH} of {int(valQ)} questions. That is within
            noise, so any epoch was defensible. The rule simply makes the choice before test is seen.
          </p>
          <h3>Why not choose on test</h3>
          <p>
            Picking the best of {rec.epochs} test scores and then reporting it overstates the result, because the
            reported number includes the luck of the pick. Test stays untouched until one checkpoint is fixed.{' '}
            <Code>finrag eval</Code> refuses to score the same checkpoint on test twice without <Code>--force</Code>.
          </p>
        </div>
      </div>
    </Sec>
  );
}

// ── 11 · the Mac ─────────────────────────────────────────────────────────
function Mac({ rec, train }: { rec: Recipe; train: Run | null }) {
  const secs = train?.train?.train_seconds;
  const perStep = secs != null && rec.total_steps ? secs / rec.total_steps : null;
  return (
    <Sec id="mac" n={11} eyebrow="Speed" title="Making it fast on a Mac GPU">
      <p>
        Training ran natively on an M1 Pro's GPU through PyTorch's MPS backend. The first measurement ran at{' '}
        {MEASURED.stepBefore} per step, which would have meant 3 to 4 hours. Two fixes, both measured on 2026-10-12:
      </p>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>problem</th>
              <th>cause</th>
              <th>fix</th>
              <th>step time</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="k">kernel recompiles</td>
              <td>
                Each batch was padded to its own longest text, so almost every batch had a new tensor shape, and MPS
                compiles new GPU kernels for each new shape.
              </td>
              <td>
                Pad to a multiple of {rec.pad_to_multiple_of} tokens: at most 8 sequence lengths (64 … 512), each compiled once.{' '}
                <Src>train.py:109</Src>
              </td>
              <td className="nw">
                {MEASURED.stepBefore} → fast, then drifting to {MEASURED.stepDrift}
              </td>
            </tr>
            <tr>
              <td className="k">memory creeping into swap</td>
              <td>
                The MPS allocator kept freed blocks cached. Memory grew past physical RAM and into swap (19 of 20 GB of
                swap in use).
              </td>
              <td>
                <Code>torch.mps.empty_cache()</Code> after every step, from a trainer callback (MPS only).{' '}
                <Src>train.py:70</Src>
              </td>
              <td className="nw">
                {MEASURED.stepDrift} → {MEASURED.stepAfter}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <div className="split">
        <div>
          <p>
            Result: {int(rec.total_steps)} steps in {secs != null ? seconds(secs) : DASH} (
            {perStep != null ? `${perStep.toFixed(1)} s` : DASH} per step, including four checkpoint saves). Val scoring
            of the checkpoints ran afterwards.
          </p>
          <p>
            <b>Neither fix changes the maths.</b> Padding tokens are masked out of attention, so the [CLS] vector does
            not depend on how much padding there is. Scoring resets the padding (<Src>harness.py:107</Src>), so base
            and tuned are encoded the same way at test time.
          </p>
        </div>
        <ul className="tight">
          <li>
            <Code>make train</Code> runs under <Code>caffeinate</Code>, so the Mac doesn't sleep mid-run.
          </li>
          <li>Float32, as the recipe advises for MPS. Mixed precision is the first speed lever on a CUDA GPU.</li>
          <li>
            Docker on a Mac can't reach the Apple GPU, so Mac training stays native. The <Code>train</Code> image (CUDA
            12.6) runs the same code on an NVIDIA host. There the cache callback is off and the padding is harmless.
            The reference run took {MEASURED.philGpu}.
          </li>
        </ul>
      </div>
    </Sec>
  );
}

// ── 12 · scoring ─────────────────────────────────────────────────────────
function Scoring({ f }: { f: Facts }) {
  return (
    <Sec id="scoring" n={12} eyebrow="Scoring" title="Base and tuned go through the same harness">
      <div className="split">
        <div>
          <ol className="tight">
            <li>
              Encode all {int(f.passages)} passages, 64 at a time, each batch padded to its longest text (max 512
              tokens). Cached on disk per model.
            </li>
            <li>Encode the questions the same way, with the same empty prompt.</li>
            <li>Cut both to the size being scored (768 or 64) and rescale to length 1.</li>
            <li>Dot product with every passage, which equals cosine similarity at length 1.</li>
            <li>A stable sort, so ties keep corpus order and reruns give identical ranks.</li>
            <li>
              Record each golden passage's rank over the <b>whole</b> corpus, not just the top 10. A move from rank 40
              to 12 still shows.
            </li>
          </ol>
          <p className="small">
            <Src>harness.py · score_matrix, DenseRetriever.scores</Src> The only difference between the base row and the
            tuned row is the weights.
          </p>
        </div>
        <dl className="kv">
          <dt>Recall@k</dt>
          <dd>the share of questions with any golden passage in the top k (the headline uses k = 10)</dd>
          <dt>nDCG@10</dt>
          <dd>
            rewards golden passages for ranking higher within the top 10 (relevance 1 or 0), divided by the best
            possible ordering
          </dd>
          <dt>MRR@10</dt>
          <dd>1 / the rank of the first golden passage if it is in the top 10, else 0, averaged over questions</dd>
          <dt>harness check</dt>
          <dd>
            On Phil Schmid's setup (random 90/10 rows, all 7,000 rows as the corpus) the base model scores nDCG@10{' '}
            {MEASURED.check} against his published {MEASURED.phil768[0]}. Across 20 split seeds the same setup ranges{' '}
            {MEASURED.checkRange[0]}–{MEASURED.checkRange[1]}, so the harness reproduces the reference within the
            noise of the split.
          </dd>
        </dl>
      </div>
    </Sec>
  );
}

// ── 13 · evidence ────────────────────────────────────────────────────────
function Evidence({ ov }: { ov: OverviewData }) {
  const c = ov.compare;
  if (!c)
    return (
      <Sec id="evidence" n={13} eyebrow="Statistics" title="Is the gain real?">
        <div className="empty">No tuned and base pair on test yet.</div>
      </Sec>
    );
  const disc = c.fixed + c.broken;
  return (
    <Sec id="evidence" n={13} eyebrow="Statistics" title="Is the gain real?">
      <div className="kpis">
        <div className="stat accent">
          <div className="n">{delta('recall@10', c.delta['recall@10'].value)}</div>
          <div className="l">
            Recall@10, 95% CI {interval('recall@10', c.delta['recall@10'].ci_low, c.delta['recall@10'].ci_high)}
          </div>
        </div>
        <div className="stat">
          <div className="n">
            {int(c.fixed)} / {int(c.broken)}
          </div>
          <div className="l">fixed / broken questions</div>
        </div>
        <div className="stat">
          <div className="n">{pText(c.p_exact)}</div>
          <div className="l">exact McNemar</div>
        </div>
        <div className="stat">
          <div className="n">{delta('ndcg@10', c.delta['ndcg@10'].value)}</div>
          <div className="l">nDCG@10</div>
        </div>
      </div>
      <div className="split">
        <p>
          <b>Paired, question by question.</b> The same {int(c.n)} test questions were scored by both models, so the
          right comparison is per question, not two averages. Only {int(disc)} questions changed outcome:{' '}
          {int(c.fixed)} fixed (tuned hits, base missed) and {int(c.broken)} broken. If fine-tuning did nothing, those{' '}
          {int(disc)} would split like coin flips. McNemar's exact test asks how likely a split of {int(c.fixed)} to{' '}
          {int(c.broken)} is under that assumption: {pText(c.p_exact)}.
        </p>
        <p>
          <b>Paired bootstrap.</b> Resample the {int(c.n)} questions with replacement 10,000 times and recompute the
          difference each time; the middle 95% is the interval. nDCG@10 ({delta('ndcg@10', c.delta['ndcg@10'].value)})
          and MRR@10 ({delta('mrr@10', c.delta['mrr@10'].value)}) rose too, so golden passages moved up throughout the
          top 10, not only across the rank-10 line. <Link to={overviewPath()}>Every number on the overview →</Link>
        </p>
      </div>
    </Sec>
  );
}

// ── 14 · alternatives ────────────────────────────────────────────────────
type Verdict = 'used' | 'next' | 'option' | 'no';
const VERDICT: Record<Verdict, string> = { used: 'ok', next: 'info', option: '', no: 'warn' };
type Alt = { name: string; what: ReactNode; trade: ReactNode; verdict: [Verdict, string] };

function alternatives(f: Facts, bm25: RunSummary | undefined): [string, Alt[]][] {
  const b = pct(r10(f.base, '768'));
  return [
    [
      "Don't fine-tune",
      [
        { name: 'The base model as is', what: 'bge-base with no training', trade: <>Free and no risk to other domains, but {b} Recall@10 against {pct(r10(f.tuned, '768'))}.</>, verdict: ['used', 'baseline row'] },
        { name: 'BM25 (keywords)', what: 'bm25s with stemming and English stopwords', trade: <>No model, explainable. Here it ties the base model ({pct(r10(bm25, 'lexical'))}) because questions reuse the filings' exact terms. It misses paraphrases.</>, verdict: ['used', 'baseline row (the floor)'] },
        { name: 'Hybrid BM25 + dense', what: 'fuse the two rankings, e.g. reciprocal rank fusion', trade: 'Often beats either alone, but it measures the fusion, not the fine-tune.', verdict: ['no', 'ruled out (D10)'] },
        { name: 'A bigger open model', what: 'bge-large-en-v1.5 (335M, 1,024 dims), e5-large-v2, gte-large', trade: 'Usually a few points better than base size. About 3× slower to encode, bigger vectors, still not finance-specific.', verdict: ['option', 'a fair extra baseline'] },
        { name: 'A hosted embedding API', what: 'OpenAI text-embedding-3-large, Cohere embed, Voyage voyage-finance-2', trade: 'Strong with no training. Per-token cost, filings leave the machine, the model can change underneath you, and it cannot be fine-tuned or pinned.', verdict: ['option', 'not measured; voyage-finance-2 is the domain rival'] },
        { name: 'A re-ranker on top', what: 'a cross-encoder (e.g. bge-reranker-base) re-scores the top 50', trade: "Reorders well, so nDCG and MRR rise. It can't rescue a passage the retriever didn't return, and it adds a model call per candidate per query.", verdict: ['next', 'complementary, later'] },
      ],
    ],
    [
      'A different loss',
      [
        { name: 'CachedMNRL (GradCache)', what: 'the same loss, with the batch processed in chunks and gradients cached', trade: 'Batches of 256–1,024 at constant memory: many more negatives per step, usually better. Slower per step.', verdict: ['next', 'next experiment'] },
        { name: 'MNRL + mined hard negatives', what: 'add, per question, passages the base model ranks high that are wrong', trade: 'The strongest known lever for this loss. A mined "negative" that also answers the question teaches the wrong thing, so it needs filtering.', verdict: ['next', 'next experiment'] },
        { name: 'GISTEmbedLoss', what: 'MNRL with a guide model that drops in-batch negatives it thinks are positives', trade: 'Handles near-duplicate false negatives that NO_DUPLICATES (exact text) misses. A second model forward per step.', verdict: ['option', 'if false negatives show up'] },
        { name: 'Symmetric MNRL', what: 'also ask each passage to pick its question', trade: 'Sometimes a small gain. Not in the reference recipe.', verdict: ['option', 'not tried'] },
        { name: 'Triplet or contrastive loss', what: '(anchor, positive, negative) with a margin, or labelled similar/dissimilar pairs', trade: 'Needs explicit negatives the dataset doesn’t have. One negative per example is a weaker signal than 31.', verdict: ['no', 'not suited'] },
        { name: 'CoSENT / AnglE', what: 'learn graded similarity scores', trade: 'Needs scores such as 0–5, not binary pairs.', verdict: ['no', 'not suited'] },
        { name: 'Plain MNRL, no Matryoshka', what: 'train the 768-dim embedding only', trade: <>Perhaps a touch better at 768, but the 64-dim prefix would stay as weak as the base model's ({pct(r10(f.base, '64'))}).</>, verdict: ['no', 'rejected: 64 dims is a goal'] },
      ],
    ],
    [
      'More or different data',
      [
        { name: 'LLM-written questions', what: 'generate extra questions for more 10-K passages', trade: 'More pairs and coverage. Costs money, needs quality checks, and raises licence questions.', verdict: ['option', 'if data limits the gain'] },
        { name: 'Domain pre-training (TSDAE, MLM)', what: 'unsupervised training on raw 10-K text', trade: 'For when no pairs exist. This dataset has pairs.', verdict: ['no', 'not needed'] },
        { name: 'A random row split', what: "Phil Schmid's 90/10 split of rows", trade: <>Leaks about {MEASURED.leak} test passages into training and inflates the score.</>, verdict: ['no', 'used only for the harness check'] },
      ],
    ],
    [
      'Train less of the model',
      [
        { name: 'LoRA or adapters', what: 'train small low-rank matrices, freeze the rest', trade: 'Saves memory on billion-parameter models. At 110M the whole model fits, and a full fine-tune is the usual best for small encoders.', verdict: ['no', 'not needed'] },
        { name: 'Freeze the lower layers', what: 'train only the top few layers', trade: 'Faster and forgets less, usually a little worse.', verdict: ['option', 'not tried'] },
        { name: 'A query-side adapter', what: 'train only a small transform on question vectors', trade: "The corpus never needs re-embedding, which matters for huge indexes. Much less capacity.", verdict: ['option', 'at scale'] },
      ],
    ],
    [
      'Choose or tune differently',
      [
        { name: 'Pick the epoch on test', what: 'report the best of the four test scores', trade: 'Overstates the result by the luck of the pick.', verdict: ['no', 'never'] },
        { name: 'Score inside the training loop', what: "Sentence Transformers' InformationRetrievalEvaluator each epoch", trade: 'Convenient, but a second scoring path to keep identical to the harness, and slower steps on MPS.', verdict: ['no', 'harness scores after training'] },
        { name: 'A hyperparameter sweep', what: 'try several learning rates, batch sizes and epoch counts', trade: <>Might add a point or two. Val has only {int(f.queries?.val)} questions, so gaps under about 1 point are noise, and a sweep would overfit val.</>, verdict: ['option', 'not done'] },
      ],
    ],
    [
      'Where to train',
      [
        { name: 'Docker on the Mac', what: 'run the train image locally', trade: "Docker on macOS can't reach the Apple GPU: CPU only, hours.", verdict: ['no', 'native instead'] },
        { name: 'A cloud GPU', what: 'EC2 g6 or a Modal A10G with the CUDA image', trade: 'Minutes instead of 22, but it costs money and setup.', verdict: ['used', 'fallback, image ready'] },
        { name: 'MLX', what: "Apple's array framework", trade: "Fast on Apple silicon, but Sentence Transformers' trainer doesn't run on it, so it would mean a rewrite.", verdict: ['no', 'not used'] },
      ],
    ],
  ];
}

function Alternatives({ f, bm25 }: { f: Facts; bm25: RunSummary | undefined }) {
  return (
    <Sec id="alternatives" n={14} eyebrow="Alternatives" title="What else could have been done, and why this approach">
      <div className="callout">
        <p>
          <b>Why this approach.</b> The project asks one question: does fine-tuning this model on these pairs put the
          golden passage in the top 10 more often? So the choices favour a fixed, checkable recipe and a fair comparison
          over the highest possible score.
        </p>
        <ul className="tight">
          <li>
            <b>One variable.</b> Same model, corpus, questions, prompt and harness before and after. Only the weights
            change.
          </li>
          <li>
            <b>Checkable.</b> It reproduces a published recipe, so both the harness and the training can be checked
            against a known result (<a href="#scoring">§12</a>).
          </li>
          <li>
            <b>Cheap and local.</b> $0, {f.secs != null ? seconds(f.secs) : DASH} on a laptop, and no data leaves the
            machine.
          </li>
          <li>
            <b>Two answers from one run.</b> The gain at 768 dims, and whether 64 Matryoshka dims can replace 768.
          </li>
          <li>
            <b>Honest.</b> Val makes every choice, test is scored once, and the statistics are paired.
          </li>
        </ul>
        <p className="small">Several alternatives below would probably score higher. The table says which, and what each costs.</p>
      </div>
      <div className="tablewrap">
        <table className="alts">
          <thead>
            <tr>
              <th>alternative</th>
              <th>what it is</th>
              <th>trade-off</th>
              <th>here</th>
            </tr>
          </thead>
          {alternatives(f, bm25).map(([group, rows]) => (
            <tbody key={group}>
              <tr className="group">
                <th colSpan={4} scope="colgroup">
                  {group}
                </th>
              </tr>
              {rows.map((a) => (
                <tr key={a.name}>
                  <td className="alt-name">{a.name}</td>
                  <td>{a.what}</td>
                  <td>{a.trade}</td>
                  <td>
                    <span className={`badge ${VERDICT[a.verdict[0]]}`}>{a.verdict[1]}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          ))}
        </table>
      </div>
    </Sec>
  );
}

// ── 15 · limits ──────────────────────────────────────────────────────────
function Limits() {
  const items: [string, string][] = [
    ['Out of domain is unmeasured', 'Fine-tuning may have cost general retrieval quality. The MTEB FiQA2018 check (M5) answers that and has not run.'],
    ['One training run', 'The bootstrap interval covers which questions were drawn, not training randomness. A second seed would show how much the result moves.'],
    ['A small val split', '693 questions make epoch-to-epoch gaps of a few questions indistinguishable from noise.'],
    ['Same kind of questions', 'Test questions are phrased like train questions, about the same filings (different passages). Real users ask differently.'],
    ['Easy negatives', 'Random batches rarely contain the passages that actually compete. The gain from hard negatives is untested.'],
    ['Licence unknown', 'The dataset card declares no licence, so the model is registered in MLflow only, not pushed to the Hub (D6).'],
  ];
  return (
    <Sec id="limits" n={15} eyebrow="Limits" title="What this result doesn't show yet">
      <div className="cards">
        {items.map(([k, v]) => (
          <div key={k} className="card">
            <b>{k}</b>
            <p className="small">{v}</p>
          </div>
        ))}
      </div>
    </Sec>
  );
}

// ── 16 · explaining it ───────────────────────────────────────────────────
function Explain({ f, ov }: { f: Facts; ov: OverviewData }) {
  const c = ov.compare;
  const b = pct(r10(f.base, '768'));
  const t = pct(r10(f.tuned, '768'));
  const faq: [string, ReactNode][] = [
    ['What does fine-tuning an embedding model actually do?', <>It adjusts the encoder's weights so a question and the passage that answers it end up close together, and other passages end up further away. It learns from (question, answer) pairs, with the rest of each batch as wrong answers.</>],
    ['Why did it help here?', <>The base model is general purpose and sees finance passages as similar to each other. Training on {int(f.nPairs)} in-domain pairs taught it the distinctions that matter here: company, year, line item. The live batch in §6 shows wrong passages pushed further away.</>],
    ['Did it overfit?', <>It started to: training loss kept falling while val Recall@10 flattened after epoch 1. That is why the epoch was chosen on val. Test was never used for a choice, and it still rose from {b} to {t}.</>],
    ['Is the gain real or luck?', c ? <>Paired over the same {int(c.n)} questions: {int(c.fixed)} fixed against {int(c.broken)} broken, exact McNemar {pText(c.p_exact)}; Recall@10 {delta('recall@10', c.delta['recall@10'].value)} with a 95% CI of {interval('recall@10', c.delta['recall@10'].ci_low, c.delta['recall@10'].ci_high)}.</> : 'Not measured yet.'],
    ['Why does tuned at 64 dims beat base at 768?', <>Matryoshka training puts the most useful information in the first dimensions and trains the 64-dim prefix directly. The base model spreads information across all 768, so cutting it loses some. Result: a 12× smaller index that retrieves better ({MEASURED.tuned64VsBase768.delta}).</>],
    ['Why not just use OpenAI embeddings?', <>Per-token cost, filings leaving the machine, and a model you can neither pin nor fine-tune. It would be a fair extra baseline, but the question here is the effect of fine-tuning.</>],
    ['How do you know the harness is right?', <>It reproduced the published reference (nDCG@10 {MEASURED.check} against {MEASURED.phil768[0]}, inside the {MEASURED.checkRange[0]}–{MEASURED.checkRange[1]} range across split seeds). Unit tests pin the metrics, and the same code scores both models.</>],
    ['What did it cost?', <>$0: {f.secs != null ? seconds(f.secs) : DASH} on an M1 Pro. The same image runs on a cloud GPU in minutes.</>],
    ['What would you do next?', <>Hard negatives mined with the base model, bigger batches with CachedMNRL, the FiQA2018 out-of-domain check, and a re-ranker on top.</>],
  ];
  return (
    <Sec id="explain" n={16} eyebrow="Explaining it" title="How to explain it, and the questions you'll get">
      <div className="callout">
        <p>
          <span className="tag">30 seconds</span>
          <br />
          "I fine-tuned an open embedding model, bge-base, on {int(f.nPairs)} finance question–passage pairs from 10-K
          filings, using a contrastive loss where every other passage in the batch is a wrong answer. I kept the
          epoch that did best on a validation split and scored a held-out test set once. Retrieval of the right passage
          in the top 10 went from {b} to {t}, a paired, significant gain. Because I trained with Matryoshka loss, the
          64-dimension version beats the original 768-dimension model, so the index is 12 times smaller."
        </p>
      </div>
      <div className="cards faq">
        {faq.map(([q, a]) => (
          <div key={q} className="card">
            <b>{q}</b>
            <p className="small">{a}</p>
          </div>
        ))}
      </div>
    </Sec>
  );
}

// ── 17 · glossary ────────────────────────────────────────────────────────
function Glossary() {
  const terms: [string, string][] = [
    ['embedding', 'A fixed-length list of numbers standing for a text; similar meanings get nearby vectors.'],
    ['cosine similarity', 'The cosine of the angle between two vectors, from −1 to 1; for length-1 vectors it equals the dot product.'],
    ['encoder', 'A transformer that reads the whole text at once and outputs one vector per token; here the [CLS] token’s vector is the embedding.'],
    ['in-batch negative', 'A passage in the same training batch that belongs to a different question, used as a wrong answer.'],
    ['hard negative', 'A wrong passage that the model already scores high, and so the most informative to train against.'],
    ['false negative', 'A "wrong" passage that actually answers the question, such as a duplicate; it teaches the wrong thing.'],
    ['temperature / scale', 'A multiplier on similarities before the softmax; scale 20 = temperature 0.05, a sharper distribution.'],
    ['Matryoshka', 'Training so that the first k numbers of an embedding work on their own, for several k.'],
    ['step · epoch', 'A step is one batch and one weight update; an epoch is one pass over all training pairs.'],
    ['checkpoint', 'The weights saved at a point in training; one per epoch here.'],
    ['warmup', 'Ramping the learning rate up from 0 over the first steps.'],
    ['Recall@10', 'The share of questions whose golden passage is in the top 10.'],
    ['nDCG@10', 'A top-10 score that gives more credit the higher the golden passage ranks.'],
    ['MRR@10', 'The mean of 1/rank of the first golden passage within the top 10.'],
    ['McNemar test', 'A paired test that uses only the questions where two systems disagree.'],
    ['bootstrap CI', 'An interval from re-sampling the questions many times and recomputing the difference.'],
  ];
  return (
    <Sec id="glossary" n={17} eyebrow="Glossary" title="The terms, in one line each">
      <dl className="kv glossary">
        {terms.map(([k, v]) => (
          <Fragment key={k}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </Fragment>
        ))}
      </dl>
    </Sec>
  );
}

// ── 18 · reproduce ───────────────────────────────────────────────────────
function Reproduce({ trainId }: { trainId: string | null }) {
  return (
    <Sec id="reproduce" n={18} eyebrow="Reproduce it" title="Run it yourself">
      <div>
        <pre>
          <code>{`make setup            # uv sync + npm ci
make platform-up      # the central MLflow on :5000
make split            # split v1 (frozen; already committed)
make baseline check   # BM25, bge-small, bge-base; the harness check
make train            # ~22 min on an M1 Pro; epoch chosen on val
make eval TRAIN_RUN=<train run id>   # test, once
make compare A=<tuned eval> B=<base baseline>`}</code>
        </pre>
        <dl className="kv">
          <dt>recipe</dt>
          <dd>
            <Code>configs/train_bge_base.yaml</Code>
          </dd>
          <dt>training</dt>
          <dd>
            <Code>src/finance_rag_sft/train.py</Code>
          </dd>
          <dt>harness</dt>
          <dd>
            <Code>src/finance_rag_sft/harness.py</Code>
          </dd>
          <dt>statistics</dt>
          <dd>
            <Code>src/finance_rag_sft/compare.py</Code>
          </dd>
          <dt>this page's example</dt>
          <dd>
            <Code>src/finance_rag_sft/explain.py</Code> · <Code>GET /api/method/batch</Code>
          </dd>
          <dt>the run</dt>
          <dd>{trainId ? <Link to={runPath(trainId)} className="mono">{trainId}</Link> : DASH}</dd>
          <dt>try the models</dt>
          <dd>
            <Link to={searchPath()}>live search, base against tuned →</Link>
          </dd>
        </dl>
      </div>
    </Sec>
  );
}
