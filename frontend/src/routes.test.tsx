/**
 * Every route renders against a fake API (src/test/fixtures.ts, shaped per docs/api.md):
 * the populated state, the empty state before any run exists, the `?vs=` compare lens, and errors.
 * Then the URL grammar: every address the viewer builds lands on the page it names.
 */
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { RouterProvider, createMemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it } from 'vitest';
import { vi } from 'vitest';
import {
  dataPassagePath,
  dataPath,
  dataQuestionPath,
  runPath,
  runQuestionPath,
  runsPath,
  searchPath,
} from './lib/url';
import { routes } from './routes';
import * as fx from './test/fixtures';
import { type Overrides, fetched, mockApi } from './test/mockApi';

afterEach(() => {
  vi.unstubAllGlobals();
});

function renderAt(url: string, overrides: Overrides = {}) {
  const fetchMock = mockApi(overrides);
  const router = createMemoryRouter(routes, { initialEntries: [url] });
  render(<RouterProvider router={router} />);
  return { router, fetchMock };
}

const h1 = (name: RegExp | string) => screen.findByRole('heading', { level: 1, name });

describe('shell', () => {
  it('shows the brand, the four tabs and the health pill', async () => {
    renderAt('/');
    expect(await screen.findByText(`api ok · ${fx.SHA}`)).toBeTruthy();
    expect(screen.getByText(/^mlflow /).textContent).toBe('mlflow ● up');
    const nav = screen.getByRole('navigation', { name: 'Pages' });
    expect(within(nav).getAllByRole('link').map((a) => a.textContent)).toEqual(['Overview', 'Data', 'Runs', 'Search']);
    expect(screen.getByRole('link', { name: 'finance-rag-sft' })).toBeTruthy();
    expect(screen.getByRole('contentinfo').textContent).toContain(fx.SHA);
  });

  it('labels a down API and a down MLflow in words, not colour alone', async () => {
    renderAt('/', { 'GET /healthz': { status: 503, body: { detail: 'unavailable' } } });
    expect(await screen.findByText('✕ api down')).toBeTruthy();
    renderAt('/runs', { 'GET /healthz': { ...fx.health, mlflow: 'down' } });
    expect(await screen.findByText(/mlflow ○ down/)).toBeTruthy();
  });

  it('an unknown address says so', async () => {
    renderAt('/nowhere');
    expect(await h1('No page here')).toBeTruthy();
  });
});

describe('/ overview', () => {
  it('leads with Recall@10 for base and tuned at 768 and 64 dims', async () => {
    renderAt('/');
    expect(await h1(/golden passage in the top 10/)).toBeTruthy();
    const head = screen.getByRole('group', { name: 'Recall@10, base and tuned' });
    for (const v of ['81.2%', '73.4%', '85.3%', '80.7%']) expect(within(head).getByText(v)).toBeTruthy();
    expect(screen.getByText('Recall@10, tuned · 768 dims')).toBeTruthy();
    expect(screen.getByText('Recall@10, base · 64 dims')).toBeTruthy();
  });

  it('summarises the compare: fixed, broken, p and the deltas', async () => {
    renderAt('/');
    expect(await screen.findByText('48')).toBeTruthy();
    expect(screen.getByText('19')).toBeTruthy();
    expect(screen.getByText('< 0.001')).toBeTruthy();
    expect(screen.getByText('+4.1 pp')).toBeTruthy();
    expect(screen.getByText('[+1.8, +6.5] pp')).toBeTruthy();
    const link = screen.getByRole('link', { name: /Open every question/ });
    expect(link.getAttribute('href')).toBe(runPath(fx.RUN_TUNED, { dim: '768', vs: fx.RUN_BASE }));
  });

  it('lists the baselines, milestones and split counts', async () => {
    renderAt('/');
    await h1(/golden passage/);
    expect(screen.getByRole('heading', { name: 'Every baseline, newest first' })).toBeTruthy();
    expect(screen.getAllByText('BM25').length).toBeGreaterThan(0);
    expect(screen.getByText('bge-small (base)')).toBeTruthy();
    expect(screen.getByText('M3 · Fine-tune bge-base')).toBeTruthy();
    expect(screen.getAllByText('✓ done')).toHaveLength(3);
    expect(screen.getByText('○ pending')).toBeTruthy();
    expect(screen.getByText('6,757')).toBeTruthy();
    expect(screen.getByText('700 questions')).toBeTruthy();
  });

  it('before any run: empty states name the milestone that fills them', async () => {
    renderAt('/', { 'GET /api/overview': fx.overviewEmpty });
    expect(await h1(/golden passage/)).toBeTruthy();
    expect(screen.getAllByText(/No tuned model yet — M3/).length).toBe(3); // two headline cards + the compare
    expect(screen.getAllByText(/No base run yet — M2/).length).toBe(2);
    expect(screen.getByText(/No baseline runs yet/)).toBeTruthy();
    expect(screen.getByText(/No split yet — M1/)).toBeTruthy();
  });
});

describe('/data explorer', () => {
  it('lists questions with split badge and golden count, and the split stats', async () => {
    renderAt(dataPath());
    expect(await h1(/questions and passages/)).toBeTruthy();
    expect(await screen.findByText(fx.Q_TEXT)).toBeTruthy();
    expect(screen.getByText('2 golden passages')).toBeTruthy();
    expect(screen.getByText('1 golden')).toBeTruthy();
    expect(screen.getAllByText('✓ ok').length).toBe(2); // the KPI and the leak-check card
    expect(screen.getByText('Passage length')).toBeTruthy();
    expect(screen.getByText('Question length')).toBeTruthy();
    expect(screen.getByText(/88 questions with more than one golden passage/)).toBeTruthy();
    expect(screen.getAllByRole('img', { name: /length: median/ })).toHaveLength(2);
  });

  it('the split and search lens reach the API, and the split is a chip', async () => {
    const { fetchMock } = renderAt(dataPath({ split: 'test', q: 'dividends' }));
    await screen.findByText(fx.Q_TEXT);
    expect(fetched(fetchMock)).toContain('/api/split/queries?split=test&q=dividends&offset=0&limit=50');
    expect(screen.getByRole('button', { name: 'test', pressed: true })).toBeTruthy();
  });

  it('typing in the search box rewrites the lens and refetches', async () => {
    const { fetchMock, router } = renderAt(dataPath());
    await screen.findByText(fx.Q_TEXT);
    fireEvent.change(screen.getByRole('searchbox'), { target: { value: 'capex' } });
    await waitFor(() => expect(router.state.location.search).toBe('?q=capex'));
    await waitFor(() => expect(fetched(fetchMock).some((u) => u.includes('q=capex'))).toBe(true));
  });

  it('a question opens inside the list: golden passage, its other questions, runs that scored it', async () => {
    renderAt(dataQuestionPath(fx.QID));
    const detail = await screen.findByRole('article', { name: `question ${fx.QID}` });
    expect(await within(detail).findByText(fx.P_TEXT)).toBeTruthy();
    expect(within(detail).getByText('1 other question points here')).toBeTruthy();
    expect(await within(detail).findByText(fx.Q3_TEXT)).toBeTruthy();
    const run = within(detail).getByRole('link', { name: 'bge-base (base)' });
    expect(run.getAttribute('href')).toBe(runQuestionPath(fx.RUN_BASE, fx.QID, { dim: '768' }));
    expect(within(detail).getByText('14')).toBeTruthy();
    // the detail renders under its row, inside the list
    expect(detail.closest('li')?.className).toContain('open');
  });

  it('a passage opens in the passages view with every question pointing at it', async () => {
    renderAt(dataPassagePath(fx.PID));
    const detail = await screen.findByRole('article', { name: `passage ${fx.PID}` });
    expect(within(detail).getByText('2 questions point here')).toBeTruthy();
    expect(within(detail).getByText(fx.Q3_TEXT)).toBeTruthy();
    expect(screen.getByRole('button', { name: 'passages', pressed: true })).toBeTruthy();
  });

  it('a question not on the current page opens above the list', async () => {
    renderAt(dataQuestionPath(fx.QID), { 'GET /api/split/queries': { total: 0, items: [] } });
    const detail = await screen.findByRole('article', { name: `question ${fx.QID}` });
    expect(within(detail).getByRole('heading', { name: fx.Q_TEXT })).toBeTruthy();
    expect(screen.getByText(/No questions/)).toBeTruthy();
  });
});

describe('/runs', () => {
  it('tables every run with kind badge, dims, metrics and MLflow link', async () => {
    renderAt(runsPath());
    expect(await h1(/Every run/)).toBeTruthy();
    const table = await screen.findByRole('table');
    expect(within(table).getByText('bge-base tuned (epoch 2)')).toBeTruthy();
    expect(within(table).getAllByText('baseline')).toHaveLength(3);
    expect(within(table).getByText('train')).toBeTruthy();
    expect(within(table).getByText('lexical')).toBeTruthy();
    expect(within(table).getByText('85.3%')).toBeTruthy();
    expect(within(table).getByText('0.712')).toBeTruthy();
    expect(within(table).getAllByRole('link', { name: 'MLflow ↗' })).toHaveLength(3);
  });

  it('with no runs, says how they appear', async () => {
    renderAt(runsPath(), { 'GET /api/runs': [] });
    expect(await screen.findByText(/No runs yet/)).toBeTruthy();
  });
});

describe('/runs/:runId', () => {
  it('shows metrics per dim, setup and every question', async () => {
    const { fetchMock } = renderAt(runPath(fx.RUN_TUNED));
    expect(await h1('bge-base tuned (epoch 2)')).toBeTruthy();
    expect(screen.getByRole('heading', { name: 'Every metric, per dimension size' })).toBeTruthy();
    expect(screen.getByText('85.3%')).toBeTruthy();
    expect(screen.getByText('80.7%')).toBeTruthy();
    expect(await screen.findByRole('link', { name: 'bge-base fine-tune, 3 epochs' })).toBeTruthy();
    expect(await screen.findByText(fx.Q2_TEXT)).toBeTruthy();
    expect(screen.getByText('○ miss')).toBeTruthy();
    expect(fetched(fetchMock)).toContain(`/api/runs/${fx.RUN_TUNED}/queries?dim=768&filter=all&offset=0&limit=100`);
    // no compare yet: fixed/broken filters need ?vs=
    expect(screen.queryByRole('button', { name: 'fixed' })).toBeNull();
    const pick = (await screen.findByRole('combobox', { name: 'compare with…' })) as HTMLSelectElement;
    const labels = [...pick.options].map((o) => o.textContent ?? '');
    expect(labels.some((l) => l.startsWith('bge-base (base)'))).toBe(true);
    expect(labels.some((l) => l.includes('BM25') && l.includes('(no 768 dims)'))).toBe(true);
  });

  it('picking a run to compare with writes ?vs= into the address', async () => {
    const { router } = renderAt(runPath(fx.RUN_TUNED));
    const pick = (await screen.findByRole('combobox', { name: 'compare with…' })) as HTMLSelectElement;
    fireEvent.change(pick, { target: { value: fx.RUN_BASE } });
    await waitFor(() => expect(router.state.location.search).toBe(`?vs=${fx.RUN_BASE}`));
    expect(await screen.findByText('[+1.8, +6.5] pp')).toBeTruthy();
  });

  it('?vs= adds the compare panel, the fixed/broken filters and the change column', async () => {
    const { fetchMock } = renderAt(runPath(fx.RUN_TUNED, { vs: fx.RUN_BASE }));
    expect(await screen.findByRole('heading', { name: 'This run against another' })).toBeTruthy();
    expect(await screen.findByText('48')).toBeTruthy();
    expect(screen.getByText('< 0.001')).toBeTruthy();
    expect(screen.getByText('+0.061')).toBeTruthy();
    expect(screen.getByText(/up 10\+/)).toBeTruthy();
    expect(fetched(fetchMock)).toContain(`/api/compare?a=${fx.RUN_TUNED}&b=${fx.RUN_BASE}&dim=768`);
    expect(screen.getByRole('button', { name: 'fixed' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'broken' })).toBeTruthy();
    await screen.findByText(fx.Q2_TEXT);
    const table = screen.getAllByRole('table').at(-1)!;
    expect(within(table).getByText('vs rank')).toBeTruthy();
    expect(within(table).getByText('▲ fixed')).toBeTruthy();
    expect(within(table).getByText('▼ broken')).toBeTruthy();
    const q = within(table).getByRole('link', { name: fx.Q_TEXT });
    expect(q.getAttribute('href')).toBe(runQuestionPath(fx.RUN_TUNED, fx.QID, { vs: fx.RUN_BASE, dim: '768' }));
  });

  it('the filter, dim, search and page live in the lens and reach the API', async () => {
    const { fetchMock } = renderAt(
      runPath(fx.RUN_TUNED, { vs: fx.RUN_BASE, dim: '64', filter: 'fixed', q: 'AT&T', offset: 100 }),
    );
    await screen.findByText(fx.Q_TEXT);
    expect(fetched(fetchMock)).toContain(
      `/api/runs/${fx.RUN_TUNED}/queries?dim=64&filter=fixed&q=AT%26T&offset=100&limit=100&vs=${fx.RUN_BASE}`,
    );
    expect(fetched(fetchMock)).toContain(`/api/compare?a=${fx.RUN_TUNED}&b=${fx.RUN_BASE}&dim=64`);
    expect(screen.getByRole('button', { name: 'fixed', pressed: true })).toBeTruthy();
    expect(screen.getByRole('button', { name: '64', pressed: true })).toBeTruthy();
    expect(screen.getByText(/101–200 of 700/)).toBeTruthy();
  });

  it('a compare the API refuses shows its reason with an error label', async () => {
    renderAt(runPath(fx.RUN_TUNED, { vs: fx.RUN_BM25 }), {
      'GET /api/compare': { status: 400, body: { detail: 'runs do not share dim 768' } },
    });
    expect(await screen.findByText('runs do not share dim 768')).toBeTruthy();
    expect(screen.getAllByText('✕ error').length).toBeGreaterThan(0);
  });

  it('a train run shows each epoch on val and the chosen one', async () => {
    renderAt(runPath(fx.RUN_TRAIN));
    expect(await h1('bge-base fine-tune, 3 epochs')).toBeTruthy();
    expect(screen.getByRole('heading', { name: 'Epoch 2 chosen, by val recall@10 at 768' })).toBeTruthy();
    expect(screen.getByText('✓ chosen').closest('tr')?.className).toContain('chosen');
    expect(screen.getByText('86.1%')).toBeTruthy();
    expect(screen.getByText('5,512 (question, golden passage) pairs')).toBeTruthy();
    expect(await screen.findByRole('link', { name: 'bge-base tuned (epoch 2)' })).toBeTruthy();
    // a train run has no per-question table and no compare
    expect(screen.queryByRole('heading', { name: /golden rank/ })).toBeNull();
  });

  it('a check run shows the reference against what we got', async () => {
    renderAt(runPath(fx.RUN_CHECK));
    expect(await screen.findByRole('heading', { name: 'Reference against what we got' })).toBeTruthy();
    expect(screen.getByText('86.0%')).toBeTruthy();
    expect(screen.getAllByText('86.1%').length).toBeGreaterThan(0);
    expect(screen.getByText('Within 0.5 pp of the published number.')).toBeTruthy();
  });

  it('an unknown run shows the API detail with an error label', async () => {
    renderAt(runPath('nope'));
    expect(await screen.findByText('Not found: /api/runs/nope')).toBeTruthy();
    expect(screen.getByText('✕ error')).toBeTruthy();
  });
});

describe('/runs/:runId/:qid', () => {
  it('shows the question, its golden passage and the top 10 with the golden row marked', async () => {
    renderAt(runQuestionPath(fx.RUN_TUNED, fx.QID, { dim: '768' }));
    expect(await h1(fx.Q_TEXT)).toBeTruthy();
    const golden = screen.getAllByText('★ golden');
    expect(golden).toHaveLength(1);
    expect(golden[0].closest('li')?.className).toBe('gold');
    expect(golden[0].closest('li')?.querySelector('.r')?.textContent).toBe('1');
    expect(screen.getAllByRole('listitem').filter((li) => li.closest('ol.rk'))).toHaveLength(10);
  });

  it('?vs= puts both top 10s side by side, with the gap down to the golden rank', async () => {
    renderAt(runQuestionPath(fx.RUN_TUNED, fx.QID, { dim: '768', vs: fx.RUN_BASE }));
    expect(await h1(fx.Q_TEXT)).toBeTruthy();
    const cols = screen.getAllByRole('list').filter((l) => l.matches('ol.rk'));
    expect(cols).toHaveLength(2);
    // left: the other run (base), golden at 14 below a dashed gap; right: this run (tuned), golden at 1
    expect(within(cols[0]).getByText('ranks 11–13 · golden passage not in the top 10')).toBeTruthy();
    expect(within(cols[0]).getByText('14')).toBeTruthy();
    expect(within(cols[1]).queryByText(/golden passage not in the top 10/)).toBeNull();
    expect(screen.getAllByText('▲ fixed').length).toBeGreaterThan(0);
    expect(screen.getByText('miss → hit at k=10')).toBeTruthy();
    expect(screen.getByRole('link', { name: 'bge-base (base)' })).toBeTruthy();
  });

  it('a bare address waits for the run to learn its first dim', async () => {
    const { fetchMock } = renderAt(runQuestionPath(fx.RUN_TUNED, fx.QID));
    expect(await h1(fx.Q_TEXT)).toBeTruthy();
    expect(fetched(fetchMock)).toContain(`/api/runs/${fx.RUN_TUNED}/queries/${fx.QID}?dim=768`);
  });
});

describe('/search', () => {
  it('lists the retrievers as chips, cached ones on by default, and the examples', async () => {
    renderAt(searchPath());
    expect(await h1(/Ask the corpus/)).toBeTruthy();
    expect(await screen.findByRole('button', { name: 'BM25', pressed: true })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'bge-small · not cached', pressed: false })).toBeTruthy();
    expect(screen.getByRole('button', { name: fx.Q_TEXT })).toBeTruthy();
    expect(screen.getByText(/BM25 ignores it/)).toBeTruthy();
  });

  it('a query in the address runs, one column per retriever, golden passages marked', async () => {
    const { fetchMock } = renderAt(searchPath({ q: fx.Q_TEXT, dim: '64' }));
    const results = await screen.findByRole('region', { name: 'results' });
    const post = fetchMock.mock.calls.find(([, init]) => init?.method === 'POST');
    expect(JSON.parse(String(post?.[1]?.body))).toEqual({
      query: fx.Q_TEXT,
      retrievers: ['bm25', 'bge-base', `tuned:${fx.RUN_TRAIN}`],
      dim: '64',
      k: 10,
    });
    expect(within(results).getAllByRole('heading', { level: 4 })).toHaveLength(3);
    expect(within(results).getAllByText('★ golden')).toHaveLength(2); // bm25 rank 2 and tuned rank 1; base misses
    const link = within(results).getByRole('link', { name: fx.QID });
    expect(link.getAttribute('href')).toBe(dataQuestionPath(fx.QID));
    // the API answered 768 for a 64 ask on two dense columns: flagged in words
    expect(within(results).getAllByText('asked 64, got 768')).toHaveLength(2);
  });

  it('an example fills the lens; toggling a chip rewrites ?r=', async () => {
    const { router } = renderAt(searchPath());
    fireEvent.click(await screen.findByRole('button', { name: fx.Q2_TEXT }));
    await waitFor(() => expect(new URLSearchParams(router.state.location.search).get('q')).toBe(fx.Q2_TEXT));
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'BM25', pressed: true }));
    });
    await waitFor(() =>
      expect(new URLSearchParams(router.state.location.search).get('r')).toBe(`bge-base,tuned:${fx.RUN_TRAIN}`),
    );
  });

  it('a failed search shows the API detail with an error label', async () => {
    renderAt(searchPath({ q: 'anything' }), {
      'POST /api/search': { status: 422, body: { detail: 'unknown retriever' } },
    });
    expect(await screen.findByText('unknown retriever')).toBeTruthy();
    expect(screen.getByText('✕ error')).toBeTruthy();
  });
});

describe('the URL grammar: every address the viewer builds lands on the page it names', () => {
  async function land(url: string) {
    mockApi();
    const router = createMemoryRouter(routes, { initialEntries: [url] });
    await waitFor(() => expect(router.state.initialized).toBe(true));
    const ids = router.state.matches.map((m) => m.route.id);
    const path = decodeURIComponent(`${router.state.location.pathname}${router.state.location.search}`);
    router.dispose();
    return { ids, path };
  }

  it.each([
    ['/', ['shell', 'overview']],
    [dataPath({ split: 'test' }), ['shell', 'data']],
    [dataQuestionPath(fx.QID, { split: 'test' }), ['shell', 'data', 'data-question']],
    [dataPassagePath(fx.PID), ['shell', 'data', 'data-passage']],
    [runsPath(), ['shell', 'runs']],
    [runPath(fx.RUN_TUNED, { vs: fx.RUN_BASE, dim: '64' }), ['shell', 'run']],
    [runQuestionPath(fx.RUN_TUNED, fx.QID, { vs: fx.RUN_BASE }), ['shell', 'run-question']],
    [runQuestionPath(`tuned:${fx.RUN_TRAIN}`, fx.QID), ['shell', 'run-question']],
    [searchPath({ q: 'What is EBITDA?', r: 'bm25,bge-base' }), ['shell', 'search']],
    ['/runs/a/b/c', ['shell', 'missing']],
  ])('%s', async (url, ids) => {
    const got = await land(url);
    expect(got.ids).toEqual(ids);
  });

  it('a detail renders inside its list: the data list stays mounted', async () => {
    const got = await land(dataQuestionPath(fx.QID));
    expect(got.ids).toEqual(['shell', 'data', 'data-question']);
  });

  it('a lens survives the round trip readable', async () => {
    const got = await land(runPath(fx.RUN_TUNED, { vs: fx.RUN_BASE, filter: 'fixed' }));
    expect(got.path).toBe(`/runs/${fx.RUN_TUNED}?vs=${fx.RUN_BASE}&filter=fixed`);
  });
});
