/** A fake `finrag serve`: answers the viewer's fetches from the fixtures, by method and path. */
import { vi } from 'vitest';
import * as fx from './fixtures';

export type Reply = unknown | { status: number; body: unknown };
export type Overrides = Record<string, Reply | ((url: URL, body: unknown) => Reply)>;

const isStatus = (r: unknown): r is { status: number; body: unknown } =>
  typeof r === 'object' && r !== null && 'status' in r && 'body' in r && Object.keys(r).length === 2;

/** The default answer for a request, per docs/api.md; undefined means 404. */
function route(method: string, url: URL): unknown {
  const p = url.pathname.split('/').filter(Boolean).map(decodeURIComponent);
  const vs = url.searchParams.get('vs');
  if (method === 'POST') return url.pathname === '/api/search' ? fx.searchResult : undefined;
  if (url.pathname === '/healthz') return fx.health;
  if (p[0] !== 'api') return undefined;
  switch (p[1]) {
    case 'overview':
      return fx.overview;
    case 'retrievers':
      return fx.retrievers;
    case 'compare':
      return fx.compare;
    case 'method': {
      if (p[2] !== 'batch') return undefined;
      const sp = url.searchParams;
      return { ...fx.methodBatch, split: sp.get('split') ?? 'train', dim: sp.get('dim') ?? '768', seed: Number(sp.get('seed') ?? 0) };
    }
    case 'split':
      if (p.length === 2) return fx.split;
      if (p[2] === 'queries') return p[3] ? (p[3] === fx.QID ? fx.queryDetail : undefined) : fx.queryList;
      if (p[2] === 'passages') return p[3] ? (p[3] === fx.PID ? fx.passageDetail : undefined) : fx.passageList;
      return undefined;
    case 'runs':
      if (p.length === 2) return fx.runs;
      if (p.length === 3) return fx.runsById[p[2]];
      if (p[3] === 'queries' && p.length === 4) return vs ? fx.runQueriesVs : fx.runQueries;
      if (p[3] === 'queries' && p[4] === fx.QID) return vs ? fx.runQuestionVs : fx.runQuestion;
      return undefined;
    default:
      return undefined;
  }
}

/** Stub global fetch. Overrides are keyed `"GET /api/runs"` (path only, no query string). */
export function mockApi(overrides: Overrides = {}) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), 'http://viewer.test');
    const method = (init?.method ?? 'GET').toUpperCase();
    const key = `${method} ${url.pathname}`;
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    let reply: Reply;
    if (key in overrides) {
      const o = overrides[key];
      reply = typeof o === 'function' ? (o as (u: URL, b: unknown) => Reply)(url, body) : o;
    } else {
      const d = route(method, url);
      reply = d === undefined ? { status: 404, body: { detail: `Not found: ${url.pathname}` } } : d;
    }
    const [status, payload] = isStatus(reply) ? [reply.status, reply.body] : [200, reply];
    return {
      ok: status >= 200 && status < 300,
      status,
      json: async () => structuredClone(payload),
    } as unknown as Response;
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

/** Every URL fetched so far, as strings (path + query). */
export function fetched(fetchMock: ReturnType<typeof mockApi>): string[] {
  return fetchMock.mock.calls.map(([u]) => String(u));
}
