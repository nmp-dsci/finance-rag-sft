/**
 * One address per thing. Every viewer URL is built here, from tau2-loop's three rules:
 *
 * 1. One id, the same string everywhere: a run id (`20261012-170000-baseline-bge-base`), a
 *    question id (`q` + 12 hex), a passage id (`p` + 12 hex). The API, the tables and the
 *    URLs all print the same string.
 * 2. The path names the subject; the query string holds the lens. If dropping a value leaves
 *    nothing to show, it is the subject (a run, a question, a passage) and goes in the path.
 *    If dropping it falls back to a default (a dimension size, a filter, a search, a page,
 *    a compared run) it is a lens. So a comparison is `?vs=<run>` on a page, not a tab.
 * 3. Picking from a list opens the detail in place: `/data/q/:qid` renders inside the Data
 *    list, so the address changes and the list stays.
 */
import { useLocation, useNavigate, useSearchParams } from 'react-router-dom';

export type Lens = Record<string, string | number | null | undefined>;

// ── paths ─────────────────────────────────────────────────────────────────
export const overviewPath = (): string => '/';
export const dataPath = (lens?: Lens): string => `/data${search(lens)}`;
export const dataQuestionPath = (qid: string, lens?: Lens): string => `/data/q/${enc(qid)}${search(lens)}`;
export const dataPassagePath = (pid: string, lens?: Lens): string => `/data/p/${enc(pid)}${search(lens)}`;
export const runsPath = (lens?: Lens): string => `/runs${search(lens)}`;
export const runPath = (runId: string, lens?: Lens): string => `/runs/${enc(runId)}${search(lens)}`;
export const runQuestionPath = (runId: string, qid: string, lens?: Lens): string =>
  `/runs/${enc(runId)}/${enc(qid)}${search(lens)}`;
export const searchPath = (lens?: Lens): string => `/search${search(lens)}`;

// ── encoding ──────────────────────────────────────────────────────────────
/** Escape one path segment, leaving `:` and `,` readable (both legal in a path, RFC 3986 §3.3). */
function enc(v: string): string {
  return encodeURIComponent(v).replace(/%3A/gi, ':').replace(/%2C/gi, ',');
}

/** Escape a query value, leaving `/`, `:` and `,` readable (legal in a query, RFC 3986 §3.4),
 *  so a lens reads `?vs=20261012-170000-baseline-bge-base&r=bm25,tuned:2026…`. */
function encQuery(v: string): string {
  return encodeURIComponent(v).replace(/%2F/gi, '/').replace(/%3A/gi, ':').replace(/%2C/gi, ',');
}

// ── the lens ──────────────────────────────────────────────────────────────
/** `?a=1&b=x`, skipping empty values; '' when nothing is set. Keys keep their given order. */
export function search(lens?: Lens): string {
  const parts = Object.entries(lens ?? {})
    .filter((e): e is [string, string | number] => e[1] != null && e[1] !== '')
    .map(([k, v]) => `${encQuery(k)}=${encQuery(String(v))}`);
  return parts.length ? `?${parts.join('&')}` : '';
}

/** The current lens with `patch` applied (null or '' removes a key), as a `?…` string. */
export function patchLens(current: URLSearchParams, patch: Lens): string {
  const next: Lens = Object.fromEntries(current.entries());
  for (const [k, v] of Object.entries(patch)) next[k] = v;
  return search(next);
}

/** Only the named keys of a lens, for carrying it to another page. */
export function pickLens(current: URLSearchParams, keys: string[]): Lens {
  return Object.fromEntries(keys.map((k) => [k, current.get(k)]));
}

/** A whole-number lens value, falling back when absent or not a number ≥ 0. */
export function intLens(current: URLSearchParams, key: string, fallback = 0): number {
  const n = Number(current.get(key));
  return current.has(key) && Number.isInteger(n) && n >= 0 ? n : fallback;
}

/** The page's lens and a setter that rewrites it in place (replace, not push). */
export function useLens(): [URLSearchParams, (patch: Lens) => void] {
  const [sp] = useSearchParams();
  const nav = useNavigate();
  const { pathname, hash } = useLocation();
  const set = (patch: Lens) => nav({ pathname, search: patchLens(sp, patch), hash }, { replace: true });
  return [sp, set];
}
