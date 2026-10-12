import { describe, expect, it } from 'vitest';
import {
  dataPassagePath,
  dataPath,
  dataQuestionPath,
  intLens,
  patchLens,
  pickLens,
  runPath,
  runQuestionPath,
  runsPath,
  search,
  searchPath,
} from './url';

const RUN = '20261012-170000-baseline-bge-base';

describe('search (the lens as a query string)', () => {
  it('skips empty values and keeps key order', () => {
    expect(search({ vs: RUN, dim: '64', q: '', filter: null, offset: undefined })).toBe(`?vs=${RUN}&dim=64`);
    expect(search({})).toBe('');
    expect(search()).toBe('');
  });
  it('keeps / : , readable and escapes the rest', () => {
    expect(search({ r: `bm25,tuned:${RUN}` })).toBe(`?r=bm25,tuned:${RUN}`);
    expect(search({ q: "AT&T's 2023?" })).toBe("?q=AT%26T's%202023%3F");
  });
  it('prints numbers, including 0', () => {
    expect(search({ offset: 0 })).toBe('?offset=0');
    expect(search({ offset: 100 })).toBe('?offset=100');
  });
});

describe('patchLens', () => {
  it('sets, replaces and removes keys', () => {
    const cur = new URLSearchParams(`vs=${RUN}&filter=fixed&offset=100`);
    expect(patchLens(cur, { filter: 'miss', offset: null })).toBe(`?vs=${RUN}&filter=miss`);
    expect(patchLens(cur, { vs: '', filter: null, offset: null })).toBe('');
    expect(patchLens(new URLSearchParams(), { q: 'capex' })).toBe('?q=capex');
  });
});

describe('pickLens and intLens', () => {
  const cur = new URLSearchParams('dim=64&vs=b&offset=200&q=x');
  it('carries only the named keys', () => {
    expect(pickLens(cur, ['dim', 'vs', 'filter'])).toEqual({ dim: '64', vs: 'b', filter: null });
  });
  it('reads a whole number ≥ 0, else the fallback', () => {
    expect(intLens(cur, 'offset')).toBe(200);
    expect(intLens(cur, 'missing', 7)).toBe(7);
    expect(intLens(new URLSearchParams('offset=-5'), 'offset')).toBe(0);
    expect(intLens(new URLSearchParams('offset=1.5'), 'offset')).toBe(0);
    expect(intLens(new URLSearchParams('offset=abc'), 'offset', 3)).toBe(3);
  });
});

describe('paths: the path names the subject, the query holds the lens', () => {
  it('builds every page address', () => {
    expect(dataPath()).toBe('/data');
    expect(dataPath({ split: 'test' })).toBe('/data?split=test');
    expect(dataQuestionPath('q3f9a0c1b2d4e', { split: 'test' })).toBe('/data/q/q3f9a0c1b2d4e?split=test');
    expect(dataPassagePath('p5a6b7c8d9e0f')).toBe('/data/p/p5a6b7c8d9e0f');
    expect(runsPath()).toBe('/runs');
    expect(runPath(RUN, { vs: 'other', dim: '64' })).toBe(`/runs/${RUN}?vs=other&dim=64`);
    expect(runQuestionPath(RUN, 'q3f9a0c1b2d4e', { vs: 'other' })).toBe(`/runs/${RUN}/q3f9a0c1b2d4e?vs=other`);
    expect(searchPath({ q: 'capex', r: 'bm25,bge-base' })).toBe('/search?q=capex&r=bm25,bge-base');
  });
  it('escapes a path segment once, keeping : readable and / escaped', () => {
    expect(runPath('tuned:abc')).toBe('/runs/tuned:abc');
    expect(runPath('a/b')).toBe('/runs/a%2Fb');
    expect(decodeURIComponent(runPath('a b'))).toBe('/runs/a b');
  });
});
