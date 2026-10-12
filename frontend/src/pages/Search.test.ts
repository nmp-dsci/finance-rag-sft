import { describe, expect, it } from 'vitest';
import type { RetrieverOption } from '../lib/api';
import { pickRetrievers } from './Search';

const opts: RetrieverOption[] = [
  { id: 'bm25', label: 'BM25', dims: ['lexical'], ready: true },
  { id: 'bge-small', label: 'bge-small', dims: ['384', '64'], ready: false },
  { id: 'bge-base', label: 'bge-base', dims: ['768', '64'], ready: true },
];

describe('pickRetrievers', () => {
  it('defaults to the retrievers whose embeddings are cached', () => {
    expect(pickRetrievers(null, opts)).toEqual(['bm25', 'bge-base']);
  });
  it('falls back to all when none is cached', () => {
    expect(pickRetrievers(null, opts.map((o) => ({ ...o, ready: false })))).toEqual(['bm25', 'bge-small', 'bge-base']);
  });
  it('takes the lens as given, dropping ids the API does not know', () => {
    expect(pickRetrievers('bge-small,nope', opts)).toEqual(['bge-small']);
    expect(pickRetrievers('', opts)).toEqual([]);
  });
});
