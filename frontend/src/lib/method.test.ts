import { describe, expect, it } from 'vitest';
import { methodBatch, runTrain } from '../test/fixtures';
import { RECIPE, bytes, diagStats, indexBytes, lrAt, recipeOf, rescued, sci, warmupSteps } from './method';

describe('the recipe', () => {
  it('reads the run, and falls back to the config file per key', () => {
    const r = recipeOf(runTrain);
    expect(r.epochs).toBe(3); // the fixture's config says 3
    expect(r.batch_size).toBe(32);
    expect(r.learning_rate).toBe(RECIPE.learning_rate); // the fixture names it `lr`, so the file's value
    expect(r.total_steps).toBe(525);
    expect(r.steps_per_epoch).toBe(175);
    expect(r.warmup_steps).toBe(53);
  });

  it('with no run, derives steps from the pair count', () => {
    const r = recipeOf(null, 5598);
    expect(r.steps_per_epoch).toBe(175); // 5,598 / 32 rounded up: the last batch is kept
    expect(r.total_steps).toBe(700);
    expect(r.warmup_steps).toBe(70);
  });
});

describe('the learning-rate schedule (Transformers v5, linear)', () => {
  it('reads a warmup below 1 as a ratio, rounded up', () => {
    expect(warmupSteps(0.1, 700)).toBe(70);
    expect(warmupSteps(0.1, 525)).toBe(53);
    expect(warmupSteps(100, 700)).toBe(100);
  });

  it('ramps up, peaks at the end of warmup, and reaches 0 at the last step', () => {
    expect(lrAt(0, 2e-5, 70, 700)).toBe(0);
    expect(lrAt(35, 2e-5, 70, 700)).toBeCloseTo(1e-5);
    expect(lrAt(70, 2e-5, 70, 700)).toBeCloseTo(2e-5);
    expect(lrAt(385, 2e-5, 70, 700)).toBeCloseTo(1e-5);
    expect(lrAt(700, 2e-5, 70, 700)).toBe(0);
  });
});

describe('the worked example', () => {
  it('averages the diagonal and the rest', () => {
    expect(diagStats([[1, 0.2], [0.4, 0.8]])).toEqual({ diag: 0.9, off: expect.closeTo(0.3) });
  });

  it("finds questions the base model misses in the corpus and the tuned model doesn't", () => {
    const [base, tuned] = methodBatch.models;
    expect(rescued(base, tuned)).toEqual([1]);
  });
});

describe('sizes and numbers', () => {
  it('prices a float32 index', () => {
    expect(indexBytes(768, 6757)).toEqual({ per: 3072, total: 20_757_504 });
    expect(bytes(3072)).toBe('3,072 B');
    expect(bytes(20_757_504)).toBe('20.8 MB');
    expect(bytes(1_729_792)).toBe('1.7 MB');
  });

  it('writes learning rates in short scientific form', () => {
    expect(sci(2e-5)).toBe('2e-5');
    expect(sci(1.9714e-5)).toBe('1.97e-5');
    expect(sci(0)).toBe('0');
  });
});
