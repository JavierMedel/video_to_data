import { test } from 'node:test';
import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { episodes, media, metrics } from '../src/data/project';
import { validateContent, validateInventory } from './content-validation';

test('real content references resolve and unknown references are rejected', () => {
  assert.deepEqual(validateContent(episodes, media, metrics, resolve('public')), []);
  const invalid = structuredClone(episodes); invalid[0].media.push('nonexistent');
  assert.ok(validateContent(invalid, media, metrics, resolve('public')).some(e => e.includes('Unknown media')));
});
test('missing scores and a genuine scored zero retain different meanings', () => {
  assert.equal(metrics.filter(m => m.track === 'track_2_tier3').length, 4);
  assert.ok(metrics.filter(m => m.track === 'track_2_tier3').every(m => m.value === null && m.submissionId === null));
  assert.equal(metrics.find(m => m.track === 'track_3' && m.label === 'MP-SR')?.value, 0);
  assert.equal(metrics.filter(m => m.track === 'track_3').length, 6);
  assert.equal(metrics.find(m => m.track === 'track_3' && m.label === 'CD-O')?.value, 0.57037);
});
test('unavailable media cannot pretend to be a completed artifact', () => {
  const invalid = structuredClone(media); invalid.find(m => m.availability === 'unavailable')!.src = 'media/e15-input.mp4';
  assert.ok(validateContent(episodes, invalid, metrics, resolve('public')).some(e => e.includes('Unavailable media')));
});
test('Track 3 input and saved visualization timebases stay distinct', () => {
  const overlay = media.find(m => m.id === 't3-hands')!;
  assert.equal(overlay.sourceFps, 20);
  assert.equal(overlay.playbackFps, 30);
  assert.equal(episodes.find(e => e.id === overlay.episode)?.fps, 20);
});
test('published assets reject a missing file and a mismatched content hash', () => {
  assert.ok(validateInventory([{ output: 'media/not-here.mp4', bytes: 1, sha256: 'bad' }], resolve('public')).length);
  assert.ok(validateInventory([{ output: 'media/hero.webp', bytes: 1, sha256: 'bad' }], resolve('public')).some(e => e.includes('hash/size')));
});
