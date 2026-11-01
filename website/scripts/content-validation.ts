import { existsSync, readFileSync, statSync } from 'node:fs';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';
import type { Episode, Media, Metric } from '../src/data/project';

export function validateContent(episodes: Episode[], media: Media[], metrics: Metric[], publicDir: string) {
  const errors: string[] = [];
  function unique(items: { id: string }[], name: string) {
    const ids = new Set<string>();
    for (const item of items) { if (ids.has(item.id)) errors.push(`Duplicate ${name}: ${item.id}`); ids.add(item.id); }
  }
  unique(episodes, 'episode'); unique(media, 'media'); unique(metrics, 'metric');
  function localAsset(path: string, context: string) {
    if (/^https:\/\//.test(path)) return;
    if (path.startsWith('/') || path.includes('..') || path.includes('\\')) { errors.push(`Unsafe local asset: ${context}`); return; }
    const file = resolve(publicDir, path);
    if (!existsSync(file) || !statSync(file).isFile()) errors.push(`Missing asset ${path}: ${context}`);
    else if (readFileSync(file).subarray(0, 100).toString().startsWith('version https://git-lfs.github.com')) errors.push(`LFS pointer instead of media: ${path}`);
  }
  for (const item of media) {
    if (!episodes.some(e => e.id === item.episode)) errors.push(`Unknown episode on ${item.id}`);
    if (item.availability === 'available' && (!item.src || !item.poster)) errors.push(`Available media has no asset: ${item.id}`);
    if (item.availability === 'unavailable' && item.src !== null) errors.push(`Unavailable media must have null source: ${item.id}`);
    if (item.src) localAsset(item.src, item.id);
    if (item.poster) localAsset(item.poster, item.id);
    if (!item.caption || !item.alt || !item.sourcePath || !item.permission) errors.push(`Missing provenance/caption: ${item.id}`);
    if (item.frameRange && item.frameRange[0] > item.frameRange[1]) errors.push(`Invalid frame range: ${item.id}`);
  }
  for (const ep of episodes) {
    for (const id of [...ep.media, ...(ep.mesh ? [ep.mesh] : [])]) if (!media.some(m => m.id === id)) errors.push(`Unknown media ${id} on ${ep.id}`);
    for (const id of ep.officialMetrics) if (!metrics.some(m => m.id === id)) errors.push(`Unknown metric ${id} on ${ep.id}`);
  }
  for (const metric of metrics) {
    if (!['track_1', 'track_2_tier1', 'track_2_tier2', 'track_2_tier3', 'track_3'].includes(metric.track)) errors.push(`Unknown metric track: ${metric.id}`);
    if (metric.value !== null && (!Number.isFinite(metric.value) || metric.value < 0)) errors.push(`Invalid metric ${metric.id}`);
    if (metric.value !== null && (!metric.run || !metric.timestamp || !metric.source || !metric.submissionId)) errors.push(`Unproven result: ${metric.id}`);
  }
  return errors;
}

export function validateInventory(inventory: { output: string; sha256: string; bytes: number }[], publicDir: string) {
  const errors: string[] = [];
  for (const item of inventory) {
    const file = resolve(publicDir, item.output);
    if (!file.startsWith(resolve(publicDir) + '/')) { errors.push(`Unsafe inventory path: ${item.output}`); continue; }
    if (!existsSync(file)) { errors.push(`Missing inventory asset ${item.output}`); continue; }
    const bytes = readFileSync(file);
    if (createHash('sha256').update(bytes).digest('hex') !== item.sha256 || bytes.length !== item.bytes) errors.push(`Inventory hash/size mismatch: ${item.output}`);
  }
  return errors;
}
