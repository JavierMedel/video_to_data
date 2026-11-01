import { readFileSync, readdirSync, statSync } from 'node:fs';
import { resolve } from 'node:path';
import { episodes, media, metrics, tracks } from '../src/data/project';
import { validateContent, validateInventory } from './content-validation';

const publicDir = resolve('public');
const inventory = JSON.parse(readFileSync('asset-inventory.json', 'utf8'));
const errors = [...validateContent(episodes, media, metrics, publicDir), ...validateInventory(inventory, publicDir)];
const publishedInventory = JSON.parse(readFileSync('public/evidence/asset-inventory.json', 'utf8'));
if (JSON.stringify(inventory) !== JSON.stringify(publishedInventory)) errors.push('Published inventory differs from source.');
const exportedMetrics = JSON.parse(readFileSync('public/evidence/results.json', 'utf8'));
if (JSON.stringify(metrics) !== JSON.stringify(exportedMetrics)) errors.push('Public score export differs from typed content.');
const snapshot = JSON.parse(readFileSync('../reconstruction/incident_reports/cross_track_validation_plan/snapshots_20261025_completion_v1/summary.json', 'utf8'));
for (const metric of metrics) {
  const saved = snapshot.competitions[metric.id]?.own_latest;
  if (metric.value !== (saved?.publicScore != null ? Number(saved.publicScore) : null)) errors.push(`Score differs from snapshot: ${metric.id}`);
  if (metric.submissionId !== (saved?.ref ?? null)) errors.push(`Submission differs from snapshot: ${metric.id}`);
  const definitionPath = metric.definition.split(`/blob/`)[1]?.split('/').slice(1).join('/');
  if (!definitionPath || !statSync(resolve('..', definitionPath)).isFile()) errors.push(`Missing metric definition: ${metric.id}`);
}
for (const track of tracks) if (Number(track.value) !== metrics.find(m => m.track === track.link && m.label === track.metric.split(' · ')[0])?.value) errors.push(`Unproven track headline: ${track.title}`);
function scan(dir: string) {
  for (const file of readdirSync(dir)) {
    const path = resolve(dir, file);
    if (statSync(path).isDirectory()) { scan(path); continue; }
    if (/\.(json|csv|md|svg|html|tsx?|css)$/.test(file)) {
      const content = readFileSync(path, 'utf8');
      if (/\/home\/|\/mnt\/[a-z]\/Users|(?:sk-proj-|ghp_)[A-Za-z0-9]{15}/.test(content)) errors.push(`Private path or token pattern: ${path}`);
    }
  }
}
scan(publicDir); scan(resolve('src'));
for (const path of ['evidence/README.md', 'evidence/MISSING.md', 'favicon.svg', 'media/hero.webp']) if (!statSync(resolve(publicDir, path)).isFile()) errors.push(`Missing resource: ${path}`);
if (errors.length) { console.error(errors.join('\n')); process.exit(1); }
console.log(`Validated ${episodes.length} episodes/cohorts, ${media.length} media records, ${metrics.length} scores, and ${inventory.length} asset hashes. Missing Tier 3 scores remain null.`);
