import { copyFileSync } from 'node:fs';
for (const name of ['README.md', 'MISSING.md']) copyFileSync(name, `public/evidence/${name}`);
copyFileSync('asset-inventory.json', 'public/evidence/asset-inventory.json');
