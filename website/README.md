# Video to Data — independent research showcase

Javier Medel’s static research website covers all three NVIDIA Video to Data Challenge 2026 tracks. It presents real local artifacts and the **saved October 25, 2026 scoring snapshot**. It is an independent project, not an NVIDIA website. No live leaderboard query or reconstruction inference is part of the site.

## Run the website

Use Node.js 22.12+ (Node 24 was used for local validation).

```sh
cd website
npm ci
npm run dev
```

Open the printed URL with `/video_to_data/`. The project is isolated from the Python/GPU packages. No model dependencies are installed for the website.

```sh
npm run test
npm run build
npm run preview
```

The build runs strict TypeScript and content validation before writing `website/dist/`. `npm run validate` checks episode/media references, all media hashes and sizes, LFS pointers, exact agreement with the preserved scoring snapshot, missing scores, public export parity, and private path/token patterns. No lint configuration is installed; type and content checks are explicit.

To use another project subpath:

```sh
VITE_BASE_PATH=/my-repository/ npm run build
VITE_BASE_PATH=/my-repository/ npm run preview
```

Every local content URL uses Vite’s base path. There are no server-side routes. For an account root or custom domain use `VITE_BASE_PATH=/`.

## Visual theme

`src/brand.css` defines the NVIDIA-inspired presentation: black section backgrounds, the official green `#76B900` for filled controls and rules, brighter `#9BDC35` for small text, heavier typography, a challenge banner, a larger unaltered official logo, and full-width hero media. Track shortcuts link directly to the three result cards. This is an original independent-project layout; the non-endorsement attribution remains visible.

## Portrait layout and autoplay

The whole website appears in a centered **2:3 portrait frame** that fits the laptop or phone viewport. Content scrolls inside that frame; navigation remains sticky. Container queries make the layout respond to the frame width. The aspect ratio does not crop or stretch any research media.

Hero, explorer and dialog videos **autoplay muted and loop when visible**, pause when out of view or when the browser tab is hidden, and allow only one active clip. A manual Pause or Reset stays paused when scrolling away and back. Reduced-motion preferences disable automatic playback; explicit Play still works. If a browser blocks autoplay, the player explains this and keeps its Play control available. Offscreen video files are not downloaded eagerly.

For a local laptop preview, run `npm run dev` and open **http://localhost:5173/video_to_data/**. A static `file://` URL does not provide the website's asset routing.

## GitHub Pages

`.github/workflows/website-pages.yml` installs with `npm ci`, tests, validates and builds the website, stages existing static `docs/` alongside it, and deploys the combined artifact. The showcase becomes the project homepage. The original toolkit landing page is preserved as `toolkit.html`; documentation subdirectories remain in place. It runs manually or on website/static-doc changes on `main` or the project working branch `v2d-track1-blackwell`. Pull requests build without deploying.

The publication target is **https://javiermedel.github.io/video_to_data/**. Pages uses **Settings → Pages → Build and deployment → Source: GitHub Actions**. A push to the working branch can publish without merging its research history into `main`. The initial implementation was local; publication was subsequently authorized by the author. The workflow uses repository-name-derived base paths, no secrets, and grants Pages/id-token write permission only to its deployment job.

The inherited `Docs` workflow’s deployment is disabled specifically for `JavierMedel/video_to_data` to avoid two workflows overwriting the same Pages deployment. Its upstream deployment and documentation builds are otherwise retained. The showcase workflow stages the checked-in documentation, not the separate Sphinx multiversion build. If the repository is renamed, update that fork-specific exclusion in `docs.yml`.

## Content architecture

- `src/data/project.ts`: typed project, episode, media, pipeline, observations and acknowledgment records.
- `src/data/results.json`: typed official score inputs; missing results use `null`.
- `src/data/provenance.json`: inspected commit, dates and verified public resource URLs.
- `src/data/coverage.json`: file counts derived from Episode 15’s original cache.
- `asset-inventory.json`: repository-relative original paths, transformations, SHA-256, file sizes and decoded video metadata.
- `public/evidence/`: downloadable inventory, scores, this guide and missing-asset record.
- `public/media/`: browser-friendly derivatives only. No model weights or large arrays.

The original caches are Git-ignored and are **not** expected on Pages or in a fresh CI checkout. Their media provenance links open the published inventory rather than a nonexistent GitHub file. Tracked reports and scoring definitions link to the inspected commit. The inventory uses repository-relative source identifiers; no private workstation paths appear in the public site.

## Prepare media without running reconstruction

Prepared derivatives are included as ordinary files and need no regeneration to build the site. To regenerate, the corresponding original caches must exist. Never run reconstruction merely to populate the website.

From the repository root, use an environment that already has Pillow, NumPy, OpenCV and trimesh. The existing reconstruction environment supplied these in the inspected checkout. FFmpeg must include H.264 encoding:

```sh
reconstruction/.venv/bin/python website/scripts/prepare_media.py --ffmpeg /path/to/ffmpeg
```

An optional isolated encoder setup, without changing the reconstruction environment:

```sh
reconstruction/.venv/bin/pip install --target website/.media-venv/vendor --no-deps imageio-ffmpeg
reconstruction/.venv/bin/python website/scripts/prepare_media.py
```

The script reads original outputs only. It produces H.264/yuv420p MP4s with fast-start metadata, preserves complete frame ranges and source frame rates, resizes without cropping, and creates WebP posters. Encoding is lossy for browser delivery; no interpolation, enhancement, stabilization or inference is used. Existing MP4 derivatives are retained on rerun; to change encoding settings, explicitly choose new derivative filenames and update the manifest.

SAM2’s frame-154 mask is overlaid on that exact RGB frame. Depth uses a **relative grayscale display**, normalized over the positive-value 2nd–98th percentile; dark near, light far, invalid black. This does not establish ground-truth or calibrated metric depth. The mesh poster is an orthographic flat-shaded render of the actual simplified backup; the GLB retains its own materials. The existing original mesh remains untouched and is not included as a public download.

The lazy Three.js viewer loads only after **Explore in 3D**. Orbit/zoom, keyboard rotation/zoom, camera reset, wireframe, download, and a WebGL failure fallback are provided. It does not rotate automatically. Display mesh complexity and original complexity are separately labeled, and no units are assumed to be metres.

## Add an episode

1. Locate real outputs and record their exact run, frame range, timebase, attribution and distribution terms. Decode/check the video before claiming alignment. Do not reuse another episode’s preview.
2. Add explicit derivatives to `scripts/prepare_media.py`, or prepare them with a documented equivalent process. Add the outputs, source identifiers, transformation and hashes to `asset-inventory.json`. Keep large originals outside the website.
3. Add a typed `Episode` and `Media` records in `src/data/project.ts`, referencing the episode ID and available outputs. Use `src: null`, `poster: null`, and `availability: 'unavailable'` for missing media. The gallery/explorer populate from these records; no component edits are necessary.
4. Add only sourced metrics. Keep every unmeasured value `null`; a true scored zero must have a run, timestamp and submission receipt. Snapshot updates must update both typed JSON and the public JSON/CSV, and the validator’s snapshot source.
5. Run `npm test`, `npm run build`, and browser checks. Review attribution and the public inventory. Update missing-content notes.

The Track 3 input recording is 20 FPS; its saved overlay is encoded at 30 FPS. The player labels both, preserves the saved output timing, and does not present output seconds as input-video seconds. The website intentionally presents separate clips rather than claiming unverified synchronization. The RGB/mask slider uses one exact frame. If synchronized playback is added later, validate episode, timebase, offsets and drift handling first.

## Browser checks

```sh
npx playwright install chromium
npm run build
npm run test:browser
```

For a workspace-local or temporary browser cache, set `PLAYWRIGHT_BROWSERS_PATH` consistently during install and test (the implementation used `/tmp/v2d-playwright`). Browser tests cover keyboard dialog/focus behavior, accessible metric tabs, null versus zero, episode/view selection, the aligned-mask slider, video controls, media errors, WebGL fallback, valid anchors, lazy loading, and 360/768/1440/1920 px page overflow. They run against the production build under `/video_to_data/` and save responsive screenshots in ignored `test-results/`.

See `VALIDATION.md` for the actual recorded checks, and `MISSING.md` for unavailable evidence. Upstream toolkit/model licenses apply separately from the dataset’s NVIDIA / CC BY 4.0 attribution. LinkedIn was supplied by the author; automated verification was blocked by LinkedIn’s HTTP 999 response. Repository, upstream, challenge and dataset URLs resolved during research.

## Official NVIDIA logo attribution

At the author’s request, the hero and acknowledgments use the unaltered green/white horizontal NVIDIA logo shipped in the official toolkit (`docs/hydra-0/img/nvidia-logo.png`). An unchanged horizontal SVG downloaded from NVIDIA’s official brand-guidelines page is also preserved in `public/brand/`. Both sources and hashes are recorded in the inventory. No recoloring, reshaping, or animation is applied. NVIDIA and its logo are NVIDIA trademarks and are not covered by the dataset’s CC BY 4.0 license. The site labels Javier’s work as independent and states that no endorsement is implied.

Official source: https://www.nvidia.com/en-us/about-nvidia/legal-info/logo-brand-usage/
