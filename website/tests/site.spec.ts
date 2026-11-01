import { expect, test } from '@playwright/test';

test('Pages subpath serves valid anchors and lazy media without page errors', async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
  const requests: string[] = []; page.on('request', r => requests.push(r.url()));
  await page.goto('./'); await expect(page.getByRole('heading', { level: 1 })).toHaveText('Video to Data (V2D) Challenge');
  expect(requests.filter(r => /\.mp4(?:$|\?)/.test(r)).every(r => r.endsWith('/media/t1-comparison.mp4'))).toBe(true);
  expect(requests.filter(r => /MeshViewer|\.glb(?:$|\?)/.test(r))).toHaveLength(0);
  expect(await page.locator('a[href^="#"]').evaluateAll(links => links.every(a => !!document.getElementById(a.getAttribute('href')!.slice(1))))).toBe(true);
  await expect(page.locator('.hero-media video')).toHaveAttribute('poster', '/video_to_data/media/t1-comparison.webp');
  expect(errors).toEqual([]);
});
test('gallery dialog traps focus, closes with Escape and restores focus', async ({ page }) => {
  await page.goto('./'); const trigger = page.getByRole('button', { name: 'Inspect Inspecting the mesh projection', exact: true });
  await trigger.click(); const dialog = page.getByRole('dialog'); await expect(dialog).toBeVisible();
  await expect(page.getByRole('button', { name: 'Close preview' })).toBeFocused();
  await page.keyboard.press('Shift+Tab'); expect(await page.evaluate(() => !!document.activeElement?.closest('dialog'))).toBe(true);
  await page.keyboard.press('Escape'); await expect(dialog).not.toBeVisible(); await expect(trigger).toBeFocused();
});
test('metric tabs support keyboard navigation and distinguish null from scored zero', async ({ page }) => {
  await page.goto('./'); await page.getByRole('tab', { name: 'Track 1', exact: true }).focus(); await page.keyboard.press('ArrowRight');
  await expect(page.getByRole('tab', { name: 'Track 2 · Tier 1', exact: true })).toHaveAttribute('aria-selected', 'true');
  await page.getByRole('tab', { name: 'Track 2 · Tier 3', exact: true }).click(); await expect(page.locator('.metrics-table').getByText('Not yet available')).toHaveCount(4);
  await page.getByRole('tab', { name: 'Track 3', exact: true }).click(); await expect(page.locator('.metrics-table').getByText('0.00000', { exact: true })).toHaveCount(1);
  await expect(page.locator('.metrics-table').getByText('0.57037', { exact: true })).toHaveCount(1);
});
test('explorer changes views, reveals matched mask and handles missing rollout', async ({ page }) => {
  await page.goto('./'); await page.getByLabel('Output view', { exact: true }).selectOption('e15-depth');
  await expect(page.locator('.explorer-caption')).toContainText('Estimated relative depth');
  await page.getByRole('button', { name: 'Compare RGB + mask' }).click(); const slider = page.getByRole('slider', { name: 'Mask overlay reveal' });
  await slider.fill('80'); await expect(slider).toHaveValue('80');
  await page.getByLabel('Episode / cohort', { exact: true }).selectOption('t2-cohort'); await expect(page.locator('.explorer-content .missing')).toBeVisible();
  await expect(page.locator('.explorer-content')).toContainText('Preview not yet available');
});
test('video seek, play, speed, reset and media failure remain usable', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('./'); const player = page.locator('.explorer-content .video-player');
  await player.getByRole('button', { name: 'Play video' }).click(); await expect(player.locator('video')).toHaveJSProperty('paused', false);
  await player.getByRole('button', { name: 'Pause video' }).click(); await player.getByRole('slider', { name: 'Seek video in seconds' }).fill('5');
  await expect(player.locator('video')).toHaveJSProperty('currentTime', 5); await player.getByLabel('Playback speed').selectOption('0.5'); await expect(player.locator('video')).toHaveJSProperty('playbackRate', 0.5);
  await player.getByRole('button', { name: 'Reset', exact: true }).click(); await expect(player.locator('video')).toHaveJSProperty('currentTime', 0);
  await page.route('**/media/t3-hands.mp4', route => route.abort()); await page.getByLabel('Episode / cohort', { exact: true }).selectOption('t3-e38');
  await page.locator('.explorer-content').getByRole('button', { name: 'Play video' }).click(); await expect(page.locator('.explorer-content [role=status]')).toContainText(/unavailable|could not/);
});
for (const width of [360, 768, 1440, 1920]) test(`responsive layout at ${width}px has no page overflow`, async ({ page }) => {
  await page.setViewportSize({ width, height: 1000 }); await page.goto('./');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  const frame = page.locator('.site-frame');
  const bounds = await frame.boundingBox();
  expect(bounds!.width / bounds!.height).toBeCloseTo(2 / 3, 2);
  expect(await frame.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
  expect(await page.locator('.track-card h3').evaluateAll(headings => headings.every(el => el.scrollWidth <= el.clientWidth))).toBe(true);
  if (width === 360) { await page.getByRole('button', { name: 'Menu' }).click(); await page.locator('#main-nav').getByRole('link', { name: 'Results', exact: true }).click(); await expect(page.getByRole('button', { name: 'Menu' })).toHaveAttribute('aria-expanded', 'false'); }
  await page.screenshot({ path: `test-results/hero-${width}.png` });
  await page.screenshot({ path: `test-results/layout-${width}.png`, fullPage: true });
});

test('visible videos autoplay muted, preserve manual pause, and pause other players', async ({ page }) => {
  await page.goto('./');
  const hero = page.locator('.hero-media');
  const heroVideo = hero.locator('video');
  await heroVideo.scrollIntoViewIfNeeded();
  await expect(heroVideo).toHaveJSProperty('paused', false);
  await expect(heroVideo).toHaveJSProperty('muted', true);
  await expect(heroVideo).toHaveJSProperty('loop', true);
  await hero.getByRole('button', { name: 'Pause video' }).click();
  await page.locator('#metrics').scrollIntoViewIfNeeded();
  await heroVideo.scrollIntoViewIfNeeded();
  await expect(heroVideo).toHaveJSProperty('paused', true);
  await hero.getByRole('button', { name: 'Play video' }).click();
  const explorerVideo = page.locator('.explorer-content video');
  await explorerVideo.scrollIntoViewIfNeeded();
  await expect(explorerVideo).toHaveJSProperty('paused', false);
  await expect(heroVideo).toHaveJSProperty('paused', true);
});

test('reduced motion keeps autoplay off while allowing explicit playback', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('./');
  const hero = page.locator('.hero-media');
  await hero.locator('video').scrollIntoViewIfNeeded();
  await expect(hero.locator('video')).toHaveJSProperty('paused', true);
  await hero.getByRole('button', { name: 'Play video' }).click();
  await expect(hero.locator('video')).toHaveJSProperty('paused', false);
});

test('blocked autoplay explains the browser policy and retains Play', async ({ page }) => {
  await page.addInitScript(() => { HTMLMediaElement.prototype.play = () => Promise.reject(new DOMException('Autoplay blocked', 'NotAllowedError')); });
  await page.goto('./');
  const hero = page.locator('.hero-media');
  await hero.locator('video').scrollIntoViewIfNeeded();
  await expect(hero.locator('[role=status]')).toContainText('Autoplay is blocked');
  await expect(hero.getByRole('button', { name: 'Play video' })).toBeVisible();
});
test('WebGL failure preserves a static mesh and download', async ({ page }) => {
  await page.addInitScript(() => { const original = HTMLCanvasElement.prototype.getContext; HTMLCanvasElement.prototype.getContext = function(this: HTMLCanvasElement, type: string, ...args: unknown[]) { if (type.startsWith('webgl')) return null; return original.apply(this, [type, ...args] as Parameters<typeof original>); } as typeof original; });
  await page.goto('./'); await page.locator('.case-feature').getByRole('button', { name: 'Explore in 3D' }).click();
  await expect(page.locator('.case-feature [role=status]')).toContainText('WebGL unavailable');
  await expect(page.locator('.case-feature .mesh-viewer img')).toBeVisible();
  await expect(page.locator('.case-feature').getByRole('link', { name: 'Download GLB' })).toHaveAttribute('href', '/video_to_data/media/e15-display.glb');
});

test('actual GLB loads on demand with keyboard orbit, wireframe and reset', async ({ page }) => {
  await page.goto('./'); const mesh = page.locator('.case-feature');
  await mesh.getByRole('button', { name: 'Explore in 3D' }).click();
  await expect(mesh.locator('[role=status]')).toContainText('Mesh loaded');
  await mesh.getByRole('checkbox', { name: 'Wireframe' }).check();
  const canvas = mesh.locator('canvas'); await canvas.focus(); await page.keyboard.press('ArrowRight'); await page.keyboard.press('+');
  await mesh.getByRole('button', { name: 'Reset camera' }).click();
  await expect(canvas).toBeVisible();
  await mesh.screenshot({ path: 'test-results/mesh-viewer.png' });
});
