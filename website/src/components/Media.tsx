import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import { asset, episodes, media, repoFile, type Media as MediaRecord } from '../data/project';

const MeshViewer = lazy(() => import('./MeshViewer'));

export function Missing({ text = 'Preview not yet available' }: { text?: string }) {
  return <div className="missing"><span className="missing-icon" aria-hidden="true">↗</span><strong>{text}</strong><span>Evidence is documented. A visual preview is still to be added.</span></div>;
}

export function Poster({ item, eager = false }: { item: MediaRecord; eager?: boolean }) {
  const [failed, setFailed] = useState(false);
  if (!item.poster || failed) return <Missing text={failed ? 'Preview could not be loaded' : undefined} />;
  return <img src={asset(item.poster)} alt={item.alt} loading={eager ? 'eager' : 'lazy'} onError={() => setFailed(true)} />;
}

export function VideoPlayer({ item }: { item: MediaRecord }) {
  const video = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const fps = item.playbackFps ?? item.sourceFps;
  const [duration, setDuration] = useState(item.frameRange && fps ? (item.frameRange[1] - item.frameRange[0] + 1) / fps : 0);
  const [rate, setRate] = useState(1);
  const [message, setMessage] = useState('');
  const requestedTime = useRef(0);
  useEffect(() => {
    const el = video.current;
    if (!el) return;
    const observer = new IntersectionObserver(([entry]) => { if (!entry.isIntersecting) el.pause(); }, { threshold: 0.05 });
    const hidden = () => { if (document.hidden) el.pause(); };
    observer.observe(el);
    document.addEventListener('visibilitychange', hidden);
    return () => { observer.disconnect(); document.removeEventListener('visibilitychange', hidden); el.pause(); };
  }, []);
  if (!item.src) return <Missing />;
  async function toggle() {
    const el = video.current;
    if (!el) return;
    if (el.paused) {
      // A page contains at most one actively playing research clip.
      document.querySelectorAll('video').forEach(other => { if (other !== el) other.pause(); });
      try { await el.play(); setMessage(''); } catch { setMessage('Playback could not start. Try again or download the clip.'); }
    } else el.pause();
  }
  function seek(value: number) {
    requestedTime.current = value;
    setTime(value);
    if (video.current) {
      if (video.current.readyState === 0) video.current.load();
      else video.current.currentTime = value;
    }
  }
  return <div className="video-player">
    <video ref={video} src={asset(item.src)} poster={item.poster ? asset(item.poster) : undefined} preload="none" playsInline muted aria-label={item.title}
      onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)}
      onTimeUpdate={e => setTime(e.currentTarget.currentTime)}
      onLoadedMetadata={e => { setDuration(e.currentTarget.duration); e.currentTarget.currentTime = requestedTime.current; e.currentTarget.playbackRate = rate; }}
      onWaiting={() => setMessage('Buffering…')} onPlaying={() => setMessage('')}
      onCanPlay={() => setMessage('')} onError={() => setMessage('Video unavailable. The poster remains available; try the download link.')} />
    <div className="player-controls">
      <button className="play-button" onClick={toggle} aria-label={playing ? 'Pause video' : 'Play video'}>{playing ? 'Ⅱ' : '▶'} <span>{playing ? 'Pause' : 'Play'}</span></button>
      <label className="seek-label"><span className="sr-only">Seek video in seconds</span><input type="range" min="0" max={duration || 1} step={fps ? 1 / fps : 0.01} value={Math.min(time, duration)} onChange={e => seek(Number(e.target.value))} aria-valuetext={`${time.toFixed(2)} seconds`} /></label>
      <output className="time">{time.toFixed(1)} / {duration.toFixed(1)} s</output>
      <label className="speed"><span className="sr-only">Playback speed</span><select value={rate} onChange={e => { const next = Number(e.target.value); setRate(next); if (video.current) video.current.playbackRate = next; }}>{[0.25, 0.5, 1, 1.5, 2].map(r => <option key={r} value={r}>{r}×</option>)}</select></label>
      <button className="quiet-button" onClick={() => { video.current?.pause(); seek(0); setRate(1); if (video.current) video.current.playbackRate = 1; }}>Reset</button>
    </div>
    <div className="player-readout"><span>{fps ? `${item.playbackFps ? 'Output frame' : 'Frame'} ${Math.min(item.frameRange?.[1] ?? Infinity, Math.floor(time * fps) + (item.frameRange?.[0] ?? 0))} · ${item.playbackFps ? `${fps} FPS output / ${item.sourceFps} FPS input` : `${fps} FPS source`}` : 'Precomputed output'}</span><span role="status">{message}</span><a href={asset(item.src)} download>Download clip ↓</a></div>
  </div>;
}

export function MeshPanel({ item }: { item: MediaRecord }) {
  const [load, setLoad] = useState(false);
  return <div className="mesh-panel">{load ? <Suspense fallback={<div className="mesh-loading">Loading viewer…</div>}><MeshViewer item={item} /></Suspense> : <><Poster item={item} /><button className="button primary mesh-load" onClick={() => setLoad(true)}>Explore in 3D <span aria-hidden="true">↗</span></button></>}</div>;
}

export function MediaDialog({ item, close }: { item: MediaRecord | null; close: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    if (!item) return;
    const previous = document.activeElement as HTMLElement | null;
    dialog.current?.showModal();
    const old = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { dialog.current?.close(); document.body.style.overflow = old; previous?.focus(); };
  }, [item]);
  return <dialog ref={dialog} className="media-dialog" aria-labelledby="dialog-title" onCancel={e => { e.preventDefault(); close(); }} onClick={e => { if (e.target === e.currentTarget) close(); }}>
    {item && <><div className="dialog-top"><span className="eyebrow">{item.stage} / {item.status}</span><button autoFocus className="quiet-button" aria-label="Close preview" onClick={close}>Close <span aria-hidden="true">×</span></button></div>
      <h2 id="dialog-title">{item.title}</h2>
      {item.availability === 'unavailable' ? <Missing /> : item.type === 'video' ? <VideoPlayer key={item.id} item={item} /> : item.type === 'mesh' ? <MeshPanel item={item} /> : <div className="dialog-image"><Poster item={item} /></div>}
      <p>{item.caption}</p><dl className="media-details"><div><dt>Episode / cohort</dt><dd>{episodes.find(e => e.id === item.episode)?.title}</dd></div><div><dt>Run</dt><dd>{item.run}</dd></div><div><dt>Frame range</dt><dd>{item.frameRange ? item.frameRange.join('–') : 'Not established'}</dd></div></dl>
      <p className="fine">{item.attribution} {item.permission}</p>
      <div className="link-row"><a href={repoFile(item.sourcePath)} target="_blank" rel="noreferrer">Source provenance ↗</a>{item.src && <a href={asset(item.src)} download>Download {item.type === 'mesh' ? 'display GLB' : 'display asset'} ↓</a>}</div>
    </>}
  </dialog>;
}

function AlignedComparison() {
  const [position, setPosition] = useState(50);
  return <div className="aligned-comparison"><div className="compare-images"><img src={asset('media/e15-input.webp')} alt="Unmodified RGB frame 154" /><img style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }} src={asset('media/e15-segmentation.webp')} alt="Same frame with translucent SAM2 object mask" /><div className="compare-divider" style={{ left: `${position}%` }} /></div><label>Mask overlay reveal <input type="range" min="0" max="100" value={position} onChange={e => setPosition(Number(e.target.value))} aria-valuetext={`${position}% overlay revealed`} /></label><p className="fine">Matched RGB and mask at frame 154. This slider reveals segmentation, not a reconstruction improvement.</p></div>;
}

export function EpisodeExplorer() {
  const [episodeId, setEpisodeId] = useState('t1-e15');
  const [viewId, setViewId] = useState('e15-input');
  const [compare, setCompare] = useState(false);
  const episode = episodes.find(e => e.id === episodeId)!;
  const item = media.find(m => m.id === viewId)!;
  return <section id="explorer" className="section explorer-section"><div className="section-top"><div><p className="eyebrow">02 / Explore the evidence</p><h2>Take a closer look.</h2></div><p>Browse precomputed outputs. Each view keeps its original run and scope visible.</p></div>
    <div className="explorer-shell"><aside className="explorer-sidebar"><label className="field-label">Episode / cohort<select aria-label="Episode / cohort" value={episodeId} onChange={e => { const next = episodes.find(x => x.id === e.target.value)!; setEpisodeId(next.id); setViewId(next.media[0]); setCompare(false); }}>{episodes.map(e => <option key={e.id} value={e.id}>{e.title}</option>)}</select></label>
      <label className="field-label">Output view<select aria-label="Output view" value={viewId} onChange={e => { setViewId(e.target.value); setCompare(false); }}>{episode.media.map(id => { const m = media.find(x => x.id === id)!; return <option key={id} value={id} disabled={m.availability !== 'available'}>{m.stage}{m.availability !== 'available' ? ' · unavailable' : ''}</option>; })}</select></label>
      <div className="episode-info"><span className="tag">{episode.track}</span><h3>{episode.object.replaceAll('_', ' ')}</h3><p className="mono">{episode.title.split(' · ')[1]}</p><dl><div><dt>Source frames</dt><dd>{episode.frames ?? 'Not applicable'}</dd></div><div><dt>Source FPS</dt><dd>{episode.fps ?? 'Not applicable'}</dd></div><div><dt>Status</dt><dd>{episode.status}</dd></div></dl></div>
      {episodeId === 't1-e15' && <button className="button secondary" onClick={() => setCompare(v => !v)} aria-pressed={compare}>{compare ? 'Return to output view' : 'Compare RGB + mask'}</button>}
      <p className="fine">Separate runs are presented individually. No synchronization between different clips is implied.</p>
    </aside><div className="explorer-content"><div className="media-label"><span>{compare ? 'RGB ↔ segmentation · exact frame 154' : item.title}</span><span className="mono">{item.status}</span></div>
      {compare ? <AlignedComparison /> : item.availability === 'unavailable' ? <Missing /> : item.type === 'video' ? <VideoPlayer key={item.id} item={item} /> : item.type === 'mesh' ? <MeshPanel key={item.id} item={item} /> : <div className="explorer-image"><Poster key={item.id} item={item} /></div>}
      <div className="explorer-caption"><p>{compare ? 'The same source frame and its saved object mask. Segmentation identifies the visible object pixels.' : item.caption}</p><a href={repoFile(item.sourcePath)}>Inspect provenance ↗</a></div>
    </div></div>
  </section>;
}
