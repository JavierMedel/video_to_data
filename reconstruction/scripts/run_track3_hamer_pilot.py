#!/usr/bin/env python3
"""Queue the bounded HaMeR pass after verified model provisioning completes."""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import threading
import time

import numpy as np

from prepare_track3_campaign import ROOT, sha256
from run_track3_object_pilot import cpath, run_stage, save, telemetry
from run_track3_hand_pilot import command, guard


def main():
    work = ROOT / 'data/track3/pilot/episode_000038'
    reports = ROOT / 'incident_reports/track3_baseline/episode_000038'
    provision = ROOT / 'incident_reports/track3_baseline/hamer_provisioning.json'
    status = reports / 'hamer_pipeline.json'
    state = {'state': 'WAITING_FOR_VERIFIED_WEIGHTS', 'episode_index': 38,
             'policy_trained': False, 'submission_ready': False, 'uploaded': False,
             'pytorch_memory_fraction': .55, 'gpu_memory_guard_mib': 19000}
    save(status, state)
    print('Waiting for HaMeR download, gzip verification and extraction', flush=True)
    while True:
        if provision.exists():
            current = json.loads(provision.read_text())
            if current['state'] == 'COMPLETE': break
            if current['state'] == 'INTERRUPTED_OR_FAILED':
                state.update(state='WEIGHT_PROVISIONING_FAILED', error=current.get('error'))
                save(status, state)
                raise RuntimeError('HaMeR provisioning failed; see hamer_provisioning.json')
        time.sleep(10)
    lock = (work / 'pilot.lock').open('w')
    state['state'] = 'WAITING_FOR_GPU_STAGE_LOCK'; save(status, state)
    fcntl.flock(lock, fcntl.LOCK_EX)
    if json.loads((reports / 'hand_pipeline.json').read_text()).get('state') != 'COMPLETE':
        raise RuntimeError('The reviewed hand-mask/association pipeline must complete before HaMeR refinement')
    masks, tracks = work / 'hand_masks', work / 'hand_tracks.json'
    for track in ('2', '3'):
        if len(list((masks / track).glob('*.png'))) != 263:
            raise RuntimeError('Complete propagated hand masks are required')
    weights = ROOT / 'data/weights/hamer'
    checkpoint = weights / '_DATA/hamer_ckpts/checkpoints/hamer.ckpt'
    if sha256(checkpoint) != current['checkpoint_sha256']:
        raise ValueError('HaMeR checkpoint hash changed after provisioning')
    # The licensed MANO asset is already in the provisioned WiLoR bundle.
    # Copy that unchanged asset into HaMeR's model and renderer conventions.
    source = ROOT / 'data/weights/wilor/pretrained_models/MANO_RIGHT.pkl'
    for relative in ['_DATA/data/mano/MANO_RIGHT.pkl', '_DATA/data/models/MANO_RIGHT.pkl']:
        target = weights / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists(): shutil.copy2(source, target)
        if sha256(target) != sha256(source):
            raise ValueError(f'MANO asset mismatch: {target}')
    mano = ROOT / 'data/weights/track3_mano'
    raw, aligned, filled = work / 'hamer_raw', work / 'hamer_depth_aligned', work / 'hamer_depth_aligned_filled'
    intrinsics = work / 'intrinsics_stable.json'
    common = [Path(__file__), ROOT / 'scripts/run_track3_hand_pilot.py', ROOT / 'scripts/track3_gpu_python.py', tracks,
              source, checkpoint, weights / '_DATA/hamer_ckpts/model_config.yaml']
    state.update(state='RUNNING', checkpoint_sha256=current['checkpoint_sha256'])
    save(status, state)
    stop = threading.Event()
    threads = [threading.Thread(target=telemetry, args=(stop, reports / 'hamer_pipeline_gpu.csv'), daemon=True),
               threading.Thread(target=guard, args=(stop, reports / 'hamer_pipeline_guard.json'), daemon=True)]
    for thread in threads: thread.start()
    try:
        run_stage('hamer_prediction', command('v2d_track3_hands:blackwell', 'v2d.hamer.lib.masks_to_hands',
            ['--frames_dir', cpath(work / 'frames'), '--masks_dir', cpath(masks), '--tracks_path', cpath(tracks),
             '--output_dir', cpath(raw), '--weights_dir', cpath(weights)]), [raw],
            [*common, *sorted(masks.glob('*/*.png'))], reports)
        run_stage('hamer_depth_alignment', command('v2d_track3_hands:blackwell', 'v2d.hamer.lib.align_hands',
            ['--hamer_dir', cpath(raw), '--depth_dir', cpath(work / 'depth'), '--intrinsics_path', cpath(intrinsics),
             '--mano_assets_root', cpath(mano), '--output_dir', cpath(aligned), '--hand_masks_dir', cpath(masks),
             '--object_masks_dir', cpath(work / 'masks/1')]), [aligned],
            [*common, intrinsics, *sorted(raw.glob('*/*.json')), *sorted((work / 'depth').glob('*.png')),
             *sorted(masks.glob('*/*.png')), *sorted((work / 'masks/1').glob('*.png'))], reports)
        run_stage('hamer_gap_fill', command('v2d_track3_hands:blackwell', 'v2d.wilor.lib.tracks_interpolate',
            ['--aligned_dir', cpath(aligned), '--masks_dir', cpath(masks), '--output_dir', cpath(filled),
             '--betas', 'fixed', '--max_gap_frames', '15', '--no_extrapolate']), [filled],
            [*common, *sorted(aligned.glob('*/*.json')), *sorted(masks.glob('*/*.png'))], reports)
        overlay = work / 'hamer_depth_aligned_overlay.mp4'
        run_stage('hamer_alignment_overlay', command('v2d_track3_hands:blackwell', 'v2d.hamer.lib.render_hands_aligned_video',
            ['--frames_dir', cpath(work / 'frames'), '--aligned_dir', cpath(filled),
             '--mano_assets_root', cpath(mano), '--output_path', cpath(overlay)]), [overlay],
            [*common, *sorted(filled.glob('*/*.json'))], reports)
        counts = {}
        for track in ('2', '3'):
            records = [json.loads(p.read_text()) for p in sorted((filled / track).glob('*.json'))]
            if not records: raise ValueError(f'No aligned frames for hand track {track}')
            for record in records:
                if not np.isfinite(record['camera']['cam_t']).all():
                    raise ValueError('Nonfinite HaMeR hand translation')
            counts[track] = {'frames': len(records), 'interpolated': sum(bool(r.get('interpolated')) for r in records)}
        state.update(state='COMPLETE', tracks=counts, overlay=str(overlay.relative_to(ROOT)),
                     limitations=['Camera-frame estimates; world alignment, contact validation and policy training remain.'])
    except Exception as exc:
        state.update(state='FAILED', error=str(exc))
        raise
    finally:
        stop.set()
        for thread in threads: thread.join(timeout=10)
        state['observed_utc'] = datetime.now(timezone.utc).isoformat(); save(status, state)


if __name__ == '__main__':
    main()
