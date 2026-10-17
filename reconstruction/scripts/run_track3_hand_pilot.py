#!/usr/bin/env python3
"""Continue Episode 38 with hand masks, association, depth alignment and overlay.

Uses the completed WiLoR estimates while the separate HaMeR checkpoint download
finishes. Stages are sequential, bounded, and resume with verified hashes.
"""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading

import cv2
import numpy as np

from prepare_track3_campaign import ROOT, sha256
from run_track3_object_pilot import cpath, run_stage, save, telemetry

NAME = 'v2d-track3-hand-stage'


def command(image, module, args):
    source_mounts = (['-v', f'{ROOT}/modules/v2d_wilor/lib:/workspace/v2d_wilor/lib:ro',
                      '-v', f'{ROOT}/modules/v2d_hamer/lib:/workspace/v2d_hamer/lib:ro']
                     if image == 'v2d_track3_hands:blackwell' else [])
    return ['docker', 'run', '--rm', '--name', NAME, '--gpus', 'all',
            '--memory', '16g', '--memory-swap', '18g', '--cpus', '6',
            '--user', f'{os.getuid()}:{os.getgid()}',
            '--entrypoint', '/repo/scripts/track3_gpu_python.py',
            '-e', 'HOME=/tmp', '-e', 'OMP_NUM_THREADS=4', '-e', 'OPENBLAS_NUM_THREADS=4',
            '-e', 'V2D_TORCH_MEMORY_FRACTION=0.55',
            '-e', 'NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics',
            '-v', f'{ROOT}:/repo', *source_mounts, image, '-m', module, *args]


def guard(stop, path):
    while not stop.wait(.5):
        try:
            sample = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=5)
            if sample.returncode:
                raise RuntimeError(sample.stderr)
            used = float(sample.stdout.strip().splitlines()[0])
            if used < 19000:
                continue
            reason = f'GPU memory reached {used} MiB (limit 19000)'
        except (subprocess.TimeoutExpired, RuntimeError, ValueError, IndexError) as exc:
            reason = f'GPU memory monitor unavailable: {exc}'
        save(path, {'state': 'STOPPED_BY_GPU_GUARD', 'reason': reason,
                    'observed_utc': datetime.now(timezone.utc).isoformat()})
        subprocess.run(['docker', 'kill', NAME], capture_output=True, timeout=10)
        return


def main():
    work = ROOT / 'data/track3/pilot/episode_000038'
    reports = ROOT / 'incident_reports/track3_baseline/episode_000038'
    lock = (work / 'pilot.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    video = ROOT / 'data/v2d_challenge/track_3/evaluation/videos/chunk-000/observation.images.ego_cam_a/episode_000038.mp4'
    frames = work / 'frames'
    frames.mkdir(exist_ok=True)
    expected = {f'{i:06d}.jpg' for i in range(263)}
    if {p.name for p in frames.glob('*.jpg')} != expected:
        cap = cv2.VideoCapture(str(video))
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            if not cv2.imwrite(str(frames / f'{count:06d}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError('Cannot save source frame')
            count += 1
        cap.release()
        if count != 263: raise ValueError('Source video frame mismatch')
    mano = ROOT / 'data/weights/track3_mano'
    (mano / 'models').mkdir(parents=True, exist_ok=True)
    source_mano = ROOT / 'data/weights/wilor/pretrained_models/MANO_RIGHT.pkl'
    target_mano = mano / 'models/MANO_RIGHT.pkl'
    if not target_mano.exists() or sha256(target_mano) != sha256(source_mano):
        shutil.copy2(source_mano, target_mano)
    detections = json.loads((work / 'wilor_raw/000000.json').read_text())
    if len(detections) != 2 or {bool(d['is_right']) for d in detections} != {True, False}:
        raise ValueError('Expected one source-person left/right hand in seed frame')
    tracks = work / 'hand_tracks.json'
    prompts = work / 'hand_prompts.json'
    save(tracks, {'reference_frame': 0, 'sam2_hand_prompt_source': 'wilor_mask',
                  'tracks': [{'object_id': i + 2, 'role': 'hand', 'is_right': d['is_right'], 'score': d['score']}
                             for i, d in enumerate(detections)]})
    seeds = work / 'hand_seed_masks'
    save(prompts, {'prompts': [{'object_id': i + 2, 'frame_index': 0,
                              'mask_path': f'hand_seed_masks/{i + 2}/000000.png'}
                             for i, d in enumerate(detections)]})
    masks, matched = work / 'hand_masks', work / 'wilor_matched'
    aligned, filled = work / 'wilor_depth_aligned', work / 'wilor_depth_aligned_filled'
    intrinsics = work / 'intrinsics_stable.json'
    state = {'state': 'RUNNING', 'episode_index': 38, 'hand_source': 'wilor',
             'pytorch_memory_fraction': .55, 'gpu_memory_guard_mib': 19000,
             'container_memory_limit_gib': 16, 'policy_trained': False, 'uploaded': False}
    save(reports / 'hand_pipeline.json', state)
    stop = threading.Event()
    threads = [threading.Thread(target=telemetry, args=(stop, reports / 'hand_pipeline_gpu.csv'), daemon=True),
               threading.Thread(target=guard, args=(stop, reports / 'hand_pipeline_guard.json'), daemon=True)]
    for thread in threads: thread.start()
    try:
        common = [Path(__file__), ROOT / 'scripts/track3_gpu_python.py', tracks, target_mano,
                  ROOT / 'modules/v2d_wilor/lib/tracks_interpolate.py']
        run_stage('hand_seed_masks', command('v2d_track3_hands:blackwell', 'v2d.wilor.lib.render_prompt_masks',
            ['--wilor_json', cpath(work / 'wilor_raw/000000.json'), '--image_path', cpath(work / 'reference.png'),
             '--output_dir', cpath(seeds), '--mano_assets_root', cpath(mano), '--first_object_id', '2']),
            [seeds / '2/000000.png', seeds / '3/000000.png'],
            [*common, work / 'wilor_raw/000000.json', work / 'reference.png'], reports)
        run_stage('hand_masks', command('v2d_sam2:blackwell', 'v2d.sam2.lib.video_to_masks',
            ['--video_path', cpath(video), '--prompts_path', cpath(prompts), '--masks_dir', cpath(masks),
             '--weights_dir', '/repo/data/weights/sam2']), [masks / '2', masks / '3'],
            [*common, video, prompts, seeds / '2/000000.png', seeds / '3/000000.png',
             ROOT / 'data/weights/sam2/sam2.1_hiera_large.pt'], reports)
        for track in ('2', '3'):
            if {p.stem for p in (masks / track).glob('*.png')} != {f'{i:06d}' for i in range(263)}:
                raise ValueError('Incomplete propagated hand-mask frames')
        run_stage('hand_association', command('v2d_track3_hands:blackwell', 'v2d.wilor.lib.tracks_from_wilor_masks',
            ['--frames_dir', cpath(frames), '--wilor_raw_dir', cpath(work / 'wilor_raw'),
             '--masks_dir', cpath(masks), '--tracks_path', cpath(tracks), '--output_dir', cpath(matched),
             '--mano_assets_root', cpath(mano), '--min_iou', '0.1']), [matched],
            [*common, *sorted((work / 'wilor_raw').glob('*.json')), *sorted(masks.glob('*/*.png'))], reports)
        run_stage('hand_depth_alignment', command('v2d_track3_hands:blackwell', 'v2d.hamer.lib.align_hands',
            ['--hamer_dir', cpath(matched), '--depth_dir', cpath(work / 'depth'),
             '--intrinsics_path', cpath(intrinsics), '--mano_assets_root', cpath(mano),
             '--output_dir', cpath(aligned), '--hand_masks_dir', cpath(masks),
             '--object_masks_dir', cpath(work / 'masks/1')]), [aligned],
            [*common, intrinsics, *sorted(matched.glob('*/*.json')), *sorted((work / 'depth').glob('*.png')),
             *sorted(masks.glob('*/*.png')), *sorted((work / 'masks/1').glob('*.png'))], reports)
        run_stage('hand_gap_fill', command('v2d_track3_hands:blackwell', 'v2d.wilor.lib.tracks_interpolate',
            ['--aligned_dir', cpath(aligned), '--masks_dir', cpath(masks), '--output_dir', cpath(filled),
             '--betas', 'fixed', '--max_gap_frames', '15', '--no_extrapolate']), [filled],
            [*common, *sorted(aligned.glob('*/*.json')), *sorted(masks.glob('*/*.png'))], reports)
        overlay = work / 'wilor_depth_aligned_overlay.mp4'
        run_stage('hand_alignment_overlay', command('v2d_track3_hands:blackwell', 'v2d.hamer.lib.render_hands_aligned_video',
            ['--frames_dir', cpath(frames), '--aligned_dir', cpath(filled), '--mano_assets_root', cpath(mano),
             '--output_path', cpath(overlay)]), [overlay],
            [*common, *sorted(filled.glob('*/*.json'))], reports)
        counts = {}
        for track in ('2', '3'):
            records = [json.loads(p.read_text()) for p in sorted((filled / track).glob('*.json'))]
            if not records:
                raise ValueError(f'No aligned records for hand track {track}; review masks')
            for record in records:
                if not np.isfinite(record['camera']['cam_t']).all():
                    raise ValueError('Nonfinite aligned hand translation')
            counts[track] = {'frames': len(records), 'interpolated': sum(bool(r.get('interpolated')) for r in records)}
        state.update(state='COMPLETE', tracks=counts, overlay=str(overlay.relative_to(ROOT)),
                     limitations=['Camera-frame hand estimates; world/gravity alignment and metric validation remain.',
                                  'HaMeR refinement is a later stage after its checkpoint download.'])
    except Exception as exc:
        state.update(state='FAILED', error=str(exc))
        raise
    finally:
        stop.set()
        for thread in threads: thread.join(timeout=10)
        state['observed_utc'] = datetime.now(timezone.utc).isoformat()
        save(reports / 'hand_pipeline.json', state)


if __name__ == '__main__':
    main()
