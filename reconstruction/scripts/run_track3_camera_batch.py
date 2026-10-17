#!/usr/bin/env python3
"""Resume Track 3 camera motion after depth; outputs are reference estimates."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
from v2d.depth.lib.stabilize_intrinsics import stabilize_intrinsics
from prepare_track3_campaign import ROOT, sha256
from run_track3_object_pilot import cpath, run_stage, save, telemetry

IMAGE = 'v2d_track3_droid_slam:blackwell'
NAME = 'v2d-track3-camera-stage'
GPU_LIMIT_MIB = 21000


def guard(stop, path):
    while not stop.wait(.5):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=5, check=True)
            used = float(result.stdout.strip().splitlines()[0])
            if used < GPU_LIMIT_MIB:
                continue
            reason = f'Total GPU memory {used} MiB >= {GPU_LIMIT_MIB}'
        except (subprocess.SubprocessError, ValueError, IndexError) as exc:
            reason = f'GPU monitoring failed: {exc}'
        save(path, {'state': 'STOPPED_BY_GPU_GUARD', 'reason': reason})
        subprocess.run(['docker', 'kill', NAME], capture_output=True, timeout=10)
        return


def validate_poses(folder, frames):
    paths = sorted(folder.glob('*.json'))
    if [p.stem for p in paths] != [f'{i:06d}' for i in range(frames)]:
        raise ValueError('Camera poses do not cover every source frame')
    positions = []
    for path in paths:
        record = json.loads(path.read_text())
        q, t = np.asarray(record['rotation']), np.asarray(record['translation'])
        if q.shape != (4,) or t.shape != (3,) or not np.isfinite(q).all() or not np.isfinite(t).all():
            raise ValueError(f'Invalid camera pose {path}')
        if abs(np.linalg.norm(q) - 1) > 1e-4 or not np.allclose(record['scale'], [1, 1, 1]):
            raise ValueError(f'Invalid camera rotation/scale {path}')
        positions.append(t)
    positions = np.asarray(positions)
    return {'frames': frames, 'translation_range_m': np.ptp(positions, axis=0).tolist(),
            'largest_translation_step_m': float(np.linalg.norm(np.diff(positions, axis=0), axis=1).max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for-build-pid', type=int)
    args = parser.parse_args()
    reports = ROOT / 'incident_reports/track3_baseline/camera_batch'
    reports.mkdir(parents=True, exist_ok=True)
    lock = (reports / 'batch.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'state': 'PREPARING', 'completed_episode_indices': [], 'policy_trained': False,
             'submission_ready': False, 'uploaded': False, 'pytorch_memory_fraction': .70,
             'gpu_memory_guard_mib': GPU_LIMIT_MIB, 'camera_resolution': [512, 384],
             'limitations': ['Raw upright RGB; distortion unrectified.',
                            'MoGe-depth scale prior, not independently verified metric scale.',
                            'SLAM world is arbitrary and not gravity/support aligned.',
                            'Camera motion must pass visual/temporal review before retargeting.']}
    def update(**values):
        state.update(**values, observed_utc=datetime.now(timezone.utc).isoformat())
        save(reports / 'execution.json', state)
    try:
        update(state='WAITING_FOR_BLACKWELL_IMAGE')
        if args.wait_for_build_pid:
            deadline = time.monotonic() + 3600
            while Path(f'/proc/{args.wait_for_build_pid}').exists():
                if time.monotonic() > deadline:
                    raise TimeoutError('DROID image build exceeded one hour')
                time.sleep(5)
        subprocess.run(['docker', 'image', 'inspect', IMAGE], check=True, stdout=subprocess.DEVNULL)
        checkpoint = ROOT / 'data/weights/droid_slam/droid.pth'
        if not checkpoint.exists() or checkpoint.stat().st_size < 1000000:
            raise ValueError('Verified DROID checkpoint is missing')
        update(state='QUEUED_FOR_GPU', checkpoint_sha256=sha256(checkpoint))
        inventory_path = ROOT / 'incident_reports/track3_baseline/inputs.json'
        inventory = json.loads(inventory_path.read_text())
        episodes = sorted(inventory['episodes'], key=lambda e: (e['episode_index'] != 38, e['frames']))
        gpu_lock = (ROOT / 'data/track3/pilot/episode_000038/pilot.lock').open('w')
        for episode in episodes:
            index = episode['episode_index']
            fcntl.flock(gpu_lock, fcntl.LOCK_EX)
            try:
                update(state='RUNNING', active_episode_index=index)
                work = ROOT / ('data/track3/pilot/episode_000038' if index == 38 else f'data/track3/episodes/episode_{index:06d}')
                depth = work / 'depth'
                if {p.stem for p in depth.glob('*.png')} != {f'{i:06d}' for i in range(episode['frames'])}:
                    raise ValueError(f'Depth is incomplete for episode {index}')
                video = ROOT / episode['videos']['ego_cam_a']['path']
                if sha256(video) != episode['videos']['ego_cam_a']['sha256']:
                    raise ValueError(f'Source video changed: {index}')
                intrinsics = work / 'camera_slam_intrinsics.json'
                stabilize_intrinsics(str(work / 'intrinsics'), str(intrinsics))
                poses = work / 'camera_poses_slam'
                trajectory = work / 'camera_trajectory_slam.txt'
                command = ['docker', 'run', '--rm', '--name', NAME, '--gpus', 'all',
                           '--memory', '20g', '--memory-swap', '22g', '--cpus', '8',
                           '--user', f'{os.getuid()}:{os.getgid()}', '--entrypoint', '/repo/scripts/track3_gpu_python.py',
                           '-e', 'HOME=/tmp', '-e', 'OMP_NUM_THREADS=4', '-e', 'OPENBLAS_NUM_THREADS=4',
                           '-e', 'V2D_TORCH_MEMORY_FRACTION=0.70', '-v', f'{ROOT}:/repo',
                           IMAGE, '-m', 'v2d.droid_slam.lib.video_to_slam', '--video_path', cpath(video),
                           '--poses_folder', cpath(poses), '--weights_path', cpath(checkpoint),
                           '--input_intrinsics_path', cpath(intrinsics), '--input_depth_folder', cpath(depth),
                           '--trajectory_path', cpath(trajectory), '--image_height', '384', '--image_width', '512',
                           '--buffer_size', '384', '--stride', '1']
                stop = threading.Event()
                threads = [threading.Thread(target=telemetry, args=(stop, reports / 'gpu.csv'), daemon=True),
                           threading.Thread(target=guard, args=(stop, reports / 'memory_guard.json'), daemon=True)]
                for thread in threads: thread.start()
                try:
                    run_stage(f'episode_{index:06d}', command, [poses, trajectory],
                              [Path(__file__), ROOT / 'scripts/track3_gpu_python.py', inventory_path,
                               ROOT / 'modules/v2d_droid_slam/lib/_slam.py',
                               ROOT / 'modules/v2d_droid_slam/lib/video_to_slam.py',
                               video, intrinsics, checkpoint, *sorted(depth.glob('*.png'))], reports)
                    summary = validate_poses(poses, episode['frames'])
                    save(work / 'camera_slam_manifest.json', {**summary, 'episode_index': index,
                         'pose_convention': 'world_T_camera, quaternion wxyz, arbitrary SLAM world',
                         'source_fps': episode['videos']['ego_cam_a']['fps'],
                         'metric_scale_independently_verified': False, 'gravity_aligned': False,
                         'provenance_report': str((reports / f'episode_{index:06d}.json').relative_to(ROOT))})
                    state['completed_episode_indices'].append(index)
                    print('CAMERA_COMPLETE', index, summary, flush=True)
                finally:
                    stop.set()
                    for thread in threads: thread.join(timeout=10)
            finally:
                fcntl.flock(gpu_lock, fcntl.LOCK_UN)
        update(state='COMPLETE', active_episode_index=None)
    except Exception as exc:
        update(state='FAILED', error=str(exc))
        raise


if __name__ == '__main__':
    main()
