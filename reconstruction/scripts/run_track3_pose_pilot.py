#!/usr/bin/env python3
"""Track the reconstructed pilot body in the camera frame, with explicit scale.

These are reconstruction reference estimates. They are not world trajectories
or a learned-policy rollout and must not be packed as a submission.
"""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading

import numpy as np
import trimesh

from v2d.depth.lib.stabilize_intrinsics import stabilize_intrinsics
from prepare_track3_campaign import ROOT, sha256
from run_track3_object_pilot import cpath, run_stage, save, telemetry


def memory_watchdog(stop, report_path):
    """Stop this container before WSL exhausts physical GPU memory again."""
    while not stop.wait(.5):
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            used = float(result.stdout.strip().splitlines()[0])
            if used >= 19000:
                save(report_path, {'state': 'GPU_MEMORY_GUARD_TRIGGERED', 'used_mib': used,
                                   'limit_mib': 19000, 'observed_utc': datetime.now(timezone.utc).isoformat()})
                subprocess.run(['docker', 'kill', 'v2d-track3-pilot-tracking'], capture_output=True, timeout=10)
                return


def main():
    work = ROOT / 'data/track3/pilot/episode_000038'
    reports = ROOT / 'incident_reports/track3_baseline/episode_000038'
    lock = (work / 'pilot.lock').open('w')
    print('Waiting for the pilot GPU stage lock', flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX)
    source_mesh = work / 'mesh/mini_dust_pan.glb'
    source_transform = work / 'mesh/transform.json'
    scale = np.asarray(json.loads(source_transform.read_text())['scale'], dtype=float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any() or not np.allclose(scale, scale[0]):
        raise ValueError('Expected positive uniform SAM3D scene scale')
    scene = trimesh.load(source_mesh, force='scene')
    mesh = scene.to_geometry()
    mesh.apply_scale(float(scale[0]))
    metric_mesh = work / 'mesh/mini_dust_pan_metric.glb'
    mesh.export(metric_mesh)
    ledger = {'source_glb_sha256': sha256(source_mesh), 'source_transform_sha256': sha256(source_transform),
              'source_scene_nodes_baked': True, 'uniform_scale_applied': float(scale[0]),
              'sam3d_initial_rotation_baked': False, 'sam3d_initial_translation_baked': False,
              'body_frame': 'Original GLB scene frame with scene nodes baked and scale applied once; FoundationPose estimates camera_T_body for this mesh.',
              'extents_m': mesh.extents.tolist(), 'metric_mesh_sha256': sha256(metric_mesh),
              'metric_scale_independently_verified': False}
    save(work / 'mesh/body_transform_ledger.json', ledger)
    intrinsics = work / 'intrinsics_stable.json'
    stabilize_intrinsics(str(work / 'intrinsics'), str(intrinsics))
    video = ROOT / 'data/v2d_challenge/track_3/evaluation/videos/chunk-000/observation.images.ego_cam_a/episode_000038.mp4'
    poses = work / 'poses_camera'
    command = ['docker', 'run', '--rm', '--name', 'v2d-track3-pilot-tracking',
               '--gpus', 'all', '--memory', '20g', '--memory-swap', '22g', '--cpus', '8',
               '--user', f'{os.getuid()}:{os.getgid()}',
               '--entrypoint', '/repo/scripts/track3_gpu_python.py',
               '-e', 'HOME=/tmp', '-e', 'OMP_NUM_THREADS=4', '-e', 'V2D_TORCH_MEMORY_FRACTION=0.55',
               '-v', f'{ROOT}:/repo', 'v2d_track3_foundation_pose:blackwell',
               '-m', 'v2d.foundation_pose.lib.run_video_to_poses',
               '--video_path', cpath(video), '--depth_folder', cpath(work / 'depth'),
               '--masks_folder', cpath(work / 'masks/1'), '--camera_intrinsics_path', cpath(intrinsics),
               '--mesh_path', cpath(metric_mesh), '--poses_dir', cpath(poses),
               '--weights_dir', '/repo/data/weights/foundation_pose', '--backend', 'nvidia_tensorrt',
               '--reference_frame', '0', '--reregister_iou_thresh', '0.3',
               '--target_width', '1014', '--target_height', '760']
    inputs = [video, metric_mesh, intrinsics, source_transform, Path(__file__),
              ROOT / 'scripts/track3_gpu_python.py',
              ROOT / 'modules/v2d_foundation_pose/lib/run_video_to_poses.py',
              ROOT / 'modules/v2d_foundation_pose/lib/foundation_pose_tracker.py',
              *sorted((work / 'depth').glob('*.png')), *sorted((work / 'masks/1').glob('*.png')),
              *sorted((ROOT / 'data/weights/foundation_pose/nvidia_tensorrt/deployable_v1.0').glob('*.onnx'))]
    state = {'state': 'CAMERA_OBJECT_TRACKING_RUNNING', 'episode_index': 38,
             'tracking_resolution': [1014, 760], 'torch_memory_fraction': .55,
             'gpu_memory_guard_mib': 19000, 'container_memory_limit_gib': 20,
             'pose_frame': 'OpenCV camera_T_metric_body; camera motion not removed',
             'policy_trained': False, 'submission_ready': False, 'uploaded': False}
    save(reports / 'tracking.json', state)
    stop = threading.Event()
    monitor = threading.Thread(target=telemetry, args=(stop, reports / 'tracking_gpu.csv'), daemon=True)
    monitor.start()
    guard = threading.Thread(target=memory_watchdog, args=(stop, reports / 'memory_guard.json'), daemon=True)
    guard.start()
    try:
        run_stage('poses_camera', command, [poses], inputs, reports)
        files = sorted(poses.glob('*.json'))
        if [p.stem for p in files] != [f'{i:06d}' for i in range(263)]:
            raise ValueError('Incomplete pose source-frame coverage')
        translations = []
        for path in files:
            pose = json.loads(path.read_text())
            q, t = np.asarray(pose['rotation']), np.asarray(pose['translation'])
            if q.shape != (4,) or t.shape != (3,) or not np.isfinite(q).all() or not np.isfinite(t).all():
                raise ValueError(f'Invalid pose: {path}')
            if abs(np.linalg.norm(q) - 1) > 1e-4 or not np.allclose(pose['scale'], [1, 1, 1]):
                raise ValueError(f'Invalid rotation/body scale: {path}')
            translations.append(t)
        translations = np.asarray(translations)
        state.update(state='CAMERA_OBJECT_TRACKING_COMPLETE', frames=263,
                     translation_min_m=translations.min(axis=0).tolist(),
                     translation_max_m=translations.max(axis=0).tolist(),
                     nonpositive_camera_depth_count=int((translations[:, 2] <= 0).sum()),
                     pose_json_quaternion_order='wxyz; convert explicitly to xyzw for eventual policy export',
                     body_transform_ledger=ledger,
                     limitations=['No held-out pose score; silhouette/temporal review still required.',
                                  'Raw upright video was not lens-rectified.',
                                  'Remove camera motion and validate metric scale before policy training.'])
    except Exception as exc:
        state.update(state='FAILED', error=str(exc))
        raise
    finally:
        stop.set(); monitor.join(timeout=5); guard.join(timeout=10)
        state['observed_utc'] = datetime.now(timezone.utc).isoformat()
        save(reports / 'tracking.json', state)


if __name__ == '__main__':
    main()
