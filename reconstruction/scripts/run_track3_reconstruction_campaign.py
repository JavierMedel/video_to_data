#!/usr/bin/env python3
"""Unattended video-derived Track 3 reconstruction, with per-stage provenance."""
import argparse
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
from PIL import Image, ImageDraw
import trimesh
from v2d.common.result_bundle import write_result_bundle
from v2d.depth.lib.stabilize_intrinsics import stabilize_intrinsics
from prepare_track3_campaign import ROOT, sha256
from run_track3_object_pilot import container, cpath, run_stage, save, telemetry
from run_track3_camera_batch import validate_poses

NAME = 'v2d-track3-reconstruction-stage'
LIMIT = 21000
FOUNDATION_POSE_TORCH_FRACTION = .45
PROMPTS = {'mini_dust_pan': 'yellow dustpan', 'mini_sweeper': 'small hand broom',
           'white_pot': 'white cooking pot', 'white_pot_lid': 'white pot lid',
           'wooden_spoon': 'wooden spoon', 'blue_cup': 'blue cup', 'beige_cup': 'beige cup',
           'plastic_dish_rack': 'plastic dish rack', 'water_pitcher': 'water pitcher',
           'wooden_piece_1': 'wooden block', 'wooden_piece_2': 'wooden block'}


def guard(stop, path):
    while not stop.wait(.5):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                    check=True, capture_output=True, text=True, timeout=5)
            used = float(result.stdout.strip().splitlines()[0])
            if used < LIMIT: continue
            reason = f'Total GPU memory {used} MiB >= {LIMIT}'
        except (subprocess.SubprocessError, ValueError, IndexError) as exc:
            reason = f'GPU monitoring failed: {exc}'
        save(path, {'state': 'STOPPED_BY_GPU_GUARD', 'reason': reason})
        subprocess.run(['docker', 'kill', NAME], capture_output=True, timeout=10)
        return


def stage(name, image, module, args, outputs, inputs, report, fraction=.70):
    # FoundationPose also creates substantial TensorRT/native allocations.
    # Keep PyTorch lower so their combined peak remains below the 21 GiB guard.
    if module.startswith('v2d.foundation_pose.'):
        fraction = FOUNDATION_POSE_TORCH_FRACTION
    command = container(image, module, args)
    # FoundationPose's image contains compiled mycpp extensions. A host lib
    # mount hides those binaries; use the validated image implementation.
    if module.startswith('v2d.foundation_pose.'):
        mount = f'{ROOT}/modules/v2d_foundation_pose/lib:/workspace/v2d_foundation_pose/lib:ro'
        at = command.index(mount)
        del command[at-1:at+1]
    command[command.index('--entrypoint') + 1] = '/repo/scripts/track3_gpu_python.py'
    command[2:2] = ['--name', NAME, '--memory', '20g', '--memory-swap', '22g', '--cpus', '8',
                    '-e', f'V2D_TORCH_MEMORY_FRACTION={fraction}']
    stop = threading.Event()
    threads = [threading.Thread(target=telemetry, args=(stop, report / 'gpu.csv'), daemon=True),
               threading.Thread(target=guard, args=(stop, report / 'memory_guard.json'), daemon=True)]
    gpu_lock = (ROOT / 'data/track3/pilot/episode_000038/pilot.lock').open('a')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX)
    try:
        for thread in threads: thread.start()
        run_stage(name, command, outputs, [Path(__file__), ROOT / 'scripts/track3_gpu_python.py', *inputs], report)
    finally:
        stop.set()
        for thread in threads: thread.join(timeout=10)
        gpu_lock.close()


def coverage(folder, suffix, count):
    if {p.stem for p in folder.glob('*' + suffix)} != {f'{i:06d}' for i in range(count)}:
        raise ValueError(f'Incomplete frame coverage: {folder}')


def decode(video, folder, count):
    folder.mkdir(parents=True, exist_ok=True)
    if {p.stem for p in folder.glob('*.jpg')} == {f'{i:06d}' for i in range(count)}: return
    cap = cv2.VideoCapture(str(video))
    try:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            if not cv2.imwrite(str(folder / f'{i:06d}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise ValueError('Frame export failed')
            i += 1
        if i != count: raise ValueError('Decoded source-frame count mismatch')
    finally: cap.release()


def choose_detection(detections, selected):
    for detection in sorted(detections, key=lambda d: d['confidence'], reverse=True):
        b = detection['box']
        area = (b['x1'] - b['x0']) * (b['y1'] - b['y0'])
        if area < 100 or detection['confidence'] < .30: continue
        duplicate = False
        for other in selected:
            c = other['box']
            intersection = max(0, min(b['x1'], c['x1']) - max(b['x0'], c['x0'])) * max(0, min(b['y1'], c['y1']) - max(b['y0'], c['y0']))
            union = area + (c['x1'] - c['x0']) * (c['y1'] - c['y0']) - intersection
            if intersection / max(union, 1) > .75: duplicate = True
        if not duplicate: return detection
    raise ValueError('No distinct confident detection; object identity needs review')


def reconstruct_object(episode, name, slot, work, video, report, selected):
    count = episode['frames']
    obj = work / 'objects' / f'{slot:02d}_{name}'
    obj.mkdir(parents=True, exist_ok=True)
    # Try multiple source views; never substitute development scans or another object.
    reference = None
    for frame in sorted(set([0, count // 3, 2 * count // 3, count - 1])):
        image = work / 'frames' / f'{frame:06d}.jpg'
        detections = obj / f'detections_{frame:06d}.json'
        stage(f'object_{slot}_detect_{frame}', 'v2d_grounding_dino:blackwell',
              'v2d.grounding_dino.lib.image_to_object_bboxes',
              ['--image_path', cpath(image), '--output_path', cpath(detections), '--prompt', PROMPTS[name] + '.',
               '--box_threshold', '0.30', '--model_dir', '/repo/data/weights/grounding_dino'], [detections],
              [image, ROOT / 'data/weights/grounding_dino/groundingdino_swint_ogc.pth'], report)
        try:
            detection = choose_detection(json.loads(detections.read_text()), selected.get(frame, []))
        except ValueError: continue
        reference = frame
        selected.setdefault(frame, []).append(detection)
        break
    if reference is None: raise ValueError(f'No usable object detection for {name}')
    review = Image.open(image).convert('RGB'); ImageDraw.Draw(review).rectangle(
        tuple(detection['box'][key] for key in ['x0', 'y0', 'x1', 'y1']), outline='red', width=5)
    review.save(obj / 'detection_review.jpg')
    save(obj / 'identity.json', {'object': name, 'scored_slot': slot, 'reference_frame': reference,
        'detection': detection, 'identity_review': 'automated prompt/confidence/distinct-box gate; visual review pending'})
    prompts = obj / 'prompts.json'
    save(prompts, {'prompts': [{'frame_index': reference, 'object_id': slot + 1, 'box': detection['box']}]})
    masks = work / 'object_masks'
    mask_dir = masks / str(slot + 1)
    stage(f'object_{slot}_masks', 'v2d_sam2:blackwell', 'v2d.sam2.lib.video_to_masks',
          ['--video_path', cpath(video), '--prompts_path', cpath(prompts), '--masks_dir', cpath(masks),
           '--weights_dir', '/repo/data/weights/sam2'], [mask_dir],
          [video, prompts, ROOT / 'data/weights/sam2/sam2.1_hiera_large.pt'], report)
    coverage(mask_dir, '.png', count)
    mask = mask_dir / f'{reference:06d}.png'
    if (np.asarray(Image.open(mask)) > 0).sum() < 100: raise ValueError('Empty object mask')
    mesh, transform, mesh_k = obj / 'mesh.glb', obj / 'transform.json', obj / 'mesh_intrinsics.json'
    depth = work / 'depth' / f'{reference:06d}.png'
    intrinsics = work / 'intrinsics' / f'{reference:06d}.json'
    stage(f'object_{slot}_mesh', 'v2d_sam3d:blackwell', 'v2d.sam3d.lib.image_to_mesh',
          ['--image_path', cpath(image), '--mask_path', cpath(mask), '--mesh_path', cpath(mesh),
           '--transform_path', cpath(transform), '--intrinsics_path', cpath(mesh_k),
           '--weights_dir', '/repo/data/weights/sam3d', '--depth_path', cpath(depth),
           '--depth_intrinsics_path', cpath(intrinsics), '--depth_mask_path', cpath(mask), '--seed', '42'],
          [mesh, transform, mesh_k], [image, mask, depth, intrinsics], report, fraction=.80)
    scale = np.asarray(json.loads(transform.read_text())['scale'])
    if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any() or not np.allclose(scale, scale[0]):
        raise ValueError('Invalid reconstructed mesh scale')
    geometry = trimesh.load(mesh, force='scene').to_geometry()
    geometry.apply_scale(float(scale[0]))
    if not len(geometry.faces) or not np.isfinite(geometry.vertices).all(): raise ValueError('Invalid mesh')
    metric = obj / 'mesh_metric.glb'; geometry.export(metric)
    save(obj / 'body_transform_ledger.json', {'source_glb_sha256': sha256(mesh),
        'source_transform_sha256': sha256(transform), 'uniform_scale_applied_once': float(scale[0]),
        'metric_mesh_sha256': sha256(metric), 'extents_m': geometry.extents.tolist(),
        'metric_scale_independently_verified': False, 'body_frame': 'baked GLB scene, scaled once'})
    poses = obj / 'poses_camera'
    stage(f'object_{slot}_poses', 'v2d_track3_foundation_pose:blackwell', 'v2d.foundation_pose.lib.run_video_to_poses',
          ['--video_path', cpath(video), '--depth_folder', cpath(work / 'depth'), '--masks_folder', cpath(mask_dir),
           '--camera_intrinsics_path', cpath(work / 'intrinsics_stable.json'), '--mesh_path', cpath(metric),
           '--poses_dir', cpath(poses), '--weights_dir', '/repo/data/weights/foundation_pose',
           '--backend', 'nvidia_tensorrt', '--reference_frame', str(reference), '--reregister_iou_thresh', '0.3',
           '--target_width', '1014', '--target_height', '760'], [poses],
          [video, metric, work / 'intrinsics_stable.json', *sorted(mask_dir.glob('*.png')),
           *sorted((work / 'depth').glob('*.png'))], report, fraction=.55)
    validate_poses(poses, count)
    return obj, metric, poses


def reconstruct_hands(episode, work, video, report):
    raw = work / 'wilor_raw'
    stage('wilor_raw', 'v2d_track3_hands:blackwell', 'v2d.wilor.lib.video_to_hands',
          ['--video_path', cpath(video), '--output_dir', cpath(raw), '--weights_dir', '/repo/data/weights/wilor'],
          [raw], [video, ROOT / 'data/weights/wilor/pretrained_models/MANO_RIGHT.pkl'], report)
    coverage(raw, '.json', episode['frames'])
    seed, detections = None, None
    for path in sorted(raw.glob('*.json')):
        records = json.loads(path.read_text())
        chosen = [max((d for d in records if bool(d['is_right']) == side), key=lambda d: d['score'], default=None)
                  for side in (False, True)]
        if all(d is not None and d['score'] >= .3 for d in chosen):
            seed, detections = int(path.stem), chosen
            break
    if seed is None: raise ValueError('No confident two-hand seed; preserve missing hands for review')
    seed_json = work / 'hand_seed.json'; save(seed_json, detections)
    tracks = work / 'hand_tracks.json'
    save(tracks, {'reference_frame': seed, 'tracks': [
        {'object_id': 100 + i, 'role': 'hand', 'is_right': bool(d['is_right']), 'score': d['score']}
        for i, d in enumerate(detections)]})
    seeds, masks, prompts = work / 'hand_seed_masks', work / 'hand_masks', work / 'hand_prompts.json'
    stage('hand_seeds', 'v2d_track3_hands:blackwell', 'v2d.wilor.lib.render_prompt_masks',
          ['--wilor_json', cpath(seed_json), '--image_path', cpath(work / 'frames' / f'{seed:06d}.jpg'),
           '--output_dir', cpath(seeds), '--mano_assets_root', '/repo/data/weights/track3_mano',
           '--first_object_id', '100', '--frame_index', str(seed), '--min_mask_pixels', '100'],
          [seeds / '100', seeds / '101'], [seed_json, tracks], report)
    save(prompts, {'prompts': [{'object_id': 100 + i, 'frame_index': seed,
         'mask_path': f'hand_seed_masks/{100+i}/{seed:06d}.png'} for i in range(2)]})
    stage('hand_masks', 'v2d_sam2:blackwell', 'v2d.sam2.lib.video_to_masks',
          ['--video_path', cpath(video), '--prompts_path', cpath(prompts), '--masks_dir', cpath(masks),
           '--weights_dir', '/repo/data/weights/sam2'], [masks / '100', masks / '101'],
          [video, prompts, *sorted(seeds.glob('*/*.png'))], report)
    for oid in ('100', '101'): coverage(masks / oid, '.png', episode['frames'])
    hamer, aligned, filled = work / 'hamer_raw', work / 'hamer_depth_aligned', work / 'hamer_depth_aligned_filled'
    stage('hamer', 'v2d_track3_hands:blackwell', 'v2d.hamer.lib.masks_to_hands',
          ['--frames_dir', cpath(work / 'frames'), '--masks_dir', cpath(masks), '--tracks_path', cpath(tracks),
           '--output_dir', cpath(hamer), '--weights_dir', '/repo/data/weights/hamer'], [hamer],
          [tracks, *sorted(masks.glob('*/*.png')), ROOT / 'data/weights/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt'], report)
    stage('hamer_alignment', 'v2d_track3_hands:blackwell', 'v2d.hamer.lib.align_hands',
          ['--hamer_dir', cpath(hamer), '--depth_dir', cpath(work / 'depth'),
           '--intrinsics_path', cpath(work / 'intrinsics_stable.json'), '--mano_assets_root', '/repo/data/weights/track3_mano',
           '--output_dir', cpath(aligned), '--hand_masks_dir', cpath(masks)], [aligned],
          [work / 'intrinsics_stable.json', *sorted(hamer.glob('*/*.json')), *sorted((work / 'depth').glob('*.png'))], report)
    stage('hamer_gap_fill', 'v2d_track3_hands:blackwell', 'v2d.wilor.lib.tracks_interpolate',
          ['--aligned_dir', cpath(aligned), '--masks_dir', cpath(masks), '--output_dir', cpath(filled),
           '--betas', 'fixed', '--max_gap_frames', '15', '--no_extrapolate'], [filled],
          [*sorted(aligned.glob('*/*.json')), tracks], report)
    return filled / '100', filled / '101'


def package(episode, work, objects, left, right):
    bundles = []
    for slot, name, obj, mesh, poses in objects:
        out = ROOT / f'data/track3/bundles/episode_{episode["episode_index"]:06d}_object_{slot:02d}/outputs/result'
        export_mesh = mesh.with_name('bundle_mesh.obj')
        trimesh.load(mesh, force='scene').to_geometry().export(export_mesh)
        write_result_bundle(str(out), str(work / 'frames'), str(work / 'intrinsics_stable.json'), str(export_mesh),
             object_poses_dir=str(poses), camera_to_world_dir=str(work / 'camera_poses_slam'),
             left_hand_dir=str(left), right_hand_dir=str(right),
             source_manifest={'episode_index': episode['episode_index'], 'object_slot': slot,
                              'object_name': name, 'fps': episode['videos']['ego_cam_a']['fps'],
                              'reference_only': True, 'policy_trained': False})
        with np.load(out / 'result.npz', allow_pickle=False) as data:
            if data['camera_to_world_transform'].shape[0] != episode['frames'] or not data['object_is_valid'].all():
                raise ValueError('Incomplete bundle frame mapping')
        bundles.append(str(out.relative_to(ROOT)))
    save(work / 'reconstruction_manifest.json', {'state': 'REFERENCE_RECONSTRUCTION_COMPLETE',
        'episode_index': episode['episode_index'], 'frames': episode['frames'], 'bundles': bundles,
        'source_fps': episode['videos']['ego_cam_a']['fps'], 'policy_trained': False,
        'submission_ready': False, 'metric_scale_independently_verified': False,
        'visual_identity_review_complete': episode['episode_index'] == 38})
    return bundles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, nargs='+')
    args = parser.parse_args()
    reports = ROOT / 'incident_reports/track3_baseline/reconstruction_campaign'; reports.mkdir(parents=True, exist_ok=True)
    lock = (reports / 'campaign.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inventory = json.loads((ROOT / 'incident_reports/track3_baseline/inputs.json').read_text())
    episodes = sorted(inventory['episodes'], key=lambda e: (e['episode_index'] != 38, e['frames']))
    if args.episodes: episodes = [e for e in episodes if e['episode_index'] in args.episodes]
    state = {'state': 'RUNNING', 'completed_episode_indices': [], 'failed_episodes': {}, 'policy_trained': False,
             'submission_ready': False, 'uploaded': False, 'gpu_memory_guard_mib': LIMIT}
    for episode in episodes:
        index = episode['episode_index']; state.update(active_episode_index=index, observed_utc=datetime.now(timezone.utc).isoformat())
        save(reports / 'execution.json', state)
        work = ROOT / ('data/track3/pilot/episode_000038' if index == 38 else f'data/track3/episodes/episode_{index:06d}')
        report = reports / f'episode_{index:06d}'; report.mkdir(parents=True, exist_ok=True)
        try:
            video = ROOT / episode['videos']['ego_cam_a']['path']
            if sha256(video) != episode['videos']['ego_cam_a']['sha256']: raise ValueError('Changed source video')
            decode(video, work / 'frames', episode['frames'])
            validate_poses(work / 'camera_poses_slam', episode['frames'])
            stabilize_intrinsics(str(work / 'intrinsics'), str(work / 'intrinsics_stable.json'))
            if index == 38:
                objects = [(0, 'mini_dust_pan', work / 'mesh', work / 'mesh/mini_dust_pan_metric.glb', work / 'poses_camera')]
                tracks = json.loads((work / 'hand_tracks.json').read_text())['tracks']
                left = work / 'hamer_depth_aligned_filled' / str(next(t['object_id'] for t in tracks if not t['is_right']))
                right = work / 'hamer_depth_aligned_filled' / str(next(t['object_id'] for t in tracks if t['is_right']))
            else:
                objects, selected = [], {}
                for slot, name in enumerate(episode['objects_in_scored_order']):
                    obj, mesh, poses = reconstruct_object(episode, name, slot, work, video, report, selected)
                    objects.append((slot, name, obj, mesh, poses))
                left, right = reconstruct_hands(episode, work, video, report)
            bundles = package(episode, work, objects, left, right)
            state['completed_episode_indices'].append(index)
            print('REFERENCE_COMPLETE', index, bundles, flush=True)
        except Exception as exc:
            state['failed_episodes'][str(index)] = str(exc)
            print('EPISODE_FAILED', index, f'{type(exc).__name__}: {str(exc)[:300]}', flush=True)
            if (report / 'memory_guard.json').exists():
                state['state'] = 'STOPPED_BY_GPU_GUARD'; save(reports / 'execution.json', state); raise
        save(reports / 'execution.json', state)
    state.update(state='COMPLETE' if not state['failed_episodes'] else 'PARTIAL', active_episode_index=None,
                 observed_utc=datetime.now(timezone.utc).isoformat())
    save(reports / 'execution.json', state)


if __name__ == '__main__': main()
