"""Create isolated object-only candidates using observed video evidence.

Human parameters are copied exactly from the scored v3 submission. Mesh-scored
orientation selection is followed by a bounded multi-frame geometry probe;
cached monocular depth is diagnostic evidence, not hidden reference geometry.
"""
import argparse
import hashlib
import json
import pickle
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
import trimesh
from scipy.signal import savgol_filter

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, default=Path('/repo'))
p.add_argument('--episodes', type=int, nargs='+', default=[12, 14, 2, 1, 19, 20])
p.add_argument('--iteration', type=int, choices=[4, 5], default=4)
a = p.parse_args()
sys.path.insert(0, '/workspace/v2d_foundation_pose/lib/FoundationPose')
sys.path.insert(0, str(a.root / 'modules/v2d_cari4d/lib/cari4d'))
from track1_rollout_common import paths
from prep.mhr_export_utils import read_depth_m, read_mask, read_rgb, camera_calibration, load_edex
from prep.run_foundationpose_mhr_export import _render_candidate_masks, _scaled_image_grid
from prep.foundationpose_pose_selection import measure_visible_silhouettes
from prep.temporal_pose_selection import rotation_hypotheses, select_temporal_rotations
from prep.trajectory_diagnostics import rigid_motion_diagnostics
import Utils

data = a.root / 'data/v2d_challenge/track_1'
experiment = f'cdo_v{a.iteration}'
control_label = f'submitted_v{a.iteration-1}'
control_dir = data / ('improvement_rollout_v1/candidate_episodes' if a.iteration == 4 else 'cdo_v4_experiments/candidate_episodes')
report = a.root / 'incident_reports/track1_improvement' / experiment
report.mkdir(parents=True, exist_ok=True)
scored = json.loads((a.root / 'incident_reports/track1_scored_frames.json').read_text())
ctx = dr.RasterizeCudaContext()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_renderer(vertices, faces):
    return SimpleNamespace(mesh_tensors={'pos': torch.as_tensor(vertices, dtype=torch.float32, device='cuda'),
                                       'faces': torch.as_tensor(faces, dtype=torch.int32, device='cuda')}, glctx=ctx)


def depth_render(renderer, pose, k, shape):
    vertices, faces = renderer.mesh_tensors['pos'], renderer.mesh_tensors['faces']
    pose = torch.as_tensor(pose, device='cuda', dtype=torch.float32)
    camera = vertices @ pose[:3, :3].T + pose[:3, 3]
    projection = torch.as_tensor(Utils.projection_matrix_from_intrinsics(k, height=shape[0], width=shape[1], znear=.001, zfar=100.), device='cuda', dtype=torch.float32)
    cv_to_gl = torch.as_tensor(Utils.glcam_in_cvcam, device='cuda', dtype=torch.float32)
    clip = Utils.to_homo_torch(camera) @ (projection @ cv_to_gl).T
    raster, _ = dr.rasterize(ctx, clip[None], faces, resolution=shape)
    z, _ = dr.interpolate(camera[None, :, 2:3].contiguous(), raster, faces)
    return torch.flip(z[0, :, :, 0], [0]).cpu().numpy(), torch.flip(raster[0, :, :, 3] > 0, [0]).cpu().numpy()


def observations(export, frame, k, depth_root=None, width=512):
    name = f'{frame:06d}'
    obj = read_mask(export, 'object', 0, name)
    human = read_mask(export, 'human', 0, name)
    small_k, shape = _scaled_image_grid(k, obj.shape, width)
    resize = lambda x: cv2.resize(x, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    result = {'obj': resize(obj.astype(np.uint8)) > 0, 'human': resize(human.astype(np.uint8)) > 0, 'k': small_k, 'shape': shape}
    if depth_root is not None:
        result['depth'] = resize(read_depth_m(export, 0, name, depth_root=depth_root))
    return result


def pose_from_npz(bundle):
    poses = np.tile(np.eye(4), (len(bundle['pose']), 1, 1))
    poses[:, :3, :3] = bundle['object_rotation']
    poses[:, :3, 3] = bundle['object_translation']
    return poses


def evaluate(renderer, poses, obs, w2c):
    rows, renders = [], {}
    for frame, item in obs.items():
        z, mask = depth_render(renderer, w2c @ poses[frame], item['k'], item['shape'])
        metrics = measure_visible_silhouettes(mask[None], item['human'], item['obj'])
        rows.append({'frame': int(frame), 'iou': float(metrics.iou[0])})
        renders[frame] = (z, mask)
    return rows, renders


def common_depth(parent_renders, candidate_renders, obs, frames):
    before, after, pixels = [], [], []
    for frame in frames:
        item = obs[int(frame)]
        pz, pm = parent_renders[int(frame)]; cz, cm = candidate_renders[int(frame)]
        support = pm & cm & item['obj'] & ~item['human'] & np.isfinite(item['depth']) & (item['depth'] > .05) & (pz > .05) & (cz > .05)
        pixels.append(int(support.sum()))
        if support.sum() >= 8:
            before.append(float(np.median(abs(pz[support] - item['depth'][support])) * 100.))
            after.append(float(np.median(abs(cz[support] - item['depth'][support])) * 100.))
    return {'before_cm': float(np.median(before)) if before else None,
            'after_cm': float(np.median(after)) if after else None,
            'supported_frames': len(before), 'common_support_pixels': pixels}


all_results = []
with torch.inference_mode():
    # The same depth renderer must return a front surface of 2.5m for this box.
    box = trimesh.creation.box(extents=[1., 1., 1.]); box_pose = np.eye(4); box_pose[2, 3] = 3.
    z, mask = depth_render(make_renderer(box.vertices, box.faces), box_pose, np.array([[80., 0, 48], [0, 80., 32], [0, 0, 1.]]), (64, 96))
    assert mask[32, 48] and abs(z[32, 48] - 2.5) < 1e-4
    for episode in a.episodes:
        q = paths(a.root, episode); stem = q['stem']; work = q['work']; export = work / 'run/export' / stem
        if episode in (6, 18, 28):
            work = data / 'improvement_pilot_v1' / stem
            export = work / 'run/export' / stem
        out = data / f'{experiment}_experiments/pose_candidates' / stem
        if (out / 'review.json').exists():
            all_results.append(json.loads((out / 'review.json').read_text()))
            print('TEMPORAL_CANDIDATE_REUSED', episode, flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        v3_path = control_dir / (stem + '.npz')
        old_hash = sha(v3_path)
        v3 = dict(np.load(v3_path, allow_pickle=False))
        parent = dict(np.load(work / 'submission_episodes' / (stem + '.npz'), allow_pickle=False))
        poses = pose_from_npz(parent)
        tracker = pickle.loads((work / 'run/foundationpose' / (stem + '.pkl')).read_bytes())['obj_pose_world']
        mesh_path = work / 'submission_episodes' / (stem + '_object.glb')
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        vertices = mesh.vertices * float(parent['object_scale'])
        renderer = make_renderer(vertices, mesh.faces)
        k, w2c = camera_calibration(load_edex(export), 0)
        hypotheses = rotation_hypotheses(poses[:, :3, :3], tracker[:, :3, :3])
        costs = np.zeros(hypotheses.shape[:2])
        for frame in range(len(poses)):
            item = observations(export, frame, k)
            candidates = np.tile(poses[frame], (hypotheses.shape[1], 1, 1))
            candidates[:, :3, :3] = hypotheses[frame]
            masks = _render_candidate_masks(renderer, w2c[None] @ candidates, item['k'], item['shape'])
            metrics = measure_visible_silhouettes(masks, item['human'], item['obj'])
            if metrics.observed_visible_pixels >= 8:
                costs[frame] = 1. - metrics.iou
            costs[frame] += np.arange(len(candidates)) * 1e-6
            if frame % 100 == 0:
                print('TEMPORAL_ROTATION_PROGRESS', episode, frame, len(poses), flush=True)
        selected, path, objective = select_temporal_rotations(hypotheses, costs)
        temporal = poses.copy(); temporal[:, :3, :3] = selected
        np.savez_compressed(out / 'rotation_selection.npz', hypothesis_indices=path, costs=costs, selected_rotations=selected)
        frames = np.array(scored[str(episode)])
        frames = frames[np.linspace(0, len(frames)-1, min(33, len(frames))).round().astype(int)]
        obs = {int(f): observations(export, int(f), k, q['depth'], width=768) for f in frames}
        parent_rows, parent_renders = evaluate(renderer, poses, obs, w2c)
        temporal_rows, temporal_renders = evaluate(renderer, temporal, obs, w2c)
        # Smooth translation only if independent sampled silhouette evidence
        # remains stable. Pose orientation is never smoothed without scoring.
        smoothed = temporal.copy()
        smoothed[:, :3, 3] = savgol_filter(temporal[:, :3, 3], 7, 2, axis=0)
        smooth_rows, smooth_renders = evaluate(renderer, smoothed, obs, w2c)
        smooth_accept = np.mean([r['iou'] for r in smooth_rows]) >= np.mean([r['iou'] for r in temporal_rows]) - .015
        base, base_rows, base_renders = (smoothed, smooth_rows, smooth_renders) if smooth_accept else (temporal, temporal_rows, temporal_renders)
        # Fit shared geometry/depth factors on 22 frames and check 11 disjoint
        # frames. Depth comparisons use IDENTICAL pixels for both predictions.
        fit = frames[np.arange(len(frames)) % 3 != 2]; held = frames[np.arange(len(frames)) % 3 == 2]
        grid = []; best = None
        for scale in (.90, 1., 1.10):
            trial_renderer = make_renderer(vertices * scale, mesh.faces)
            for offset in (-.10, -.05, 0., .05, .10):
                trial = base.copy(); camera = w2c[None] @ trial
                old_z = camera[:, 2, 3].copy()
                if (old_z + offset <= .1).any():
                    continue
                camera[:, :3, 3] *= ((old_z + offset) / old_z)[:, None]
                trial = np.linalg.inv(w2c)[None] @ camera
                rows, renders = evaluate(trial_renderer, trial, obs, w2c)
                by_frame = {r['frame']: r['iou'] for r in rows}; base_iou = {r['frame']: r['iou'] for r in base_rows}
                depth = common_depth(base_renders, renders, obs, fit)
                fit_iou = float(np.mean([by_frame[int(f)] for f in fit]))
                base_fit_iou = float(np.mean([base_iou[int(f)] for f in fit]))
                eligible = depth['supported_frames'] >= 8 and fit_iou >= base_fit_iou - .015
                gain = fit_iou - base_fit_iou
                if eligible:
                    gain += .15 * np.clip((depth['before_cm'] - depth['after_cm']) / 10., -1., 1.)
                row = {'scale_factor': scale, 'depth_offset_m': offset, 'fit_iou': fit_iou, 'fit_common_depth': depth, 'fit_objective_gain': float(gain), 'eligible': bool(eligible)}
                grid.append(row)
                if eligible and (best is None or gain > best[0] + 1e-8):
                    best = (gain, scale, offset, trial, rows, renders)
        geometry_accepted = False
        final, final_scale, final_rows = base, 1., base_rows
        geometry_review = {'selected_on_fit_only': True, 'accepted': False}
        if best is not None and best[0] > .005:
            _, scale, offset, trial, rows, renders = best
            test_depth = common_depth(base_renders, renders, obs, held)
            by_frame = {r['frame']: r['iou'] for r in rows}; base_iou = {r['frame']: r['iou'] for r in base_rows}
            test_iou = float(np.mean([by_frame[int(f)] for f in held])); base_test_iou = float(np.mean([base_iou[int(f)] for f in held]))
            geometry_accepted = test_depth['supported_frames'] >= 4 and test_iou >= base_test_iou - .015 and test_depth['after_cm'] <= test_depth['before_cm']
            geometry_review.update(scale_factor=scale, depth_offset_m=offset, held_out_iou_before=base_test_iou, held_out_iou_after=test_iou, held_out_common_depth=test_depth, accepted=bool(geometry_accepted))
            if geometry_accepted:
                final, final_scale, final_rows = trial, scale, rows
        result = dict(v3)
        result.update(object_rotation=final[:, :3, :3].astype(np.float32), object_translation=final[:, :3, 3].astype(np.float32), object_scale=np.array(float(parent['object_scale']) * final_scale, dtype=np.float32))
        for key in ('pose', 'scales', 'shape'):
            assert np.array_equal(result[key], v3[key]), key
        assert sha(v3_path) == old_hash and np.isfinite(final).all() and ((w2c[None] @ final)[:, 2, 3] > .1).all()
        np.savez_compressed(out / (stem + '.npz'), **result)
        shutil.copy2(mesh_path, out / (stem + '_object.glb'))
        summary = {}
        panels = []
        old_mesh = trimesh.load(v3_path.with_name(stem + '_object.glb'), force='mesh', process=False)
        for label, v, faces, sequence in [(control_label, old_mesh.vertices * float(v3['object_scale']), old_mesh.faces, pose_from_npz(v3)),
                                          ('previous_rollout', vertices, mesh.faces, poses), ('temporal_rotation', vertices, mesh.faces, temporal),
                                          ('final_candidate', vertices * final_scale, mesh.faces, final)]:
            rend = make_renderer(v, faces); rows, _ = evaluate(rend, sequence, obs, w2c)
            summary[label] = {'visible_iou_mean': float(np.mean([r['iou'] for r in rows])),
                              'translation_acceleration_cm_proxy': float(np.linalg.norm(np.diff(sequence[:, :3, 3], n=2, axis=0), axis=-1).mean()*100),
                              **rigid_motion_diagnostics(sequence, v)}
            for frame in frames[np.linspace(0, len(frames)-1, 6).round().astype(int)]:
                item = obs[int(frame)]
                mask = _render_candidate_masks(rend, w2c[None] @ sequence[frame:frame+1], item['k'], item['shape'])[0]
                rgb = cv2.resize(read_rgb(export, 0, f'{frame:06d}'), (item['shape'][1], item['shape'][0]))
                rgb[item['obj']] = (rgb[item['obj']]*.6 + np.array([0, 255, 0])*.4).astype(np.uint8)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb, contours, -1, (255, 60, 60), 2)
                rgb = cv2.resize(rgb, (512, 384))
                cv2.putText(rgb, f'{label} f{frame}', (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
                panels.append(rgb)
        sheet = np.concatenate([np.concatenate(panels[j:j+6], axis=1) for j in range(0, len(panels), 6)], axis=0)
        cv2.imwrite(str(out / 'comparison.jpg'), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        review = {'episode': episode, 'state': 'CANDIDATE_COMPLETE_CONTACT_REVIEW_PENDING', 'ground_truth_used': False,
                  'human_parameters_identical_to_submitted_control': True, 'submitted_control_sha256': old_hash, 'control_version': a.iteration-1,
                  'rotation_objective': objective, 'motion_weight': .04, 'hypotheses': 'predicted/tracked/local-average x four mesh-scored orientations; no assumed symmetry',
                  'translation_smoothing_accepted': bool(smooth_accept), 'geometry_review': geometry_review,
                  'geometry_grid': grid, 'fit_frames': fit.tolist(), 'held_out_frames': held.tolist(),
                  'summary': summary, 'candidate_npz': str(out / (stem + '.npz')),
                  'scope': 'Video silhouette/motion and cached-depth diagnostics. Official CD-O remains unknown until submission.'}
        (out / 'review.json').write_text(json.dumps(review, indent=2) + '\n')
        (report / (stem + '_temporal_review.json')).write_text(json.dumps(review, indent=2) + '\n')
        all_results.append(review)
        print('TEMPORAL_CANDIDATE_COMPLETE', episode, json.dumps(summary), flush=True)
(report / 'temporal_candidates.json').write_text(json.dumps(all_results, indent=2) + '\n')
