"""Isolated metric-ring fit for Episode 1 using video masks and input radius."""
import argparse
import json
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

root = Path('/repo'); data = root / 'data/v2d_challenge/track_1'; stem = 'episode_000001'
p = argparse.ArgumentParser(); p.add_argument('--hypotheses-only', action='store_true'); args = p.parse_args()
sys.path.insert(0, '/workspace/v2d_foundation_pose/lib/FoundationPose')
sys.path.insert(0, str(root / 'modules/v2d_cari4d/lib/cari4d'))
from prep.circular_hoop_fit import fit_circle_rays, frame_from_normal
from prep.temporal_pose_selection import select_temporal_rotations, circular_hoop_rotations
from prep.trajectory_diagnostics import rigid_motion_diagnostics
from prep.mhr_export_utils import read_mask, read_rgb, camera_calibration, load_edex
from prep.run_foundationpose_mhr_export import _render_candidate_masks, _scaled_image_grid
from prep.foundationpose_pose_selection import measure_visible_silhouettes

parent = data / 'cdo_v4_experiments/hoop_candidates' / stem
out = data / 'cdo_v4_experiments/hoop_geometry' / stem
if not args.hypotheses_only:
    assert not (out / (stem + '.npz')).exists(), 'Preserve existing experiment'
out.mkdir(parents=True, exist_ok=True)
bundle = dict(np.load(parent / (stem + '.npz'), allow_pickle=False))
mesh = trimesh.load(parent / (stem + '_object.glb'), force='mesh', process=False)
vertices = mesh.vertices * float(bundle['object_scale']); center_local = mesh.bounds.mean(axis=0) * float(bundle['object_scale'])
radius = (np.mean(mesh.extents[1:]) - mesh.extents[0]) * float(bundle['object_scale']) / 2.
renderer = SimpleNamespace(mesh_tensors={'pos': torch.as_tensor(vertices, device='cuda', dtype=torch.float32), 'faces': torch.as_tensor(mesh.faces, device='cuda', dtype=torch.int32)}, glctx=dr.RasterizeCudaContext())
export = data / 'improvement_rollout_v1' / stem / 'run/export' / stem
k, w2c = camera_calibration(load_edex(export), 0)
assert np.allclose(w2c, np.eye(4)), 'This circle fit requires the camera prediction frame'
count = len(bundle['pose'])
poses = np.tile(np.eye(4), (count, 3, 1, 1)); poses[:, :, :3, :3] = bundle['object_rotation'][:, None]; poses[:, :, :3, 3] = bundle['object_translation'][:, None]
costs = np.ones((count, 3)); evidence = []
with torch.inference_mode():
    for frame in range(count):
        obj = read_mask(export, 'object', 0, f'{frame:06d}'); human = read_mask(export, 'human', 0, f'{frame:06d}')
        small_k, shape = _scaled_image_grid(k, obj.shape, 768)
        resize = lambda x: cv2.resize(x.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        small_obj, small_human = resize(obj), resize(human)
        yy, xx = np.nonzero(obj & ~human)
        frame_evidence = {'frame': frame, 'visible_pixels': len(xx)}
        if len(xx) >= 100:
            ids = np.linspace(0, len(xx)-1, min(512, len(xx))).round().astype(int)
            rays = np.stack([(xx[ids]-k[0, 2])/k[0, 0], (yy[ids]-k[1, 2])/k[1, 1], np.ones(len(ids))], axis=1)
            ellipse = cv2.fitEllipse(rays[:, :2].astype(np.float32).reshape(-1, 1, 2))
            center, diameters, angle = ellipse
            major, minor = max(diameters)/2., min(diameters)/2.
            if major > 1e-4 and minor/major > .08:
                depth = radius / major
                initial_center = np.array([center[0]*depth, center[1]*depth, depth])
                major_angle = np.deg2rad(angle + (90. if diameters[1] > diameters[0] else 0.))
                minor_dir = np.array([-np.sin(major_angle), np.cos(major_angle)])
                ratio = np.clip(minor/major, .01, .999)
                errors = []
                for j, sign in enumerate((1., -1.), 1):
                    n0 = np.r_[sign * minor_dir * np.sqrt(1-ratio**2), ratio]
                    try:
                        fitted, normal, error = fit_circle_rays(rays, radius, initial_center, n0, focal_pixels=np.sqrt(k[0, 0]*k[1, 1]))
                    except ValueError:
                        errors.append(None); continue
                    errors.append(error)
                    if error <= 5.:
                        rotation = frame_from_normal(normal)
                        poses[frame, j, :3, :3] = rotation
                        poses[frame, j, :3, 3] = fitted - rotation @ center_local
                frame_evidence['fit_radial_residual_pixels'] = errors
        masks = _render_candidate_masks(renderer, poses[frame], small_k, shape)
        metric = measure_visible_silhouettes(masks, small_human, small_obj)
        costs[frame] = 1. - metric.iou if metric.observed_visible_pixels >= 8 else 0.
        evidence.append(frame_evidence)
        if frame % 100 == 0:
            print('HOOP_GEOMETRY_PROGRESS', frame, count, flush=True)
    hypothesis_file = out / 'fit_hypotheses.npz'
    assert not hypothesis_file.exists(), 'Preserve previously fitted hypotheses'
    np.savez_compressed(hypothesis_file, poses=poses, image_costs=costs)
    if args.hypotheses_only:
        print('HOOP_HYPOTHESES_SAVED', hypothesis_file, flush=True)
        raise SystemExit(0)
    rotations, path, objective = select_temporal_rotations(poses[:, :, :3, :3], costs, motion_weight=.06)
    selected = poses[np.arange(count), path].copy()
    smoothed = selected.copy(); smoothed[:, :3, :3] = circular_hoop_rotations(rotations)
    smoothed[:, :3, 3] = savgol_filter(selected[:, :3, 3], 9, 2, axis=0)
    scored = np.array(json.loads((root / 'incident_reports/track1_scored_frames.json').read_text())['1'])
    frames = scored[np.linspace(0, len(scored)-1, 33).round().astype(int)]
    summary = {}; panels = []
    original = poses[:, 0]
    for label, sequence in [('parent', original), ('ring_fit', selected), ('ring_fit_smooth', smoothed)]:
        ious = []
        for frame in frames:
            obj = read_mask(export, 'object', 0, f'{frame:06d}'); human = read_mask(export, 'human', 0, f'{frame:06d}')
            small_k, shape = _scaled_image_grid(k, obj.shape, 768)
            resize = lambda x: cv2.resize(x.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            obj, human = resize(obj), resize(human)
            mask = _render_candidate_masks(renderer, sequence[frame:frame+1], small_k, shape)
            ious.append(float(measure_visible_silhouettes(mask, human, obj).iou[0]))
            if frame in frames[np.linspace(0, len(frames)-1, 6).round().astype(int)]:
                rgb = cv2.resize(read_rgb(export, 0, f'{frame:06d}'), (shape[1], shape[0]))
                rgb[obj] = (rgb[obj]*.6 + np.array([0, 255, 0])*.4).astype(np.uint8)
                contours, _ = cv2.findContours(mask[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb, contours, -1, (255, 60, 60), 2)
                rgb = cv2.resize(rgb, (512, 384))
                cv2.putText(rgb, f'{label} f{frame} IoU {ious[-1]:.3f}', (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                panels.append(rgb)
        summary[label] = {'visible_iou_mean': float(np.mean(ious)), 'translation_acceleration_cm_proxy': float(np.linalg.norm(np.diff(sequence[:, :3, 3], n=2, axis=0), axis=1).mean()*100), **rigid_motion_diagnostics(sequence, vertices)}
    accepted = summary['ring_fit_smooth']['visible_iou_mean'] >= summary['ring_fit']['visible_iou_mean'] - .03
    final, label = (smoothed, 'ring_fit_smooth') if accepted else (selected, 'ring_fit')
    bundle.update(object_rotation=final[:, :3, :3].astype(np.float32), object_translation=final[:, :3, 3].astype(np.float32))
    np.savez_compressed(out / (stem + '.npz'), **bundle)
    shutil.copy2(parent / (stem + '_object.glb'), out / (stem + '_object.glb'))
    review = json.loads((parent / 'review.json').read_text())
    review.update(circle_fit={'radius_m_from_input_mesh': radius, 'objective': objective, 'summary': summary, 'smoothing_accepted': bool(accepted), 'evidence': evidence})
    review['summary']['final_candidate'] = summary[label]
    review['candidate_npz'] = str(out / (stem + '.npz'))
    (out / 'review.json').write_text(json.dumps(review, indent=2) + '\n')
    cv2.imwrite(str(out / 'comparison.jpg'), cv2.cvtColor(np.concatenate([np.concatenate(panels[j:j+6], axis=1) for j in range(0, len(panels), 6)], axis=0), cv2.COLOR_RGB2BGR))
    print('HOOP_GEOMETRY_COMPLETE', json.dumps(summary), flush=True)
