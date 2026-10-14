"""Read-only rendered-depth diagnostics against cached monocular depth, not GT."""
import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, default=Path('/repo'))
p.add_argument('--episodes', type=int, nargs='+', default=[1, 2, 12, 14, 19, 20])
p.add_argument('--translation-probe', action='store_true', help='Fit one ray-preserving depth offset on two-thirds of samples; evaluate on the held-out third')
a = p.parse_args()
sys.path.insert(0, '/workspace/v2d_foundation_pose/lib/FoundationPose')
sys.path.insert(0, str(a.root / 'modules/v2d_cari4d/lib/cari4d'))
from track1_rollout_common import paths
from prep.mhr_export_utils import read_depth_m, read_mask, camera_calibration, load_edex
from prep.run_foundationpose_mhr_export import _scaled_image_grid
import Utils

data = a.root / 'data/v2d_challenge/track_1'
out = a.root / 'incident_reports/track1_improvement/cdo_v4'
out.mkdir(parents=True, exist_ok=True)
scored = json.loads((a.root / 'incident_reports/track1_scored_frames.json').read_text())
ctx = dr.RasterizeCudaContext()


def render(vertices, faces, pose, k, shape):
    vertices = torch.as_tensor(vertices, device='cuda', dtype=torch.float32)
    faces = torch.as_tensor(faces, device='cuda', dtype=torch.int32)
    pose = torch.as_tensor(pose, device='cuda', dtype=torch.float32)
    camera = vertices @ pose[:3, :3].T + pose[:3, 3]
    projection = torch.as_tensor(Utils.projection_matrix_from_intrinsics(k, height=shape[0], width=shape[1], znear=.001, zfar=100.), device='cuda', dtype=torch.float32)
    cv_to_gl = torch.as_tensor(Utils.glcam_in_cvcam, device='cuda', dtype=torch.float32)
    clip = Utils.to_homo_torch(camera) @ (projection @ cv_to_gl).T
    raster, _ = dr.rasterize(ctx, clip[None], faces, resolution=shape)
    depth, _ = dr.interpolate(camera[None, :, 2:3].contiguous(), raster, faces)
    return torch.flip(depth[0, :, :, 0], [0]).cpu().numpy(), torch.flip(raster[0, :, :, 3] > 0, [0]).cpu().numpy()


with torch.no_grad():
    # Verify depth units and image orientation before using scene evidence.
    box = trimesh.creation.box(extents=[1., 1., 1.])
    pose = np.eye(4); pose[2, 3] = 3.
    k = np.array([[80., 0, 48], [0, 80., 32], [0, 0, 1.]])
    z, mask = render(box.vertices, box.faces, pose, k, (64, 96))
    assert mask[32, 48] and abs(z[32, 48] - 2.5) < 1e-4
    all_results = []
    for i in a.episodes:
        q = paths(a.root, i); work = q['work']; stem = q['stem']
        export = work / 'run/export' / stem
        k, w2c = camera_calibration(load_edex(export), 0)
        frames = np.array(scored[str(i)])
        frames = frames[np.linspace(0, len(frames) - 1, min(33, len(frames))).round().astype(int)]
        versions = {}
        for label, pairs in [('submitted_v3', data / 'improvement_rollout_v1/candidate_episodes'), ('rollout_candidate', work / 'submission_episodes')]:
            bundle = dict(np.load(pairs / (stem + '.npz'), allow_pickle=False))
            mesh = trimesh.load(pairs / (stem + '_object.glb'), force='mesh', process=False)
            poses = np.repeat(np.eye(4)[None], len(bundle['pose']), axis=0)
            poses[:, :3, :3] = bundle['object_rotation']; poses[:, :3, 3] = bundle['object_translation']
            versions[label] = (mesh.vertices * float(bundle['object_scale']), mesh.faces, poses)
        fp = pickle.loads((work / 'run/foundationpose' / (stem + '.pkl')).read_bytes())
        mesh = trimesh.load(work / 'run/export' / stem / 'object_mesh/output_aligned.glb', force='mesh', process=False)
        versions['gated_tracker'] = (mesh.vertices, mesh.faces, fp['obj_pose_world'])
        result = {'episode': i, 'reference': 'cached human-aligned monocular depth; not held-out ground truth', 'versions': {}}
        for label, (vertices, faces, poses) in list(versions.items()):
            rows = []
            for frame in frames:
                name = f'{frame:06d}'
                obj = read_mask(export, 'object', 0, name)
                human = read_mask(export, 'human', 0, name)
                depth = read_depth_m(export, 0, name, depth_root=q['depth'])
                small_k, shape = _scaled_image_grid(k, obj.shape, 768)
                resize = lambda x: cv2.resize(x, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
                visible = resize((obj & ~human).astype(np.uint8)) > 0
                human = resize(human.astype(np.uint8)) > 0
                depth = resize(depth)
                z, mask = render(vertices, faces, w2c @ poses[frame], small_k, shape)
                rendered = mask & ~human
                union = (rendered | visible).sum()
                support = visible & rendered & np.isfinite(depth) & (depth > .05) & (z > .05)
                row = {'frame': int(frame), 'visible_iou': float((rendered & visible).sum() / max(1, union)), 'depth_support_pixels': int(support.sum())}
                if support.sum() >= 8:
                    residual = (z[support] - depth[support]) * 100.
                    row.update(depth_bias_cm=float(np.median(residual)), depth_median_absolute_error_cm=float(np.median(abs(residual))))
                rows.append(row)
            supported = [r for r in rows if 'depth_bias_cm' in r]
            result['versions'][label] = {'sample_count': len(rows), 'supported_samples': len(supported), 'visible_iou_mean': float(np.mean([r['visible_iou'] for r in rows])), 'median_depth_bias_cm': float(np.median([r['depth_bias_cm'] for r in supported])) if supported else None, 'median_depth_absolute_error_cm': float(np.median([r['depth_median_absolute_error_cm'] for r in supported])) if supported else None, 'samples': rows}
            if a.translation_probe and label in ('submitted_v3', 'rollout_candidate'):
                fit = [r for j, r in enumerate(rows) if j % 3 != 2 and 'depth_bias_cm' in r]
                test = [int(f) for j, f in enumerate(frames) if j % 3 == 2]
                if len(fit) < 8:
                    result['versions'][label]['probe_skipped'] = 'Fewer than eight fitting samples with valid rendered-depth overlap'
                    continue
                offset = -float(np.median([r['depth_bias_cm'] for r in fit])) / 100.
                changed = np.asarray(poses).copy()
                camera_poses = w2c[None] @ changed
                old_z = camera_poses[:, 2, 3].copy()
                if not np.all(old_z + offset > .1):
                    result['versions'][label]['probe_skipped'] = 'Offset produces nonpositive or near-camera depth'
                    continue
                camera_poses[:, :3, 3] *= ((old_z + offset) / old_z)[:, None]
                changed = np.linalg.inv(w2c)[None] @ camera_poses
                # A separate experiment directory; no submitted prediction is edited.
                probe_dir = data / 'cdo_v4_experiments/depth_offset' / stem / label
                probe_dir.mkdir(parents=True, exist_ok=True)
                npz = probe_dir / (stem + '.npz')
                if npz.exists():
                    raise FileExistsError(f'Preserve existing experiment: {npz}')
                parent = data / 'improvement_rollout_v1/candidate_episodes' if label == 'submitted_v3' else work / 'submission_episodes'
                bundle = dict(np.load(parent / (stem + '.npz'), allow_pickle=False))
                bundle['object_translation'] = changed[:, :3, 3].astype(bundle['object_translation'].dtype)
                np.savez_compressed(npz, **bundle)
                probe_rows = []
                for frame in test:
                    name = f'{frame:06d}'
                    obj = read_mask(export, 'object', 0, name); human = read_mask(export, 'human', 0, name)
                    depth = read_depth_m(export, 0, name, depth_root=q['depth'])
                    small_k, shape = _scaled_image_grid(k, obj.shape, 768)
                    resize = lambda x: cv2.resize(x, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
                    visible = resize((obj & ~human).astype(np.uint8)) > 0
                    human = resize(human.astype(np.uint8)) > 0; depth = resize(depth)
                    z, mask = render(vertices, faces, w2c @ changed[frame], small_k, shape)
                    rendered = mask & ~human; support = visible & rendered & np.isfinite(depth) & (depth > .05) & (z > .05)
                    row = {'frame': frame, 'visible_iou': float((rendered & visible).sum() / max(1, (rendered | visible).sum())), 'depth_support_pixels': int(support.sum())}
                    if support.sum() >= 8:
                        row['depth_median_absolute_error_cm'] = float(np.median(abs(z[support] - depth[support])) * 100.)
                    probe_rows.append(row)
                before = [r for r in rows if r['frame'] in test]
                valid_before = [r['depth_median_absolute_error_cm'] for r in before if 'depth_median_absolute_error_cm' in r]
                valid_after = [r['depth_median_absolute_error_cm'] for r in probe_rows if 'depth_median_absolute_error_cm' in r]
                probe = {'fit_frames': [r['frame'] for r in fit], 'evaluation_frames': test, 'fitted_depth_offset_m': offset, 'held_out_iou_before': float(np.mean([r['visible_iou'] for r in before])), 'held_out_iou_after': float(np.mean([r['visible_iou'] for r in probe_rows])), 'held_out_depth_error_cm_before': float(np.median(valid_before)) if valid_before else None, 'held_out_depth_error_cm_after': float(np.median(valid_after)) if valid_after else None, 'held_out_supported_samples_before': len(valid_before), 'held_out_supported_samples_after': len(valid_after), 'samples': probe_rows, 'scope': 'translation-only probe; cached monocular depth is not ground truth; no promotion or upload', 'npz': str(npz)}
                assert not set(probe['fit_frames']) & set(test)
                (probe_dir / 'probe.json').write_text(json.dumps(probe, indent=2, allow_nan=False) + '\n')
                result['versions'][label]['translation_probe'] = probe
        (out / (stem + '_depth_audit.json')).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        all_results.append(result)
        print('DEPTH_AUDIT', i, json.dumps({k: {n: v for n, v in x.items() if n != 'samples'} for k, x in result['versions'].items()}), flush=True)
    (out / 'depth_audit.json').write_text(json.dumps(all_results, indent=2, allow_nan=False) + '\n')
