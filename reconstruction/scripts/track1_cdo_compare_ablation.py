"""Compare completed ablations to the scored submission on identical masks."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, default=Path('/repo'))
p.add_argument('--episode', type=int, default=12)
a = p.parse_args()
sys.path.insert(0, '/workspace/v2d_foundation_pose/lib/FoundationPose')
sys.path.insert(0, str(a.root / 'modules/v2d_cari4d/lib/cari4d'))
from prep.run_foundationpose_mhr_export import _render_candidate_masks, _scaled_image_grid
from prep.foundationpose_pose_selection import measure_visible_silhouettes
from prep.mhr_export_utils import read_mask, read_rgb, camera_calibration, load_edex
from prep.trajectory_diagnostics import rigid_motion_diagnostics

data = a.root / 'data/v2d_challenge/track_1'
stem = f'episode_{a.episode:06d}'
work = data / 'improvement_rollout_v1' / stem
export = work / 'run/export' / stem
k, w2c = camera_calibration(load_edex(export), 0)
scored = np.array(json.loads((a.root / 'incident_reports/track1_scored_frames.json').read_text())[str(a.episode)])
frames = scored[np.linspace(0, len(scored) - 1, min(33, len(scored))).round().astype(int)]
versions = {}
for label, pairs in [('submitted_v3', data / 'improvement_rollout_v1/candidate_episodes'), ('previous_rollout_candidate', work / 'submission_episodes')]:
    bundle = dict(np.load(pairs / (stem + '.npz'), allow_pickle=False))
    mesh = trimesh.load(pairs / (stem + '_object.glb'), force='mesh', process=False)
    poses = np.repeat(np.eye(4)[None], len(bundle['pose']), axis=0)
    poses[:, :3, :3] = bundle['object_rotation']; poses[:, :3, 3] = bundle['object_translation']
    versions[label] = (mesh.vertices * float(bundle['object_scale']), mesh.faces, poses, {})
mesh = trimesh.load(work / 'run/export' / stem / 'object_mesh/output_aligned.glb', force='mesh', process=False)
for folder in sorted((data / 'cdo_v4_experiments/refinement' / stem).iterdir()):
    path = folder / 'refined.pth'
    if not path.exists():
        continue
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    poses = np.asarray(bundle['pr'].get('pose_abs_postopt', bundle['pr']['pose_abs'])).copy()
    postopt = bundle.get('postopt', {})
    versions[folder.name] = (mesh.vertices, mesh.faces, poses, {'source': str(path), 'human_drift_diagnostics': postopt.get('final_diagnostics', {}), 'optimized_parameters': postopt.get('optimized_parameters', [])})
    del bundle

summary = {}; samples = {}; panels = []
panel_frames = set(frames[np.linspace(0, len(frames) - 1, 3).round().astype(int)])
with torch.no_grad():
    for label, (vertices, faces, poses, metadata) in versions.items():
        assert np.isfinite(poses).all() and poses.shape[1:] == (4, 4)
        renderer = SimpleNamespace(mesh_tensors={'pos': torch.as_tensor(vertices, device='cuda', dtype=torch.float32), 'faces': torch.as_tensor(faces, device='cuda', dtype=torch.int32)}, glctx=dr.RasterizeCudaContext())
        camera_poses = w2c[None] @ poses
        rows = []
        for frame in frames:
            name = f'{frame:06d}'
            human = read_mask(export, 'human', 0, name); obj = read_mask(export, 'object', 0, name)
            small_k, shape = _scaled_image_grid(k, obj.shape, 768)
            human = cv2.resize(human.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            obj = cv2.resize(obj.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            mask = _render_candidate_masks(renderer, camera_poses[frame:frame + 1], small_k, shape)
            metrics = measure_visible_silhouettes(mask, human, obj)
            rows.append({'frame': int(frame), 'visible_iou': float(metrics.iou[0])})
            if frame in panel_frames:
                rgb = cv2.resize(read_rgb(export, 0, name), (shape[1], shape[0]))
                rgb[obj] = (rgb[obj] * .6 + np.array([0, 255, 0]) * .4).astype(np.uint8)
                contours, _ = cv2.findContours(mask[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb, contours, -1, (255, 60, 60), 1)
                rgb = cv2.resize(rgb, (512, 384))
                cv2.putText(rgb, f'{label} f{frame} IoU {metrics.iou[0]:.3f}', (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
                panels.append(rgb)
        summary[label] = {'visible_iou_mean': float(np.mean([r['visible_iou'] for r in rows])), 'translation_acceleration_cm_proxy': float(np.linalg.norm(np.diff(poses[:, :3, 3], n=2, axis=0), axis=1).mean() * 100), 'negative_camera_center_depth_frames': int((camera_poses[:, 2, 3] <= 0).sum()), **rigid_motion_diagnostics(poses, vertices), **metadata}
        samples[label] = rows
        del renderer
out = a.root / 'incident_reports/track1_improvement/cdo_v4'
result = {'episode': a.episode, 'sample_count': len(frames), 'ground_truth_used': False, 'scope': 'Local silhouette and motion diagnostics, not official CD-O or penetration scores; completed variants only', 'summary': summary, 'samples': samples}
(out / (stem + '_ablation_comparison.json')).write_text(json.dumps(result, indent=2) + '\n')
cv2.imwrite(str(out / (stem + '_ablation_comparison.jpg')), cv2.cvtColor(np.concatenate([np.concatenate(panels[i:i + 3], axis=1) for i in range(0, len(panels), 3)], axis=0), cv2.COLOR_RGB2BGR))
print(json.dumps(result['summary'], indent=2), flush=True)
