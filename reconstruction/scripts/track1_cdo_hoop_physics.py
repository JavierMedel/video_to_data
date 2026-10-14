"""Resolve circle-plane ambiguity with signed body intersection evidence."""
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
p = argparse.ArgumentParser(); p.add_argument('--second-pass', action='store_true'); args = p.parse_args()
sys.path.insert(0, '/workspace/v2d_sam3d_body/lib')
sys.path.insert(0, '/workspace/v2d_foundation_pose/lib/FoundationPose')
sys.path.insert(0, str(root / 'modules/v2d_cari4d/lib/cari4d'))
sys.path.insert(0, str(root / 'v2d_submission_kit/v2d_submission_kit/tools/track1'))
from prep.run_foundationpose_mhr_export import _render_candidate_masks, _scaled_image_grid
from prep.mhr_export_utils import read_mask, read_rgb, camera_calibration, load_edex
from prep.foundationpose_pose_selection import measure_visible_silhouettes
from prep.temporal_pose_selection import select_temporal_rotations, circular_hoop_rotations
from prep.trajectory_diagnostics import rigid_motion_diagnostics
from mesh_to_mhr_params import MHR
from lib_mhr import MHRLayer
from lib_mhr.collision_proxy import load_mhr_collision_proxy, apply_mhr_collision_proxy
from lib_mhr.contact import sample_mesh_surface_points
from lib_mhr.sdf import object_inside_human_penetration_loss

source = data / 'cdo_v4_experiments/hoop_geometry' / stem
out = data / ('cdo_v4_experiments/physics_hoop_v2' if args.second_pass else 'cdo_v4_experiments/physics_hoop') / stem
assert not (out / (stem + '.npz')).exists()
out.mkdir(parents=True, exist_ok=True)
bundle = dict(np.load(source / (stem + '.npz'), allow_pickle=False))
hypotheses = dict(np.load(source / 'fit_hypotheses.npz', allow_pickle=False))
poses = hypotheses['poses']; costs = hypotheses['image_costs']
mesh = trimesh.load(source / (stem + '_object.glb'), force='mesh', process=False)
vertices = mesh.vertices * float(bundle['object_scale'])
points = torch.as_tensor(sample_mesh_surface_points(vertices, mesh.faces, 256, seed=42), device='cuda', dtype=torch.float32)
layer = MHRLayer.from_mhr_assets(mhr_assets_root=str(root / 'data/weights/sam3d_body'), device='cuda')
faces = layer.mesh_faces(device='cuda').cpu().numpy()
proxy = load_mhr_collision_proxy(faces)
del layer
torch.cuda.empty_cache()
model = MHR(str(root / 'data/weights/track1_submission_mhr/mhr_model.pt'), 'cuda', 4, 'float32')
penalties = np.zeros(costs.shape)
with torch.inference_mode():
    human, _ = model.run(torch.as_tensor(bundle['pose'], device='cuda', dtype=torch.float64), torch.as_tensor(np.r_[bundle['scales'], bundle['shape']][None], device='cuda', dtype=torch.float64))
    human = apply_mhr_collision_proxy((human/1000.).float(), torch.as_tensor(proxy.source_vertex_indices.copy(), device='cuda', dtype=torch.long), torch.as_tensor(proxy.barycentric_weights.copy(), device='cuda'))
    for frame in range(len(poses)):
        for j in range(poses.shape[1]):
            r = torch.as_tensor(poses[frame, j, :3, :3], device='cuda', dtype=torch.float32)
            t = torch.as_tensor(poses[frame, j, :3, 3], device='cuda', dtype=torch.float32)
            wp = points @ r.T + t
            penalties[frame, j] = float(object_inside_human_penetration_loss(human[frame:frame+1], proxy.faces.copy(), wp[None], conservative_aabb_rejection=True).cpu()) * 100.
        if frame % 100 == 0:
            print('HOOP_PHYSICS_PROGRESS', frame, len(poses), flush=True)
    # Transport each branch before temporal selection; a hoop's in-plane spin
    # and normal sign do not identify a distinct physical plane.
    for j in range(poses.shape[1]):
        poses[:, j, :3, :3] = circular_hoop_rotations(poses[:, j, :3, :3], smoothing_radius=0)
    penalty_weight = .2 if args.second_pass else .5
    kwargs = {'translations': poses[:, :, :3, 3]} if args.second_pass else {}
    rotations, path, objective = select_temporal_rotations(poses[:, :, :3, :3], costs + penalty_weight*np.minimum(penalties/.01, 10.), motion_weight=.06, **kwargs)
    selected = poses[np.arange(len(poses)), path].copy()
    final = selected.copy(); final[:, :3, :3] = circular_hoop_rotations(rotations)
    final[:, :3, 3] = savgol_filter(selected[:, :3, 3], 9, 2, axis=0)
    export = data / 'improvement_rollout_v1' / stem / 'run/export' / stem
    k, w2c = camera_calibration(load_edex(export), 0)
    renderer = SimpleNamespace(mesh_tensors={'pos': torch.as_tensor(vertices, device='cuda', dtype=torch.float32), 'faces': torch.as_tensor(mesh.faces, device='cuda', dtype=torch.int32)}, glctx=dr.RasterizeCudaContext())
    frames = np.array(json.loads((root / 'incident_reports/track1_scored_frames.json').read_text())['1'])
    frames = frames[np.linspace(0, len(frames)-1, 33).round().astype(int)]
    summary = {}; panels = []
    for label, sequence in [('selected', selected), ('smoothed', final)]:
        ious = []
        for frame in frames:
            obj = read_mask(export, 'object', 0, f'{frame:06d}'); hm = read_mask(export, 'human', 0, f'{frame:06d}')
            small_k, shape = _scaled_image_grid(k, obj.shape, 768)
            resize = lambda x: cv2.resize(x.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            obj, hm = resize(obj), resize(hm)
            mask = _render_candidate_masks(renderer, w2c[None] @ sequence[frame:frame+1], small_k, shape)
            ious.append(float(measure_visible_silhouettes(mask, hm, obj).iou[0]))
            if frame in frames[np.linspace(0, len(frames)-1, 6).round().astype(int)]:
                rgb = cv2.resize(read_rgb(export, 0, f'{frame:06d}'), (shape[1], shape[0]))
                rgb[obj] = (rgb[obj]*.6 + np.array([0, 255, 0])*.4).astype(np.uint8)
                contours, _ = cv2.findContours(mask[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb, contours, -1, (255, 60, 60), 2)
                rgb = cv2.resize(rgb, (512, 384))
                cv2.putText(rgb, f'{label} f{frame} IoU {ious[-1]:.3f}', (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                panels.append(rgb)
        summary[label] = {'visible_iou_mean': float(np.mean(ious)), 'translation_acceleration_cm_proxy': float(np.linalg.norm(np.diff(sequence[:, :3, 3], n=2, axis=0), axis=1).mean()*100), **rigid_motion_diagnostics(sequence, vertices)}
    tolerance = .06 if args.second_pass else .03
    accepted = summary['smoothed']['visible_iou_mean'] >= summary['selected']['visible_iou_mean'] - tolerance
    sequence, label = (final, 'smoothed') if accepted else (selected, 'selected')
    bundle.update(object_rotation=sequence[:, :3, :3].astype(np.float32), object_translation=sequence[:, :3, 3].astype(np.float32))
    np.savez_compressed(out / (stem + '.npz'), **bundle)
    shutil.copy2(source / (stem + '_object.glb'), out / (stem + '_object.glb'))
    review = json.loads((source / 'review.json').read_text())
    review['summary']['final_candidate'] = summary[label]
    review.update(physics_selection={'signed_samples_per_frame': 256, 'objective': objective, 'summary': summary, 'smoothing_accepted': bool(accepted), 'penetration_weight': penalty_weight, 'translation_penalty': bool(args.second_pass), 'smoothing_iou_tolerance': tolerance}, candidate_npz=str(out / (stem + '.npz')))
    (out / 'review.json').write_text(json.dumps(review, indent=2) + '\n')
    np.savez_compressed(out / 'selection.npz', penalties_cm=penalties, path=path)
    cv2.imwrite(str(out / 'comparison.jpg'), cv2.cvtColor(np.concatenate([np.concatenate(panels[:6], axis=1), np.concatenate(panels[6:], axis=1)], axis=0), cv2.COLOR_RGB2BGR))
    print('HOOP_PHYSICS_COMPLETE', json.dumps(summary), flush=True)
