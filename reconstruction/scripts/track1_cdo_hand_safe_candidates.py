"""Fit bounded ray corrections with hand penetration and image checks."""
import argparse
import hashlib
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
from scipy.spatial import cKDTree

root = Path('/repo'); data = root/'data/v2d_challenge/track_1'
p = argparse.ArgumentParser(); p.add_argument('--contact-aware', action='store_true'); args = p.parse_args()
sys.path.insert(0, '/workspace/v2d_sam3d_body/lib')
sys.path.insert(0, '/workspace/v2d_foundation_pose/lib/FoundationPose')
sys.path.insert(0, str(root/'modules/v2d_cari4d/lib/cari4d'))
sys.path.insert(0, str(root/'v2d_submission_kit/v2d_submission_kit/tools/track1'))
from prep.run_foundationpose_mhr_export import _render_candidate_masks, _scaled_image_grid
from prep.mhr_export_utils import read_mask, camera_calibration, load_edex
from prep.foundationpose_pose_selection import measure_visible_silhouettes
from prep.trajectory_diagnostics import rigid_motion_diagnostics
from lib_mhr.hand_surface_contact import load_mhr_hand_surface_spec
from lib_mhr.mesh_penetration_review import MeshPenetrationReview
from mesh_to_mhr_params import MHR

report = root/'incident_reports/track1_improvement/cdo_v5'
scored = json.loads((root/'incident_reports/track1_scored_frames.json').read_text())
spec = load_mhr_hand_surface_spec(root/'modules/v2d_cari4d/lib/cari4d/lib_mhr/assets/mhr_hand_surface_spec.npz')
hand_ids = spec.vertex_indices.reshape(-1)
fps_ids = (spec.sample_local_indices + np.arange(2)[:, None]*spec.vertex_indices.shape[1]).reshape(-1)
model = MHR(str(root/'data/weights/track1_submission_mhr/mhr_model.pt'), 'cuda', 4, 'float32')
ctx = dr.RasterizeCudaContext()


def poses_from(bundle):
    poses = np.tile(np.eye(4), (len(bundle['pose']), 1, 1))
    poses[:, :3, :3] = bundle['object_rotation']; poses[:, :3, 3] = bundle['object_translation']
    return poses


with torch.inference_mode():
    for episode in ((23, 18, 21, 16, 13) if args.contact_aware else (23, 18, 16, 13)):
        stem = f'episode_{episode:06d}'
        parent_dir = data/'cdo_v5_experiments/pose_candidates'/stem
        out = data/'cdo_v5_experiments'/('hand_contact_candidates' if args.contact_aware else 'hand_safe_candidates')/stem
        if (out/'probe.json').exists():
            print('HAND_SAFE_REUSED', episode, flush=True); continue
        out.mkdir(parents=True, exist_ok=True)
        control_dir = data/'cdo_v4_experiments/candidate_episodes'
        control = dict(np.load(control_dir/(stem+'.npz'), allow_pickle=False))
        candidate = dict(np.load(parent_dir/(stem+'.npz'), allow_pickle=False))
        mesh = trimesh.load(parent_dir/(stem+'_object.glb'), force='mesh', process=False)
        check = MeshPenetrationReview(mesh.vertices, mesh.faces)
        frames = np.array(scored[str(episode)]); frames = frames[np.linspace(0, len(frames)-1, 33).round().astype(int)]
        fit = np.flatnonzero(np.arange(len(frames))%3 != 2); held = np.flatnonzero(np.arange(len(frames))%3 == 2)
        cached_path = data/'cdo_v5_experiments/hand_safe_candidates'/stem/'probe.json'
        cached = json.loads(cached_path.read_text()) if args.contact_aware and cached_path.exists() else None
        if cached:
            assert cached['fit_frames'] == frames[fit].tolist() and cached['held_out_frames'] == frames[held].tolist()
            fingerprints = json.loads((parent_dir/'hand_penetration_review.sampled.json').read_text())
            for directory, key in ((parent_dir, 'candidate'), (control_dir, 'control')):
                for suffix, field in (('.npz', 'npz'), ('_object.glb', 'mesh')):
                    assert hashlib.sha256((directory/(stem+suffix)).read_bytes()).hexdigest() == fingerprints[f'{key}_{field}_sha256']
            print('HAND_SAFE_REUSE_PENETRATION_METRICS', episode, flush=True)
        human, _ = model.run(torch.as_tensor(control['pose'][frames], device='cuda', dtype=torch.float64), torch.as_tensor(np.r_[control['scales'], control['shape']][None], device='cuda', dtype=torch.float64))
        hands = human[:, hand_ids]/1000.
        work = data/('improvement_pilot_v1' if episode == 18 else 'improvement_rollout_v1')/stem
        export = work/'run/export'/stem
        k, w2c = camera_calibration(load_edex(export), 0); assert np.allclose(w2c, np.eye(4))
        obs = {}
        for frame in frames:
            obj = read_mask(export, 'object', 0, f'{frame:06d}'); hm = read_mask(export, 'human', 0, f'{frame:06d}')
            small_k, shape = _scaled_image_grid(k, obj.shape, 768)
            resize = lambda x: cv2.resize(x.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)>0
            obs[int(frame)] = (resize(obj), resize(hm), small_k, shape)

        def evaluate(poses, scale, geometry=mesh, checker=check, skip_depth=False):
            rotation = torch.as_tensor(poses[frames, :3, :3], device='cuda', dtype=torch.float64)
            translation = torch.as_tensor(poses[frames, :3, 3], device='cuda', dtype=torch.float64)
            local = torch.einsum('tpi,tij->tpj', hands-translation[:, None], rotation)/scale
            depth = None if skip_depth else checker.depths(local).cpu().numpy()*scale*100.
            rend = SimpleNamespace(mesh_tensors={'pos': torch.as_tensor(geometry.vertices*scale, device='cuda', dtype=torch.float32), 'faces': torch.as_tensor(geometry.faces, device='cuda', dtype=torch.int32)}, glctx=ctx)
            ious = []
            for frame in frames:
                obj, hm, small_k, shape = obs[int(frame)]
                mask = _render_candidate_masks(rend, poses[frame:frame+1], small_k, shape)
                ious.append(float(measure_visible_silhouettes(mask, hm, obj).iou[0]))
            closest = cKDTree(np.asarray(geometry.vertices)).query(local[:, fps_ids].cpu().numpy().reshape(-1, 3), workers=4)[0]
            proximity = closest.reshape(len(frames), len(fps_ids)).min(axis=1)*scale*100.
            return depth, np.array(ious), proximity

        old_mesh = trimesh.load(control_dir/(stem+'_object.glb'), force='mesh', process=False)
        old_check = MeshPenetrationReview(old_mesh.vertices, old_mesh.faces)
        old_depth, old_iou, old_proximity = evaluate(poses_from(control), float(control['object_scale']), old_mesh, old_check, skip_depth=bool(cached))
        old_full = cached['old_fit_mean_depth_cm'] if cached else float(old_depth[fit].mean())
        old_fps = cached['old_fit_fps512_depth_cm'] if cached else float(old_depth[fit][:, fps_ids].mean())
        old_held_full = cached['old_held_mean_depth_cm'] if cached else float(old_depth[held].mean())
        old_held_fps = cached['old_held_fps512_depth_cm'] if cached else float(old_depth[held][:, fps_ids].mean())
        options = []
        for offset in (-.06, -.03, 0., .03, .06):
            poses = poses_from(candidate)
            z = poses[:, 2, 3].copy(); poses[:, :3, 3] *= ((z+offset)/z)[:, None]
            options.append((f'ray_{offset:+.2f}', poses, float(candidate['object_scale'])))
        # Separate removal of the depth/scale probe, keeping temporal selection.
        neutral = poses_from(control)
        neutral[:, :3, :3] = np.load(parent_dir/'rotation_selection.npz')['selected_rotations']
        neutral[:, :3, 3] = savgol_filter(neutral[:, :3, 3], 7, 2, axis=0)
        options.append(('temporal_without_geometry_probe', neutral, float(control['object_scale'])))
        rows = []; best = None
        cached_options = {row['label']: row for row in cached['options']} if cached else {}
        for label, poses, scale in options:
            cached_row = cached_options.get(label)
            depth, ious, proximity = evaluate(poses, scale, skip_depth=bool(cached_row))
            fit_full = cached_row['fit_mean_depth_cm'] if cached_row else float(depth[fit].mean())
            fit_fps = cached_row['fit_fps512_depth_cm'] if cached_row else float(depth[fit][:, fps_ids].mean())
            held_full = cached_row['held_mean_depth_cm'] if cached_row else float(depth[held].mean())
            held_fps = cached_row['held_fps512_depth_cm'] if cached_row else float(depth[held][:, fps_ids].mean())
            eligible = float(ious[fit].mean()) >= max(.3, float(old_iou[fit].mean())-.015) and fit_full <= old_full+1e-8 and fit_fps <= old_fps+1e-8
            active_fit = fit[old_proximity[fit] <= 1.5]
            fit_increase = proximity[active_fit]-old_proximity[active_fit]
            proximity_fit_pass = not len(fit_increase) or (float(np.median(fit_increase)) <= 1.0 and float(np.quantile(fit_increase, .95)) <= 2.0)
            if args.contact_aware:
                eligible = eligible and proximity_fit_pass
            objective = max(fit_full/(old_full+1e-8), fit_fps/(old_fps+1e-8))
            row = {'label': label, 'fit_mean_depth_cm': fit_full, 'fit_fps512_depth_cm': fit_fps, 'fit_iou': float(ious[fit].mean()),
                   'held_mean_depth_cm': held_full, 'held_fps512_depth_cm': held_fps, 'held_iou': float(ious[held].mean()),
                   'fit_eligible': bool(eligible), 'fit_objective': objective,
                   'fit_contact_frames': len(active_fit), 'fit_median_gap_increase_cm': float(np.median(fit_increase)) if len(fit_increase) else None,
                   'fit_p95_gap_increase_cm': float(np.quantile(fit_increase, .95)) if len(fit_increase) else None}
            rows.append(row)
            print('HAND_SAFE_PROBE', episode, json.dumps(row), flush=True)
            if eligible and (best is None or objective < best[0]):
                best = (objective, label, poses, scale, row, ious, proximity)
        accepted = False; chosen_label = None
        if best:
            _, label, poses, scale, metrics, ious, proximity = best
            accepted = float(ious[held].mean()) >= max(.3, float(old_iou[held].mean())-.015) and metrics['held_mean_depth_cm'] <= old_held_full+1e-8 and metrics['held_fps512_depth_cm'] <= old_held_fps+1e-8
            active_held = held[old_proximity[held] <= 1.5]
            held_increase = proximity[active_held]-old_proximity[active_held]
            if args.contact_aware and len(held_increase):
                accepted = accepted and float(np.median(held_increase)) <= 1.0 and float(np.quantile(held_increase, .95)) <= 2.0
            chosen_label = label
            if accepted:
                result = dict(candidate); result.update(object_rotation=poses[:, :3, :3].astype(np.float32), object_translation=poses[:, :3, 3].astype(np.float32), object_scale=np.array(scale, dtype=np.float32))
                np.savez_compressed(out/(stem+'.npz'), **result)
                shutil.copy2(parent_dir/(stem+'_object.glb'), out/(stem+'_object.glb'))
                review = json.loads((parent_dir/'review.json').read_text())
                review['summary']['final_candidate'] = {'visible_iou_mean': float(ious.mean()), 'translation_acceleration_cm_proxy': float(np.linalg.norm(np.diff(poses[:, :3, 3], n=2, axis=0), axis=1).mean()*100), **rigid_motion_diagnostics(poses, mesh.vertices*scale)}
                review.update(hand_safe_probe={'fit_frames': frames[fit].tolist(), 'held_out_frames': frames[held].tolist(), 'chosen_label': label, 'accepted': True}, candidate_npz=str(out/(stem+'.npz')))
                (out/'review.json').write_text(json.dumps(review, indent=2)+'\n')
        probe = {'episode': episode, 'accepted': bool(accepted), 'chosen_label': chosen_label, 'options': rows,
                 'contact_aware': bool(args.contact_aware),
                 'fit_frames': frames[fit].tolist(), 'held_out_frames': frames[held].tolist(),
                 'old_fit_mean_depth_cm': old_full, 'old_fit_fps512_depth_cm': old_fps,
                 'old_held_mean_depth_cm': old_held_full, 'old_held_fps512_depth_cm': old_held_fps,
                 'reused_penetration_probe_sha256': hashlib.sha256(cached_path.read_bytes()).hexdigest() if cached else None,
                 'scope': 'Fit one option on 22 frames, check once on 11 disjoint frames. Full scored-frame review still required.'}
        (out/'probe.json').write_text(json.dumps(probe, indent=2)+'\n')
        (report/(stem+('_hand_contact_probe.json' if args.contact_aware else '_hand_safe_probe.json'))).write_text(json.dumps(probe, indent=2)+'\n')
        print('HAND_SAFE_COMPLETE', episode, bool(accepted), chosen_label, flush=True)
