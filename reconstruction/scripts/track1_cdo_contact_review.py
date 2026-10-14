"""Review object-only candidates against the actual submitted MHR parameters.

Measures signed object penetration into a watertight 4000-vertex human proxy,
plus unsigned hand-to-object TRIANGLE distance. Neither is a hidden GT score.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from kaolin.metrics.trianglemesh import point_to_mesh_distance

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, default=Path('/repo'))
p.add_argument('--episodes', type=int, nargs='+', default=[12, 14, 2, 1, 19, 20])
p.add_argument('--offset-probe', action='store_true', help='Fit a bounded Episode 20 correction with disjoint image/contact checks')
a = p.parse_args()
sys.path.insert(0, str(a.root / 'v2d_submission_kit/v2d_submission_kit/tools/track1'))
sys.path.insert(0, str(a.root / 'modules/v2d_cari4d/lib/cari4d'))
sys.path.insert(0, '/workspace/v2d_sam3d_body/lib')
if a.offset_probe:
    # Resolve the CARI4D tools package before importing SAM3D's model modules.
    sys.path.insert(0, '/workspace/v2d_foundation_pose/lib/FoundationPose')
    sys.path.insert(0, str(a.root / 'modules/v2d_cari4d/lib/cari4d'))
    from prep.run_foundationpose_mhr_export import _render_candidate_masks, _scaled_image_grid
from mesh_to_mhr_params import MHR
from lib_mhr import MHRLayer
from lib_mhr.collision_proxy import load_mhr_collision_proxy, apply_mhr_collision_proxy
from lib_mhr.hand_surface_contact import load_mhr_hand_surface_spec
from lib_mhr.sdf import object_inside_human_penetration_loss
from lib_mhr.contact import sample_mesh_surface_points

data = a.root / 'data/v2d_challenge/track_1'
report = a.root / 'incident_reports/track1_improvement/cdo_v4'
scored = json.loads((a.root / 'incident_reports/track1_scored_frames.json').read_text())
layer = MHRLayer.from_mhr_assets(mhr_assets_root=str(a.root / 'data/weights/sam3d_body'), device='cuda')
faces = layer.mesh_faces(device='cuda').cpu().numpy().astype(np.int32)
del layer
torch.cuda.empty_cache()
proxy = load_mhr_collision_proxy(faces)
spec = load_mhr_hand_surface_spec(a.root / 'modules/v2d_cari4d/lib/cari4d/lib_mhr/assets/mhr_hand_surface_spec.npz', faces=faces)
hand_ids = spec.vertex_indices.reshape(-1)[::8]
model = MHR(str(a.root / 'data/weights/track1_submission_mhr/mhr_model.pt'), 'cuda', 4, 'float32')
all_results = []
with torch.inference_mode():
    for episode in a.episodes:
        stem = f'episode_{episode:06d}'
        candidate_dir = data / 'cdo_v4_experiments/pose_candidates' / stem
        hoop_dir = data / 'cdo_v4_experiments/hoop_candidates' / stem
        if (hoop_dir / 'review.json').exists():
            candidate_dir = hoop_dir
        geometry_dir = data / 'cdo_v4_experiments/hoop_geometry' / stem
        if (geometry_dir / 'review.json').exists():
            candidate_dir = geometry_dir
        for group in ('physics_hoop', 'physics_hoop_v2', 'contact_projection'):
            optional_dir = data / 'cdo_v4_experiments' / group / stem
            if (optional_dir / 'review.json').exists() and not a.offset_probe:
                candidate_dir = optional_dir
        v3_dir = data / 'improvement_rollout_v1/candidate_episodes'
        baseline = dict(np.load(v3_dir / (stem + '.npz'), allow_pickle=False))
        candidate = dict(np.load(candidate_dir / (stem + '.npz'), allow_pickle=False))
        for key in ('pose', 'scales', 'shape'):
            assert np.array_equal(baseline[key], candidate[key]), (episode, key)
        frames = np.array(scored[str(episode)])
        frames = frames[np.linspace(0, len(frames)-1, min(33, len(frames))).round().astype(int)]
        human, _ = model.run(torch.as_tensor(baseline['pose'][frames], device='cuda', dtype=torch.float64),
                             torch.as_tensor(np.concatenate([baseline['scales'], baseline['shape']])[None], device='cuda', dtype=torch.float64))
        human = (human / 1000.).float()
        assert faces.max() < human.shape[1] and human.shape[1] == proxy.source_vertex_count
        collision_vertices = apply_mhr_collision_proxy(human, torch.as_tensor(proxy.source_vertex_indices.copy(), device='cuda', dtype=torch.long), torch.as_tensor(proxy.barycentric_weights.copy(), device='cuda'))
        summaries = {}
        for label, folder, bundle in [('submitted_v3', v3_dir, baseline), ('candidate', candidate_dir, candidate)]:
            mesh = trimesh.load(folder / (stem + '_object.glb'), force='mesh', process=False)
            local_vertices = np.asarray(mesh.vertices) * float(bundle['object_scale'])
            points = sample_mesh_surface_points(local_vertices, mesh.faces, 1000, seed=42)
            rotation = torch.as_tensor(bundle['object_rotation'][frames], device='cuda', dtype=torch.float32)
            translation = torch.as_tensor(bundle['object_translation'][frames], device='cuda', dtype=torch.float32)
            local = torch.as_tensor(points, device='cuda', dtype=torch.float32)
            world_points = torch.einsum('tij,pj->tpi', rotation, local) + translation[:, None]
            penetration = object_inside_human_penetration_loss(collision_vertices, proxy.faces.copy(), world_points,
                            frame_chunk_size=2, conservative_aabb_rejection=True)
            vertices = torch.as_tensor(local_vertices, device='cuda', dtype=torch.float32)
            mesh_faces = torch.as_tensor(mesh.faces, device='cuda', dtype=torch.long)
            proximity = []
            for j in range(len(frames)):
                hand = (human[j, hand_ids] - translation[j]) @ rotation[j]
                distance2, _, _ = point_to_mesh_distance(hand[None].contiguous(), vertices[mesh_faces][None].contiguous())
                proximity.append(float(distance2.min().sqrt().cpu()) * 100.)
            summaries[label] = {'signed_object_inside_human_proxy_cm': float(penetration.cpu()) * 100.,
                                'nearest_hand_object_surface_cm_median': float(np.median(proximity)),
                                'nearest_hand_object_surface_cm_p95': float(np.quantile(proximity, .95)),
                                'nearest_hand_object_surface_cm_samples': proximity}
            print('CONTACT_REVIEW_PROGRESS', episode, label, json.dumps({k:v for k,v in summaries[label].items() if not isinstance(v, list)}), flush=True)
        # Fixed human arrays certify equal CD-H/ACC-H package inputs, while
        # signed penetration is independently checked after changing objects.
        b, c = summaries['submitted_v3'], summaries['candidate']
        if a.offset_probe:
            assert episode == 20 and a.episodes == [20]
            import cv2
            import nvdiffrast.torch as dr
            from types import SimpleNamespace
            from prep.mhr_export_utils import read_mask, camera_calibration, load_edex
            from prep.run_foundationpose_mhr_export import _render_candidate_masks, _scaled_image_grid
            from prep.foundationpose_pose_selection import measure_visible_silhouettes
            from prep.trajectory_diagnostics import rigid_motion_diagnostics
            export = data / 'improvement_rollout_v1' / stem / 'run/export' / stem
            k, w2c = camera_calibration(load_edex(export), 0)
            assert np.allclose(w2c, np.eye(4))
            rend = SimpleNamespace(mesh_tensors={'pos': vertices, 'faces': mesh_faces.to(torch.int32)}, glctx=dr.RasterizeCudaContext())
            fit_ids = np.flatnonzero(np.arange(len(frames)) % 3 != 2)
            held_ids = np.flatnonzero(np.arange(len(frames)) % 3 == 2)
            options = []
            for offset in (-.08, -.04, 0., .04, .08):
                translations = np.asarray(candidate['object_translation']).copy()
                translations *= ((translations[:, 2] + offset) / translations[:, 2])[:, None]
                trans = torch.as_tensor(translations[frames], device='cuda')
                wp = torch.einsum('tij,pj->tpi', rotation, local) + trans[:, None]
                fit_pen = float(object_inside_human_penetration_loss(collision_vertices[fit_ids], proxy.faces.copy(), wp[fit_ids], frame_chunk_size=2, conservative_aabb_rejection=True).cpu()) * 100.
                held_pen = float(object_inside_human_penetration_loss(collision_vertices[held_ids], proxy.faces.copy(), wp[held_ids], frame_chunk_size=2, conservative_aabb_rejection=True).cpu()) * 100.
                ious = []
                for frame in frames:
                    name = f'{frame:06d}'
                    obj = read_mask(export, 'object', 0, name); human_mask = read_mask(export, 'human', 0, name)
                    small_k, shape = _scaled_image_grid(k, obj.shape, 768)
                    resize = lambda x: cv2.resize(x.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                    pose = np.eye(4); pose[:3, :3] = candidate['object_rotation'][frame]; pose[:3, 3] = translations[frame]
                    mask = _render_candidate_masks(rend, pose[None], small_k, shape)
                    ious.append(float(measure_visible_silhouettes(mask, resize(human_mask), resize(obj)).iou[0]))
                options.append({'offset_m': offset, 'fit_pen_cm': fit_pen, 'held_pen_cm': held_pen,
                                'fit_iou': float(np.mean(np.array(ious)[fit_ids])), 'held_iou': float(np.mean(np.array(ious)[held_ids])),
                                'mean_iou': float(np.mean(ious)), 'translations': translations})
                print('CONTACT_OFFSET_PROBE', episode, json.dumps({k:v for k,v in options[-1].items() if k != 'translations'}), flush=True)
            control = options[2]
            eligible = [o for o in options if o['fit_iou'] >= control['fit_iou'] - .015]
            chosen = min(eligible, key=lambda o: o['fit_pen_cm'])
            accepted = chosen['held_iou'] >= control['held_iou'] - .015 and chosen['held_pen_cm'] <= control['held_pen_cm']
            probe = {'fit_frames': frames[fit_ids].tolist(), 'held_out_frames': frames[held_ids].tolist(), 'accepted': bool(accepted),
                     'selected_offset_m': chosen['offset_m'], 'options': [{k:v for k,v in o.items() if k != 'translations'} for o in options]}
            if accepted and chosen['offset_m'] != 0.:
                destination = data / 'cdo_v4_experiments/contact_projection' / stem
                destination.mkdir(parents=True, exist_ok=True)
                assert not (destination / (stem + '.npz')).exists()
                candidate['object_translation'] = chosen['translations']
                np.savez_compressed(destination / (stem + '.npz'), **candidate)
                shutil.copy2(candidate_dir / (stem + '_object.glb'), destination / (stem + '_object.glb'))
                review = json.loads((candidate_dir / 'review.json').read_text())
                poses = np.tile(np.eye(4), (len(candidate['pose']), 1, 1))
                poses[:, :3, :3] = candidate['object_rotation']; poses[:, :3, 3] = candidate['object_translation']
                review['summary']['final_candidate'] = {'visible_iou_mean': chosen['mean_iou'], 'translation_acceleration_cm_proxy': float(np.linalg.norm(np.diff(poses[:, :3, 3], n=2, axis=0), axis=-1).mean()*100), **rigid_motion_diagnostics(poses, local_vertices)}
                review['candidate_npz'] = str(destination / (stem + '.npz')); review['contact_offset_probe'] = probe
                (destination / 'review.json').write_text(json.dumps(review, indent=2) + '\n')
            (report / (stem + '_contact_offset_probe.json')).write_text(json.dumps(probe, indent=2) + '\n')
        passed = c['signed_object_inside_human_proxy_cm'] <= max(.01, b['signed_object_inside_human_proxy_cm'] * 1.20)
        result = {'episode': episode, 'sample_count': len(frames), 'human_parameters_identical': True,
                  'candidate_npz_sha256': hashlib.sha256((candidate_dir / (stem + '.npz')).read_bytes()).hexdigest(),
                  'candidate_mesh_sha256': hashlib.sha256((candidate_dir / (stem + '_object.glb')).read_bytes()).hexdigest(),
                  'summary': summaries, 'contact_gate_pass': bool(passed),
                  'gate': 'Candidate signed penetration <= max(0.01 cm, 1.20 * submitted signed penetration)',
                  'scope': 'Predicted geometry, watertight human proxy, 1000 deterministic object surface samples. Not hidden-reference PEN/CD-O.'}
        (candidate_dir / 'contact_review.json').write_text(json.dumps(result, indent=2) + '\n')
        (report / (stem + '_contact_review.json')).write_text(json.dumps(result, indent=2) + '\n')
        all_results.append(result)
(report / 'contact_reviews.json').write_text(json.dumps(all_results, indent=2) + '\n')
