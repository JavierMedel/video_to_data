"""Review hand-inside-object depth, using the released PEN geometry definition.

The participant kit omits the exact host hand-role subset and GT alignment
scale. Report full 4636-vertex hands and a labelled 512-vertex FPS diagnostic;
do not mislabel these as official PEN scores.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, default=Path('/repo'))
p.add_argument('--episodes', type=int, nargs='+', default=[23, 18, 21, 16, 13])
p.add_argument('--sample-count', type=int, default=33, help='0 evaluates every scored frame')
p.add_argument('--audit-v4', action='store_true', help='Compare the five v4 changes against v3')
p.add_argument('--iteration', type=int, choices=[5, 6, 7], default=5)
p.add_argument('--candidate-group', help='Evaluate one isolated candidate group')
a = p.parse_args()
assert not a.audit_v4 or (a.iteration == 5 and a.candidate_group is None)
assert a.candidate_group is None or Path(a.candidate_group).name == a.candidate_group
sys.path.insert(0, '/workspace/v2d_sam3d_body/lib')
sys.path.insert(0, str(a.root/'modules/v2d_cari4d/lib/cari4d'))
sys.path.insert(0, str(a.root/'v2d_submission_kit/v2d_submission_kit/tools/track1'))
from mesh_to_mhr_params import MHR
from lib_mhr.hand_surface_contact import load_mhr_hand_surface_spec
from lib_mhr.mesh_penetration_review import MeshPenetrationReview

data = a.root/'data/v2d_challenge/track_1'
report = a.root/f'incident_reports/track1_improvement/cdo_v{a.iteration}'
if a.candidate_group:
    report = report/a.candidate_group
report.mkdir(parents=True, exist_ok=True)
scored = json.loads((a.root/'incident_reports/track1_scored_frames.json').read_text())
spec = load_mhr_hand_surface_spec(a.root/'modules/v2d_cari4d/lib/cari4d/lib_mhr/assets/mhr_hand_surface_spec.npz')
hand_ids = spec.vertex_indices.reshape(-1)
sample_ids = (spec.sample_local_indices + np.arange(2)[:, None]*spec.vertex_indices.shape[1]).reshape(-1)
model = MHR(str(a.root/'data/weights/track1_submission_mhr/mhr_model.pt'), 'cuda', 4, 'float32')
results = []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


with torch.inference_mode():
    for episode in a.episodes:
        started = time.monotonic()
        stem = f'episode_{episode:06d}'
        before_dir = data/('improvement_rollout_v1/candidate_episodes' if a.audit_v4 else f'cdo_v{a.iteration-1}_experiments/candidate_episodes')
        after_dir = data/('cdo_v4_experiments/candidate_episodes' if a.audit_v4 else f'cdo_v{a.iteration}_experiments/{a.candidate_group or "pose_candidates"}/{stem}')
        hand_safe = data/f'cdo_v{a.iteration}_experiments/hand_safe_candidates'/stem
        if not a.audit_v4 and not a.candidate_group and (hand_safe/'review.json').exists():
            after_dir = hand_safe
        hand_contact = data/f'cdo_v{a.iteration}_experiments/hand_contact_candidates'/stem
        if not a.audit_v4 and not a.candidate_group and (hand_contact/'review.json').exists():
            after_dir = hand_contact
        output_name = 'hand_penetration_review' + ('.sampled' if a.sample_count else '') + '.json'
        frames = np.array(scored[str(episode)])
        if a.sample_count:
            frames = frames[np.linspace(0, len(frames)-1, min(a.sample_count, len(frames))).round().astype(int)]
        existing = None
        if not a.audit_v4 and (after_dir/output_name).exists():
            existing = json.loads((after_dir/output_name).read_text())
            assert existing['candidate_npz_sha256'] == digest(after_dir/(stem+'.npz'))
            assert existing['candidate_mesh_sha256'] == digest(after_dir/(stem+'_object.glb'))
            assert existing['control_npz_sha256'] == digest(before_dir/(stem+'.npz'))
            assert existing['control_mesh_sha256'] == digest(before_dir/(stem+'_object.glb'))
            assert existing['all_scored_frames'] == (a.sample_count == 0)
            assert existing['frames'] == frames.tolist()
            if 'grasp_proximity_gate_pass' in existing:
                results.append(existing)
                print('HAND_PEN_REVIEW_REUSED', episode, flush=True)
                continue
        before = dict(np.load(before_dir/(stem+'.npz'), allow_pickle=False))
        after = dict(np.load(after_dir/(stem+'.npz'), allow_pickle=False))
        for key in ('pose', 'scales', 'shape'):
            assert np.array_equal(before[key], after[key]), (episode, key)
        human, _ = model.run(torch.as_tensor(before['pose'][frames], device='cuda', dtype=torch.float64),
                             torch.as_tensor(np.r_[before['scales'], before['shape']][None], device='cuda', dtype=torch.float64))
        hands = (human[:, hand_ids]/1000.).double()
        summaries = {}; depths_by_version = {}; proximities = {}
        for label, folder, bundle in [('control', before_dir, before), ('candidate', after_dir, after)]:
            mesh = trimesh.load(folder/(stem+'_object.glb'), force='mesh', process=False)
            check = MeshPenetrationReview(mesh.vertices, mesh.faces)
            rotation = torch.as_tensor(bundle['object_rotation'][frames], device='cuda', dtype=torch.float64)
            translation = torch.as_tensor(bundle['object_translation'][frames], device='cuda', dtype=torch.float64)
            scale = float(bundle['object_scale'])
            local = torch.einsum('tpi,tij->tpj', hands-translation[:, None], rotation)/scale
            nearest = cKDTree(np.asarray(mesh.vertices)).query(local[:, sample_ids].cpu().numpy().reshape(-1, 3), workers=4)[0]
            proximities[label] = nearest.reshape(len(frames), len(sample_ids)).min(axis=1)*scale*100.
            if existing is not None:
                summaries[label] = existing['summary'][label]
                continue
            # Frame chunks preserve progress and bound temporary GPU memory.
            chunks = []
            for start in range(0, len(frames), 8):
                depths = check.depths(local[start:start+8]).cpu().numpy()*scale*100.
                chunks.append(depths)
                if start % 32 == 0:
                    print('HAND_PEN_PROGRESS', episode, label, start, len(frames), flush=True)
            depth = np.concatenate(chunks); depths_by_version[label] = depth
            summaries[label] = {'mean_full_hand_depth_cm_unaligned': float(depth.mean()),
                                'mean_fps512_depth_cm_unaligned': float(depth[:, sample_ids].mean()),
                                'per_frame_full_hand_depth_cm': depth.mean(axis=1).tolist(),
                                'max_hand_vertex_depth_cm': float(depth.max()),
                                'penetrating_hand_vertex_fraction': float((depth > 0.).mean())}
            print('HAND_PEN_SUMMARY', episode, label, summaries[label]['mean_full_hand_depth_cm_unaligned'], summaries[label]['mean_fps512_depth_cm_unaligned'], flush=True)
            del check
        b, c = summaries['control'], summaries['candidate']
        # Require non-increasing means for BOTH independent hand selections.
        # A very small numerical allowance is only 1e-8 cm in the mean.
        passed = c['mean_full_hand_depth_cm_unaligned'] <= b['mean_full_hand_depth_cm_unaligned']+1e-8 and c['mean_fps512_depth_cm_unaligned'] <= b['mean_fps512_depth_cm_unaligned']+1e-8
        difference = depths_by_version['candidate']-depths_by_version['control'] if existing is None else None
        contact_like = proximities['control'] <= 1.5
        increase = proximities['candidate'][contact_like]-proximities['control'][contact_like]
        proximity_pass = not contact_like.any() or (float(np.median(increase)) <= 1.0 and float(np.quantile(increase, .95)) <= 2.0)
        result = {'episode': episode, 'sample_count': len(frames), 'frames': frames.tolist(), 'all_scored_frames': not bool(a.sample_count),
                  'human_arrays_identical': True, 'summary': summaries, 'hand_penetration_gate_pass': bool(passed),
                  'hand_vertex_count': len(hand_ids), 'fps_sample_vertex_count': len(sample_ids),
                  'max_pointwise_depth_increase_cm': float(difference.max()) if existing is None else existing['max_pointwise_depth_increase_cm'],
                  'pointwise_nonincrease': bool((difference <= 1e-6).all()) if existing is None else existing['pointwise_nonincrease'],
                  'grasp_proximity_gate_pass': bool(proximity_pass),
                  'grasp_proximity': {'control_contact_like_frames': int(contact_like.sum()), 'control_threshold_cm': 1.5,
                      'median_gap_increase_cm': float(np.median(increase)) if len(increase) else None,
                      'p95_gap_increase_cm': float(np.quantile(increase, .95)) if len(increase) else None,
                      'control_nearest_vertex_cm': proximities['control'].tolist(), 'candidate_nearest_vertex_cm': proximities['candidate'].tolist(),
                      'scope': 'Unsigned nearest mesh-vertex proxy on local FPS512 points; checks only control-contact-like frames. This is not a grasp label or a penetration calculation.'},
                  'candidate_npz_sha256': digest(after_dir/(stem+'.npz')), 'candidate_mesh_sha256': digest(after_dir/(stem+'_object.glb')),
                  'control_npz_sha256': digest(before_dir/(stem+'.npz')), 'control_mesh_sha256': digest(before_dir/(stem+'_object.glb')),
                  'elapsed_seconds': time.monotonic()-started,
                  'scope': 'Released generalized-winding containment and triangle distance; full predicted hands plus local FPS512 subset. Exact host hand-role IDs and positive GT alignment scale are unavailable. This is not an official PEN score.'}
        name = f'{stem}_' + ('v4_penetration_audit' if a.audit_v4 else 'hand_penetration_review') + ('.sampled' if a.sample_count else '') + '.json'
        (report/name).write_text(json.dumps(result, indent=2)+'\n')
        if not a.audit_v4:
            (after_dir/output_name).write_text(json.dumps(result, indent=2)+'\n')
        results.append(result)
        print('HAND_PEN_REVIEW_COMPLETE', episode, 'PASS' if passed else 'REJECT', 'seconds', result['elapsed_seconds'], flush=True)
name = ('v4_penetration_audit' if a.audit_v4 else 'hand_penetration_reviews') + ('.sampled' if a.sample_count else '') + '.json'
(report/name).write_text(json.dumps(results, indent=2)+'\n')
