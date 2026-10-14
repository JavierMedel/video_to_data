"""Bounded temporal depth correction for Episode 1's known circular hoop.

The tube envelope is only a conservative proposal constraint. Promotion still
requires the independent, actual-mesh generalized-winding hand check.
"""
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
from scipy.optimize import lsq_linear
from scipy.sparse import diags, vstack

root = Path('/repo'); data = root/'data/v2d_challenge/track_1'; stem = 'episode_000001'
sys.path.insert(0, '/workspace/v2d_sam3d_body/lib')
sys.path.insert(0, '/workspace/v2d_foundation_pose/lib/FoundationPose')
sys.path.insert(0, str(root/'modules/v2d_cari4d/lib/cari4d'))
sys.path.insert(0, str(root/'v2d_submission_kit/v2d_submission_kit/tools/track1'))
from prep.run_foundationpose_mhr_export import _render_candidate_masks, _scaled_image_grid
from prep.mhr_export_utils import read_mask, camera_calibration, load_edex
from prep.foundationpose_pose_selection import measure_visible_silhouettes
from prep.trajectory_diagnostics import rigid_motion_diagnostics
from lib_mhr.hand_surface_contact import load_mhr_hand_surface_spec
from mesh_to_mhr_params import MHR

parent = data/'cdo_v6_experiments/hoop_geometry'/stem
out = data/'cdo_v6_experiments/hand_clearance_candidates'/stem
assert not out.exists(), 'Preserve completed candidate evidence'
out.mkdir(parents=True)
bundle = dict(np.load(parent/(stem+'.npz'), allow_pickle=False))
mesh = trimesh.load(parent/(stem+'_object.glb'), force='mesh', process=False)
vertices = mesh.vertices*float(bundle['object_scale'])
assert mesh.is_watertight and np.ptp(vertices[:, 0]) < .1*np.ptp(vertices[:, 1])
assert .95 < np.ptp(vertices[:, 1])/np.ptp(vertices[:, 2]) < 1.05
center = (vertices.min(0)+vertices.max(0))/2
radial = np.linalg.norm(vertices[:, 1:]-center[1:], axis=1)
radius = (radial.min()+radial.max())/2
vertex_tube_radius = np.hypot(vertices[:, 0]-center[0], radial-radius).max()
# Distance to the core circle is 1-Lipschitz. Every triangle point is within
# its longest edge of a vertex; this deliberately loose bound covers faces.
edge_bound = np.linalg.norm(vertices[mesh.edges_unique[:, 0]]-vertices[mesh.edges_unique[:, 1]], axis=1).max()
tube_bound = vertex_tube_radius+edge_bound+.002
assert tube_bound < .1*radius
spec = load_mhr_hand_surface_spec(root/'modules/v2d_cari4d/lib/cari4d/lib_mhr/assets/mhr_hand_surface_spec.npz')
model = MHR(str(root/'data/weights/track1_submission_mhr/mhr_model.pt'), 'cuda', 4, 'float32')
rotation = torch.as_tensor(bundle['object_rotation'], device='cuda', dtype=torch.float64)
translation = torch.as_tensor(bundle['object_translation'], device='cuda', dtype=torch.float64)
count = len(translation)
offsets = np.linspace(-.18, .18, 181)
clearance = np.empty((count, len(offsets)))
with torch.inference_mode():
    human, _ = model.run(torch.as_tensor(bundle['pose'], device='cuda', dtype=torch.float64), torch.as_tensor(np.r_[bundle['scales'], bundle['shape']][None], device='cuda', dtype=torch.float64))
    hands = human[:, spec.vertex_indices.reshape(-1)]/1000.
    for start in range(0, count, 8):
        stop = min(start+8, count)
        t = translation[start:stop]
        shifted = t[:, None]*(1+torch.as_tensor(offsets, device='cuda')[None, :, None]/t[:, None, 2:3])
        local = torch.einsum('tkpi,tij->tkpj', hands[start:stop, None]-shifted[:, :, None], rotation[start:stop])
        local -= torch.as_tensor(center, device='cuda')
        distance = torch.hypot(local[..., 0], torch.linalg.vector_norm(local[..., 1:], dim=-1)-radius)
        clearance[start:stop] = distance.amin(-1).cpu().numpy()-tube_bound
        if start % 80 == 0:
            print('HOOP_CLEARANCE_PROGRESS', start, count, flush=True)
safe = clearance > 0
assert safe.any(axis=1).all(), 'No bounded hand-clear option for some frames'
cost = .02*(offsets/.06)**2
previous = cost+np.where(safe[0], 0., 1e12)
parents = np.zeros(safe.shape, dtype=np.int32)
transition = .02*((offsets[:, None]-offsets[None, :])/.01)**2
for t in range(1, count):
    values = previous[:, None]+transition
    parents[t] = values.argmin(0)
    previous = cost+values[parents[t], np.arange(len(offsets))]+np.where(safe[t], 0., 1e12)
path = np.empty(count, dtype=np.int32); path[-1] = previous.argmin()
for t in range(count-1, 0, -1):
    path[t-1] = parents[t, path[t]]
assert safe[np.arange(count), path].all()
# A conservative safe interval follows from the same distance Lipschitz bound.
# Bound the 3-D ray displacement, rather than treating delta-Z as distance.
rays = bundle['object_translation']/bundle['object_translation'][:, 2:3]
half_width = np.minimum(.0015, clearance[np.arange(count), path]/np.linalg.norm(rays, axis=1)*.8)
lower = np.maximum(-.18, offsets[path]-half_width); upper = np.minimum(.18, offsets[path]+half_width)
d2 = diags([np.ones(count-2), -2*np.ones(count-2), np.ones(count-2)], [0, 1, 2], shape=(count-2, count))
d1 = diags([-np.ones(count-1), np.ones(count-1)], [0, 1], shape=(count-1, count))
blocks = [diags(np.ones(count)*.1), d1*.3]
targets = [np.zeros(count), np.zeros(count-1)]
for axis in range(3):
    blocks.append(d2@diags(rays[:, axis])*10.)
    targets.append(-(d2@bundle['object_translation'][:, axis])*10.)
fit = lsq_linear(vstack(blocks).tocsr(), np.concatenate(targets), bounds=(lower, upper), tol=1e-9, max_iter=500)
assert fit.success, fit.message
final = np.tile(np.eye(4), (count, 1, 1))
final[:, :3, :3] = bundle['object_rotation']
final[:, :3, 3] = bundle['object_translation']+rays*fit.x[:, None]
export = data/'improvement_rollout_v1'/stem/'run/export'/stem
k, w2c = camera_calibration(load_edex(export), 0); assert np.allclose(w2c, np.eye(4))
renderer = SimpleNamespace(mesh_tensors={'pos': torch.as_tensor(vertices, device='cuda', dtype=torch.float32), 'faces': torch.as_tensor(mesh.faces, device='cuda', dtype=torch.int32)}, glctx=dr.RasterizeCudaContext())
frames = np.array(json.loads((root/'incident_reports/track1_scored_frames.json').read_text())['1'])
ious = []
with torch.inference_mode():
    for j, frame in enumerate(frames):
        obj = read_mask(export, 'object', 0, f'{frame:06d}'); hm = read_mask(export, 'human', 0, f'{frame:06d}')
        small_k, shape = _scaled_image_grid(k, obj.shape, 768)
        resize = lambda x: cv2.resize(x.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)>0
        mask = _render_candidate_masks(renderer, final[frame:frame+1], small_k, shape)
        ious.append(float(measure_visible_silhouettes(mask, resize(hm), resize(obj)).iou[0]))
        if j % 80 == 0:
            print('HOOP_IMAGE_REVIEW_PROGRESS', j, len(frames), flush=True)
summary = {'visible_iou_mean': float(np.mean(ious)), 'translation_acceleration_cm_proxy': float(np.linalg.norm(np.diff(final[:, :3, 3], n=2, axis=0), axis=1).mean()*100), **rigid_motion_diagnostics(final, vertices)}
bundle.update(object_translation=final[:, :3, 3].astype(np.float32))
np.savez_compressed(out/(stem+'.npz'), **bundle)
shutil.copy2(parent/(stem+'_object.glb'), out/(stem+'_object.glb'))
review = json.loads((parent/'review.json').read_text()); review['summary']['final_candidate'] = summary
review['hand_clearance'] = {'tube_bound_m': float(tube_bound), 'core_radius_m': float(radius), 'edge_bound_m': float(edge_bound), 'offset_bound_m': .18,
    'offset_min_m': float(fit.x.min()), 'offset_max_m': float(fit.x.max()), 'scored_frames': frames.tolist(), 'per_frame_iou': ious,
    'scope': 'Direct predicted-hand constraints on all video frames; no hidden ground truth. Image IoU evaluated on every scored frame. Actual-mesh penetration validation remains mandatory.'}
(out/'review.json').write_text(json.dumps(review, indent=2)+'\n')
np.savez_compressed(out/'clearance_selection.npz', offsets=offsets, clearance=clearance, path=path, final_offsets=fit.x, lower=lower, upper=upper)
print('HOOP_HAND_CLEARANCE_COMPLETE', json.dumps(summary), flush=True)
