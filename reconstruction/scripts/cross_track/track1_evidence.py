"""Participant-mask/camera/hand export with explicit raster and unit contracts."""
import argparse
from pathlib import Path
import sys
import numpy as np
import trimesh
from .common import REPO, load, save, sha, new_run, complete
from .image import silhouette, iou


def visible_iou(rendered, observed, human):
    return iou(np.asarray(rendered, bool) & ~human, np.asarray(observed, bool) & ~human)


def export_evidence(parent, episode, export_root, scored, model_path, hand_spec, output, *, max_side=384, device='cpu', historical_screen=None, metadata=None, keypoint_evidence=None):
    if not 0 <= episode < 30 or max_side < 32: raise ValueError('Invalid episode/grid')
    stem = f'episode_{episode:06d}'; parent, export_root, output = map(Path, (parent, export_root, output))
    bundle, mesh_path = parent/(stem+'.npz'), parent/(stem+'_object.glb')
    core = REPO/'reconstruction/modules/v2d_cari4d/lib/cari4d'
    sys.path.insert(0, str(core))
    from prep.mhr_export_utils import load_edex, camera_calibration, read_mask
    from .mhr_asset import script_model
    from .temporal import stretches
    import torch
    import cv2
    if historical_screen and device != 'cuda':
        raise ValueError('Historical equality uses its original GPU renderer; CPU IoU is a separate baseline')
    gpu_lock = None
    if device == 'cuda':
        from .gpu import start_guard
        gpu_lock = start_guard(output.parent/(output.name+'_gpu'))
    b = dict(np.load(bundle, allow_pickle=False)); frames = np.asarray(load(scored)[str(episode)], dtype=np.int64)
    stretches(frames, len(b['pose']))
    edex = load_edex(export_root)
    if len(edex) not in (1, len(b['pose'])): raise ValueError('Unrecognized static/framewise camera layout')
    k, w2c = camera_calibration(edex, 0)
    if not np.allclose(w2c, np.eye(4), atol=1e-6): raise ValueError('Track 1 camera frame must be verified identity')
    for frame in frames if len(edex) > 1 else []:
        other_k, other_w2c = camera_calibration(edex[int(frame):], 0)
        if not np.allclose(other_k, k) or not np.allclose(other_w2c, w2c): raise ValueError('Framewise camera varies; export calibrated per-frame evidence')
    metadata = Path(metadata or REPO/'reconstruction/data/v2d_challenge/track_1/meta/info.json')
    fps = float(load(metadata)['fps'])
    if not np.isfinite(fps) or fps <= 0: raise ValueError('Unknown source frame rate')
    # Hash actual modalities, including effective-mask sidecars that the reader uses.
    mask_inputs = [p for folder in ('masks', 'mask', 'segmentation') for p in (export_root/folder).rglob('*') if p.is_file()]
    mask_inputs += [p for p in export_root.rglob('*') if p.is_file() and ('mask' in p.name.lower() or 'effective' in p.name.lower())]
    if not mask_inputs:
        # Exact export layouts vary; do not omit modality provenance.
        mask_inputs = [p for p in export_root.rglob('*') if p.is_file() and p.suffix in ('.h5', '.hdf5', '.png')]
    inputs = [bundle, mesh_path, scored, model_path, hand_spec, export_root/'edex', metadata, *sorted(set(mask_inputs)),
              *([historical_screen] if historical_screen else []), *([keypoint_evidence] if keypoint_evidence else [])]
    new_run(output, inputs, {'episode': episode, 'max_side': max_side, 'device': device, 'method': 'participant_evidence_visible_masks_v1'})
    model = script_model(model_path).to(device=device, dtype=torch.float64)
    faces = model.character_torch.mesh.faces.cpu().numpy()
    with np.load(hand_spec, allow_pickle=False) as spec:
        hand_ids = np.asarray(spec['vertex_indices'], int).ravel()
        if str(spec['mhr_model_sha256'].item()) != sha(model_path): raise ValueError('Hand specification model digest mismatch')
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    object_masks, human_masks, hands, normals, intrinsics, baseline_ious, hand_visibility = [], [], [], [], [], [], []
    history_ious = []
    if historical_screen:
        sys.path.insert(0, str(REPO/'reconstruction/modules/v2d_foundation_pose/lib/FoundationPose'))
        from track1_mask_render import _render_candidate_masks
        import nvdiffrast.torch as dr
        from types import SimpleNamespace
        renderer = SimpleNamespace(mesh_tensors={'pos': torch.as_tensor(mesh.vertices*float(b['object_scale']), device='cuda', dtype=torch.float32),
                                               'faces': torch.as_tensor(mesh.faces, device='cuda', dtype=torch.int32)}, glctx=dr.RasterizeCudaContext())
    for frame in frames:
        name = f'{frame:06d}'
        obj = np.asarray(read_mask(export_root, 'object', 0, name), bool)
        human = np.asarray(read_mask(export_root, 'human', 0, name), bool)
        if obj.shape != human.shape or not obj.any(): raise ValueError('Missing/mismatched scored mask')
        ratio = min(1., max_side/max(obj.shape)); grid = tuple(max(1, int(round(v*ratio))) for v in obj.shape)
        scaled_k = k.copy(); scaled_k[0] *= grid[1]/obj.shape[1]; scaled_k[1] *= grid[0]/obj.shape[0]
        obj, human = [cv2.resize(v.astype(np.uint8), (grid[1], grid[0]), interpolation=cv2.INTER_NEAREST).astype(bool) for v in (obj, human)]
        with torch.no_grad():
            p = torch.tensor(np.r_[b['pose'][frame], b['scales']][None], dtype=torch.float64, device=device)
            sh = torch.tensor(b['shape'][None], dtype=torch.float64, device=device)
            vertices, _ = model(sh, p, torch.zeros((1, 72), dtype=torch.float64, device=device), True)
            vertices = vertices[0].cpu().numpy()/100*np.array([1, -1, -1])
        hn = np.asarray(trimesh.Trimesh(vertices, faces, process=False).vertex_normals)[hand_ids]
        norms = np.linalg.norm(hn, axis=1)
        if np.any(norms < 1e-8): raise ValueError('Undefined full-hand normals')
        hp = vertices[hand_ids]; hn = hn/norms[:, None]
        camera = hp@w2c[:3, :3].T+w2c[:3, 3]
        projected = camera@scaled_k.T
        valid = camera[:, 2] > 1e-5
        xy = np.zeros((len(hp), 2), int)
        xy[valid] = np.floor(projected[valid, :2]/projected[valid, 2:]).astype(int)
        inside = valid & (xy[:, 0] >= 0) & (xy[:, 0] < grid[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < grid[0])
        hit = np.zeros(len(hp), bool); hit[inside] = human[xy[inside, 1], xy[inside, 0]]
        hand_visibility.append({'frame': int(frame), 'fraction_in_image': float(inside.mean()), 'fraction_on_human_mask': float(hit.mean())})
        # Diagnostic visibility rates are not an invented universal 95% gate.
        rendered = silhouette(mesh.vertices, mesh.faces, b['object_rotation'][frame], b['object_translation'][frame],
                              float(b['object_scale']), scaled_k, w2c, grid)
        baseline_ious.append(visible_iou(rendered, obj, human))
        if historical_screen:
            if max_side != 384: raise ValueError('Historical screen uses max-side 384')
            pose = np.eye(4)[None]; pose[0, :3, :3] = b['object_rotation'][frame]; pose[0, :3, 3] = b['object_translation'][frame]
            historical = _render_candidate_masks(renderer, pose, scaled_k, grid)[0]
            history_ious.append(visible_iou(historical, obj, human))
        object_masks.append(obj); human_masks.append(human); intrinsics.append(scaled_k); hands.append(hp); normals.append(hn)
    history_review = None
    if historical_screen:
        with np.load(historical_screen, allow_pickle=False) as previous: archived = previous['ious'][frames, 0]
        delta = float(np.median(np.abs(np.asarray(history_ious)-archived)))
        history_review = {'median_absolute_iou_delta': delta, 'same_grid_and_occlusion': True, 'passed': delta <= .01}
        if not history_review['passed']: raise ValueError('Historical raster reproduction failed')
    destination = output/f'evidence_{stem}.npz'
    keypoints = {}
    if keypoint_evidence:
        with np.load(keypoint_evidence, allow_pickle=False) as raw: observed = dict(raw)
        if str(observed['parent_npz_sha256'].item()) != sha(bundle) or not np.array_equal(observed['frames'], frames):
            raise ValueError('Observed keypoint lineage/frame grid mismatch')
        indices, xy, visible = observed['keypoint_joint_indices'], observed['keypoints_xy'], observed['keypoint_visible']
        if (indices.ndim != 1 or indices.dtype.kind not in 'iu' or np.any(indices < 0) or np.any(indices >= 127)
                or xy.shape != (len(frames), len(indices), 2) or visible.shape != xy.shape[:-1]
                or visible.dtype.kind != 'b' or not np.isfinite(xy).all()):
            raise ValueError('Invalid observed keypoint grid')
        original_shape = np.asarray(observed['image_shape'], int)
        if original_shape.shape != (2,) or np.any(original_shape <= 0): raise ValueError('Unknown keypoint image grid')
        scaled_xy = xy*np.array([object_masks[0].shape[1]/original_shape[1], object_masks[0].shape[0]/original_shape[0]])
        keypoints = {'keypoint_joint_indices': indices, 'keypoints_xy': scaled_xy, 'keypoint_visible': visible}
        if 'hand_joint_indices' in observed:
            ids = observed['hand_joint_indices']
            if ids.dtype.kind not in 'iu' or ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= 127): raise ValueError('Invalid hand joint indices')
            keypoints['hand_joint_indices'] = ids
    np.savez_compressed(destination, parent_npz_sha256=sha(bundle), parent_mesh_sha256=sha(mesh_path), coordinate_frame='prediction_world_meters',
                        frames=frames, masks=np.asarray(object_masks), human_masks=np.asarray(human_masks),
                        intrinsics=np.asarray(intrinsics), world_to_camera=np.repeat(w2c[None], len(frames), axis=0),
                        hand_points=np.asarray(hands), hand_normals=np.asarray(normals), parent_visible_iou=np.asarray(baseline_ious), fps=fps, **keypoints)
    review = {'evidence_sha256': sha(destination), 'frames': frames.tolist(), 'hand_projections': hand_visibility,
              'baseline_visible_iou_mean': float(np.mean(baseline_ious)), 'historical_screen': history_review,
              'mask_contract': 'CPU candidate and parent use identical grid and human-occlusion exclusion',
              'scope': 'Participant hand specification, not an assertion of hidden host role coverage'}
    save(output/'evidence_review.json', review); complete(output, 'EVIDENCE_EXPORTED', **review)
    return review


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'export-root', 'scored', 'model', 'hand-spec', 'output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--episode', type=int, required=True); p.add_argument('--max-side', type=int, default=384)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu'); p.add_argument('--historical-screen', type=Path)
    p.add_argument('--metadata', type=Path); p.add_argument('--keypoint-evidence', type=Path)
    a = p.parse_args()
    export_evidence(a.parent, a.episode, a.export_root, a.scored, a.model, a.hand_spec, a.output,
                    max_side=a.max_side, device=a.device, historical_screen=a.historical_screen,
                    metadata=a.metadata, keypoint_evidence=a.keypoint_evidence)
