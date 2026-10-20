"""Decoded-joint diagnostics, root-only fit and differentiable articulation fit."""
import numpy as np
from scipy.spatial.transform import Rotation
from .temporal import smooth, smooth_rotations, acceleration, stretches


def root_columns(parameter_transform):
    transform = np.asarray(parameter_transform)
    # Root translation/rotation must have an independent, full-rank pose block.
    out = []
    for rows in (slice(0, 3), slice(3, 6)):
        columns = np.flatnonzero(np.any(np.abs(transform[rows, :136]) > 1e-12, axis=0))
        if len(columns) != 3 or np.linalg.matrix_rank(transform[rows, columns]) != 3:
            raise ValueError('Root parameter mapping is not a 3-DOF independent block')
        other = transform[:, columns].copy(); other[rows] = 0
        if np.any(np.abs(other) > 1e-12): raise ValueError('Root parameters also drive local articulation')
        out.append(columns)
    return out


def decompose(body, pose, scales, shape, frames):
    params = np.c_[pose, np.broadcast_to(scales, (len(pose), len(scales)))].astype(float)
    _, joints = body(params, shape)
    columns_t, columns_r = root_columns(body.parameter_transform)
    fixed = params.copy(); fixed[:, columns_t] = params[frames[0], columns_t]
    _, no_translation = body(fixed, shape)
    fixed[:, columns_r] = params[frames[0], columns_r]
    _, articulation = body(fixed, shape)
    from .common import metric
    selected = list(metric('track_1', 'ACC-O').MHR_TABLE3_BODY_JOINT_INDICES)
    joints, no_translation, articulation = [v[:, selected] for v in (joints, no_translation, articulation)]
    translation, rotation = joints-no_translation, no_translation-articulation
    root = translation+rotation
    total = acceleration(joints, frames)
    magnitude = acceleration(root, frames)
    return {'total_acceleration_cm': total, 'root_translation_acceleration_cm': acceleration(translation, frames),
            'root_rotation_acceleration_cm': acceleration(rotation, frames),
            'articulation_acceleration_cm': acceleration(articulation, frames),
            'root_acceleration_magnitude_ratio': magnitude/max(total, 1e-12),
            'recommended_arm': 'B1' if magnitude/max(total, 1e-12) >= .6 else 'B2',
            'scope': 'Magnitude decomposition is not additive energy or hidden-reference error'}


def root_fit(body, pose, scales, frames, lam):
    from .common import metric
    op = metric('track_1', 'ACC-O')
    result = np.asarray(pose).copy()
    tcols, rcols = root_columns(body.parameter_transform)
    params = np.c_[pose, np.broadcast_to(scales, (len(pose), len(scales)))]
    joint = params@body.parameter_transform[:7].T
    translations = smooth(joint[:, :3]/100, frames, lam)*100
    result[:, tcols] += (translations-joint[:, :3])@np.linalg.inv(body.parameter_transform[:3, tcols]).T
    base = body.pre_rotation[0]@op._euler_xyz_to_mat(joint[:, 3:6])
    rotations = smooth_rotations(base, frames, lam)
    angles = Rotation.from_matrix(body.pre_rotation[0].T@rotations).as_euler('xyz')
    # Map to the nearest equivalent branch, then verify the actual decoded geometry.
    delta = np.arctan2(np.sin(angles-joint[:, 3:6]), np.cos(angles-joint[:, 3:6]))
    result[:, rcols] += delta@np.linalg.inv(body.parameter_transform[3:6, rcols]).T
    result[frames[0]] = pose[frames[0]]
    unused = np.setdiff1d(np.arange(len(pose)), frames)
    result[unused] = pose[unused]
    return result


def refit(model, pose, scales, shape, frames, joint_indices, *, lam=30., prior=.001, steps=100, lr=.001):
    """Optimise decoded body/hand joints, keeping identity and anchor immutable."""
    import torch
    device = model.flip.device
    base = torch.as_tensor(pose, dtype=torch.float64, device=device)
    variable = torch.nn.Parameter(base.clone())
    identity = torch.as_tensor(scales, dtype=torch.float64, device=device).expand(len(pose), -1)
    with torch.no_grad(): target = model.joints(torch.cat([base, identity], -1))[:, joint_indices]
    optimizer = torch.optim.Adam([variable], lr=lr)
    runs = stretches(frames, len(pose))
    scored = torch.tensor(np.concatenate(runs), device=device)
    triples = np.concatenate([np.stack([r[:-2], r[1:-1], r[2:]], 1) for r in runs if len(r) >= 3])
    triples = torch.tensor(triples, device=device)
    for _ in range(steps):
        optimizer.zero_grad()
        joints = model.joints(torch.cat([variable, identity], -1))[:, joint_indices]
        curvature = joints[triples[:, 0]]-2*joints[triples[:, 1]]+joints[triples[:, 2]]
        loss = (joints[scored]-target[scored]).square().mean()+lam*curvature.square().mean()+prior*(variable-base).square().mean()
        if not torch.isfinite(loss): raise ValueError('Joint-space fit became nonfinite')
        loss.backward()
        variable.grad[frames[0]] = 0
        optimizer.step()
        with torch.no_grad():
            variable[frames[0]] = base[frames[0]]
            # The objective only authorizes scored frames.
            unused = np.setdiff1d(np.arange(len(pose)), np.asarray(frames))
            variable[unused] = base[unused]
    return variable.detach().cpu().numpy().astype(np.asarray(pose).dtype)


def evidence_gate(before_vertices, after_vertices, frames, hand_indices, before_error_px, after_error_px,
                  *, articulation=False):
    if before_error_px is None or after_error_px is None:
        raise ValueError('Human promotion requires observed 2D keypoint reprojection evidence')
    displacement = np.linalg.norm(after_vertices[frames]-before_vertices[frames], axis=-1)
    checks = {'body_p99': float(np.quantile(displacement, .99)) <= .03,
              'hand_contact_max': float(displacement[:, hand_indices].max()) <= .015,
              'reprojection': float(np.mean(after_error_px)) <= 1.05*float(np.mean(before_error_px))+1e-12,
              'anchor': np.array_equal(before_vertices[frames[0]], after_vertices[frames[0]])}
    if articulation: checks['body_max'] = float(displacement.max()) <= .05
    return {'passed': all(checks.values()), 'checks': checks,
            'state': 'READY_FOR_COUPLED_OBJECT_REVIEW' if all(checks.values()) else 'REJECTED'}


def main():
    import argparse
    from pathlib import Path
    from .common import metric, sha, new_run, complete, save
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'body-asset', 'evidence', 'output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--arm', choices=['root', 'refit'], default='root')
    p.add_argument('--lambda', dest='lam', type=float, default=30.)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    previous = new_run(a.output, [a.bundle, a.body_asset, a.evidence], {'arm': a.arm, 'lambda': a.lam})
    if previous['state'] in ('READY_FOR_COUPLED_OBJECT_REVIEW', 'REJECTED'):
        if sha(a.output/a.bundle.name) != previous['npz_sha256']:
            raise ValueError('Completed human candidate changed; preserve for review')
        return
    gpu_lock = None
    if a.device.startswith('cuda'):
        from .gpu import start_guard
        gpu_lock = start_guard(a.output / 'gpu')
    bundle = dict(np.load(a.bundle, allow_pickle=False)); evidence = dict(np.load(a.evidence, allow_pickle=False))
    if str(evidence['parent_npz_sha256'].item()) != sha(a.bundle): raise ValueError('Human evidence parent changed')
    frames = evidence['frames']; official = metric('track_1', 'ACC-O'); body = official._load_body(a.body_asset)
    diagnostic = decompose(body, bundle['pose'], bundle['scales'], bundle['shape'], frames)
    save(a.output/'diagnostics.json', diagnostic)
    params = np.c_[bundle['pose'], np.broadcast_to(bundle['scales'], (len(bundle['pose']), 68))]
    before_vertices, before_joints = body(params, bundle['shape'], chunk=8)
    if a.arm == 'root': candidate = root_fit(body, bundle['pose'], bundle['scales'], frames, a.lam)
    else:
        import torch
        from .mhr_torch import MHRBody, verify_parity
        asset = dict(np.load(a.body_asset, allow_pickle=False))
        model = MHRBody(asset).to(a.device)
        samples = np.random.default_rng(42).choice(len(params), size=100, replace=False)
        parity = verify_parity(model, body, params[samples], bundle['shape'])
        save(a.output/'decoder_parity.json', {'maximum_error_m': parity, 'sampled_frames': samples.tolist(), 'body_asset_sha256': sha(a.body_asset)})
        candidate = refit(model, bundle['pose'], bundle['scales'], bundle['shape'], frames,
                          sorted(set(official.MHR_TABLE3_BODY_JOINT_INDICES) | set(evidence['hand_joint_indices'].tolist())), lam=a.lam)
    after_vertices, after_joints = body(np.c_[candidate, params[:, 136:]], bundle['shape'], chunk=8)
    indices = evidence['keypoint_joint_indices']
    def errors(joints):
        values = []
        for i, frame in enumerate(frames):
            xyz = joints[frame, indices]@evidence['world_to_camera'][i, :3, :3].T+evidence['world_to_camera'][i, :3, 3]
            if (xyz[:, 2] <= 0).any(): raise ValueError('Human joint lies behind the evidence camera')
            k = evidence['intrinsics'][i] if evidence['intrinsics'].ndim == 3 else evidence['intrinsics']
            projected = xyz@k.T; xy = projected[:, :2]/projected[:, 2:]
            good = evidence['keypoint_visible'][i].astype(bool)
            if not good.any(): raise ValueError('Missing visible keypoint evidence on a scored frame')
            values.extend(np.linalg.norm(xy[good]-evidence['keypoints_xy'][i, good], axis=1))
        return values
    hands = np.concatenate([body.roles['left_hand'], body.roles['right_hand']])
    verdict = evidence_gate(before_vertices, after_vertices, frames, hands, errors(before_joints), errors(after_joints), articulation=a.arm == 'refit')
    metric_joints = list(official.MHR_TABLE3_BODY_JOINT_INDICES)
    control_acc = acceleration(before_joints[:, metric_joints], frames)
    candidate_acc = acceleration(after_joints[:, metric_joints], frames)
    verdict['checks']['joint_acceleration_nonincrease'] = candidate_acc <= control_acc+1e-10
    verdict['passed'] = all(verdict['checks'].values())
    verdict['state'] = 'READY_FOR_COUPLED_OBJECT_REVIEW' if verdict['passed'] else 'REJECTED'
    verdict.update(control_joint_acceleration_cm=control_acc, candidate_joint_acceleration_cm=candidate_acc)
    if candidate[frames[0]].tobytes() != bundle['pose'][frames[0]].tobytes(): raise ValueError('Human pose anchor changed')
    bundle['pose'] = candidate if verdict['passed'] else bundle['pose']
    output = a.output/a.bundle.name; np.savez_compressed(output, **bundle)
    complete(a.output, verdict['state'], **{k: v for k, v in verdict.items() if k != 'state'},
             evaluated_frames=frames.tolist(), npz_sha256=sha(output))


if __name__ == '__main__': main()
