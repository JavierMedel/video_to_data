"""Track 3 public-development decomposition, mesh repair and feasibility tools."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import trimesh
from .common import KIT, REPO, load, save, sha, metric, new_run, complete, kit_imports
from .temporal import smooth


def support_plane(points, *, up=(0., 0., 1.), threshold=.005, iterations=500, seed=42):
    points, up = np.asarray(points, float), np.asarray(up, float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError('Expected finite scene support samples')
    if np.linalg.norm(up) < 1e-12 or threshold <= 0: raise ValueError('Invalid up/threshold')
    up = up/np.linalg.norm(up); rng = np.random.default_rng(seed); best = None
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b-a, c-a); norm = np.linalg.norm(normal)
        if norm < 1e-10: continue
        normal /= norm
        if normal@up < 0: normal = -normal
        if normal@up < np.cos(np.deg2rad(30)): continue
        inside = np.abs(points@normal-normal@a) <= threshold
        if best is None or inside.sum() > best.sum(): best = inside
    if best is None or best.sum() < 3: raise ValueError('No sufficiently horizontal support plane')
    center = points[best].mean(axis=0)
    _, _, vt = np.linalg.svd(points[best]-center, full_matrices=False)
    normal = vt[-1]
    if normal@up < 0: normal = -normal
    return {'normal': normal.tolist(), 'offset': float(-normal@center), 'inliers': int(best.sum()),
            'fraction': float(best.mean()), 'rms_m': float(np.sqrt(np.mean(((points[best]-center)@normal)**2))),
            'seed': seed}


def reference_diagnostics(positions, rotations, meshes, hands, plane, *, speed_threshold=.001):
    """Frame-aligned, observed-hand feasibility diagnostics, not task success."""
    positions, rotations, hands = map(lambda a: np.asarray(a, float), (positions, rotations, hands))
    if positions.ndim != 3 or positions.shape[-1] != 3 or rotations.shape != (*positions.shape[:2], 3, 3):
        raise ValueError('Expected positions T,B,3 and rotations T,B,3,3')
    if hands.ndim != 3 or hands.shape[0] != len(positions) or hands.shape[-1] != 3:
        raise ValueError('Missing time-aligned hand evidence')
    if not all(np.isfinite(a).all() for a in (positions, rotations, hands)):
        raise ValueError('Nonfinite reference/evidence')
    normal = np.asarray(plane['normal']); offset = plane['offset']
    if not np.isclose(np.linalg.norm(normal), 1): raise ValueError('Plane normal must be unit length')
    rows = []
    for body, (vertices, faces) in enumerate(meshes):
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        gaps, heights = [], []
        for frame in range(len(positions)):
            world = np.asarray(vertices)@rotations[frame, body].T+positions[frame, body]
            heights.append(float((world@normal+offset).min()))
            local = (hands[frame]-positions[frame, body])@rotations[frame, body]
            distances = []
            for start in range(0, len(local), 16):
                distances.extend(trimesh.proximity.closest_point_naive(mesh, local[start:start+16])[1])
            gaps.append(float(np.min(distances)))
        moving = np.r_[False, np.linalg.norm(np.diff(positions[:, body], axis=0), axis=1) > speed_threshold]
        floating = moving & (np.asarray(gaps) > .02)
        rows.append({'body': body, 'floating_motion_frames': np.flatnonzero(floating).tolist(),
                     'support_height_m': heights, 'nearest_hand_gap_m': gaps})
    return rows


def repair_reference(positions, rotations, meshes, plane, resting, *, frames, lam=10.):
    """Condition explicitly observed resting frames; never attach to guessed hands."""
    result = np.asarray(positions, float).copy()
    normal = np.asarray(plane['normal'], float); offset = float(plane['offset'])
    if not np.isclose(np.linalg.norm(normal), 1) or np.asarray(resting).shape != result.shape[:2]:
        raise ValueError('Resting labels/plane mismatch')
    for body, (vertices, _) in enumerate(meshes):
        targets = result[:, body].copy(); weight = np.zeros(len(result))
        for frame in np.flatnonzero(resting[:, body]):
            world = np.asarray(vertices)@rotations[frame, body].T+result[frame, body]
            height = (world@normal+offset).min()
            targets[frame] -= height*normal; weight[frame] = 100.
        result[:, body] = smooth(result[:, body], frames, lam, contact_target=targets, contact_weights=weight)
    return result


def cleanup(mesh, *, largest=True, fill=True, taubin=True):
    candidate = mesh.copy()
    if largest:
        parts = candidate.split(only_watertight=False)
        if not len(parts): raise ValueError('Empty connected-component result')
        candidate = max(parts, key=lambda part: part.area)
    if fill: trimesh.repair.fill_holes(candidate)
    if taubin: trimesh.smoothing.filter_taubin(candidate, lamb=.5, nu=.53, iterations=10)
    if not np.isfinite(candidate.vertices).all() or candidate.area <= 0:
        raise ValueError('Invalid repaired mesh')
    return candidate


def preserve_body_frame(parent, candidate, *, tolerance=.005):
    """Estimate candidate->parent rigid transform, apply it to mesh only."""
    official = metric('track_3', 'CD-O')
    parent_points = official._sample_mesh_surface(parent.vertices, parent.faces, 3000, 42)
    candidate_points = official._sample_mesh_surface(candidate.vertices, candidate.faces, 3000, 42)
    parent_radius = official._surface_moments(parent.vertices, parent.faces)[1]
    candidate_radius = official._surface_moments(candidate.vertices, candidate.faces)[1]
    ratio = candidate_radius/parent_radius
    if not .95 <= ratio <= 1.05: raise ValueError('Candidate has a body-frame scale discrepancy >5%')
    rotation, translation = official._align_rigid(parent_points, candidate_points)
    aligned = candidate.copy(); aligned.vertices = candidate.vertices@rotation.T+translation
    moved = candidate_points@rotation.T+translation
    distances = np.r_[cKDTree(parent_points).query(moved)[0], cKDTree(moved).query(parent_points)[0]]
    median = float(np.median(distances))
    if median > tolerance: raise ValueError('Body-frame median surface displacement exceeds 5 mm')
    transform = np.eye(4); transform[:3, :3] = rotation; transform[:3, 3] = translation
    return aligned, {'candidate_to_parent_transform': transform.tolist(), 'radius_ratio': float(ratio),
                     'median_bidirectional_surface_distance_m': median, 'poses_changed': False}


def dev_score(achieved, reference, candidates, scans, anchors, points, axes, orders, episode):
    """Official registration/symmetry functions against explicitly public dev assets."""
    kit_imports()
    from v2dlb import track3_pose as pose
    achieved, reference = np.asarray(achieved).copy(), np.asarray(reference).copy()
    if achieved.shape != reference.shape or achieved.ndim != 4 or achieved.shape[1] != 1:
        raise ValueError('Development scoring requires one-world matched T,1,B,7 grids')
    from v2dlb import object_pose as op
    achieved = op._normalise_quaternions(achieved, 'candidate', ValueError)
    reference = op._normalise_quaternions(reference, 'dev reference', ValueError)
    cd = metric('track_3', 'CD-O'); mesh_scores = []
    for slot in range(achieved.shape[2]):
        anchor, rotation = pose._register_object_body(candidates[slot], scans[slot], anchors[slot], 1000003*episode+101*slot)
        achieved[:, :, slot] = pose._convert_object_body(achieved[:, :, slot], anchor, rotation)
        reference[:, :, slot, :3] += np.einsum('...ij,j->...i', op._rotation_matrix(reference[:, :, slot, 3:]), anchors[slot])
        mesh_scores.append(cd._object_chamfer_cm(scans[slot], candidates[slot], cd._object_seed(episode, slot)))
    achieved = pose._align_registered_scene(achieved, reference, axes)
    centered = [p-a for p, a in zip(points, anchors)]
    values = {name: pose._registered_pose_metric(name, achieved, reference, centered, axes, orders)
              for name in ('add_auc', 'spider_sr', 'maniptrans_sr', 'rpe_cm', 'mppe_cm')}
    values['cd_o_cm'] = float(np.mean(mesh_scores))
    return values


def bottleneck(rows):
    """Aggregate all four arms across exactly the same development episodes."""
    arms = {arm: {r['episode_index']: r['metrics'] for r in rows if r['arm'] == arm} for arm in 'OMRA'}
    roster = set(arms['O'])
    if len(roster) != 4 or any(set(v) != roster for v in arms.values()): raise ValueError('Require four matched dev episodes in every O/M/R/A arm')
    means = {arm: {m: float(np.mean([r[m] for r in values.values()])) for m in ('add_auc', 'spider_sr')} for arm, values in arms.items()}
    if means['O']['spider_sr'] == 0: diagnosis = 'policy_training'
    else:
        mesh_loss = means['O']['add_auc']-means['M']['add_auc']
        reference_loss = means['O']['add_auc']-means['R']['add_auc']
        diagnosis = 'mesh' if mesh_loss > reference_loss else 'trajectory_and_world'
    return {'means': means, 'priority': diagnosis, 'scope': 'Development evidence; ablations can interact, no evaluation-score guarantee'}


def dev_jobs(manifest, output):
    from .policy import training_job
    data = load(manifest)
    if data.get('split') != 'development' or len(data['episodes']) != 4:
        raise ValueError('Choose exactly four publicly released development episodes')
    jobs = []
    for episode in data['episodes']:
        for arm in 'OMRA':
            reference = Path(episode['arms'][arm]['reference']).resolve()
            if 'evaluation' in reference.parts or 'confidential_holdout' in reference.parts:
                raise ValueError('Oracle experiment must never consume evaluation/withheld data')
            if sha(reference) != episode['arms'][arm]['reference_sha256']: raise ValueError('Dev arm reference changed')
            target = Path(output)/f'episode_{episode["episode_index"]:06d}'/arm/'policy'
            jobs.append({**training_job('track_3', episode['episode_index'], reference, target, 10000384),
                         'dev_arm': arm, 'split': 'development'})
    save(Path(output)/'jobs.json', jobs)
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    mesh = sub.add_parser('mesh')
    mesh.add_argument('--parent', type=Path, required=True); mesh.add_argument('--output', type=Path, required=True)
    dev = sub.add_parser('dev-jobs'); dev.add_argument('--manifest', type=Path, required=True); dev.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.action == 'dev-jobs': dev_jobs(a.manifest, a.output); return
    new_run(a.output, [a.parent], {'method': 'largest_fill_taubin'})
    parent = trimesh.load(a.parent, force='mesh', process=False)
    candidate, review = preserve_body_frame(parent, cleanup(parent))
    candidate.export(a.output/'candidate.glb')
    complete(a.output, 'MESH_REQUIRES_DEVELOPMENT_SCORE', **review, mesh_sha256=sha(a.output/'candidate.glb'))


if __name__ == '__main__': main()


def select_mesh_methods(records):
    """Choose a method per class on paired public development objects only."""
    if any(r.get('split') != 'development' for r in records):
        raise ValueError('Mesh method selection must use public development objects')
    decisions = {}
    for name in sorted({r['object_class'] for r in records}):
        group = [r for r in records if r['object_class'] == name]
        base = {r['episode_index']: r['cd_o_cm'] for r in group if r['method'] == 'baseline'}
        if not base: raise ValueError('Missing baseline mesh scores for '+name)
        means = {'baseline': float(np.mean(list(base.values())))}
        for method in sorted({r['method'] for r in group} - {'baseline'}):
            scores = {r['episode_index']: r['cd_o_cm'] for r in group if r['method'] == method}
            if set(scores) != set(base): raise ValueError('Mesh method dev episode cohort differs')
            means[method] = float(np.mean(list(scores.values())))
        best = min(means, key=means.get)
        chosen = best if means[best] <= .85*means['baseline'] else 'baseline'
        decisions[name] = {'method': chosen, 'means_cm': means, 'required_improvement_fraction': .15}
    return decisions
