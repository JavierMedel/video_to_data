"""Recentered, contact-constrained Track 1 candidates with measured evidence gates."""
import argparse
from pathlib import Path
import numpy as np
import trimesh
from .common import load, save, sha, new_run, complete, metric
from .temporal import smooth, acceleration, qp, gcv_lambda
from scipy import sparse
from .image import silhouette, iou
from track1_metric_harness import recenter, verify_recenter


def penetration_constraints(vertices, faces, rotations, translations, scale, hands, frames, epsilon=.001, return_contacts=False):
    """Linearize inside/near hand-point constraints in world translation space."""
    official = metric('track_1', 'ACC-O')
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    constraints, depths, gaps, contacts = [], [], [], []
    for index, frame in enumerate(frames):
        local = (hands[index]-translations[frame])@rotations[frame]/scale
        depth = official._penetration_depth(local, vertices, faces)*scale
        depths.append(depth)
        # Bounded point chunks avoid the O(points*faces) temporary memory spike.
        closest, distance, face_index = [], [], []
        for start in range(0, len(local), 16):
            c, d, f = trimesh.proximity.closest_point_naive(mesh, local[start:start+16])
            closest.extend(c); distance.extend(d); face_index.extend(f)
        closest, distance, face_index = np.asarray(closest), np.asarray(distance), np.asarray(face_index)
        gaps.append(float(distance.min()*scale))
        contacts.append(hands[index, int(np.argmin(distance))])
        for point in np.flatnonzero((depth > 0) | (distance*scale < epsilon)):
            direction = closest[point]-local[point] if depth[point] > 0 else local[point]-closest[point]
            norm = np.linalg.norm(direction)
            normal = direction/norm if norm > 1e-12 else mesh.face_normals[face_index[point]]
            normal = rotations[frame]@normal
            bound = normal@(hands[index, point]-scale*rotations[frame]@closest[point])-epsilon
            constraints.append((int(frame), normal, float(bound)))
    result = (constraints, np.asarray(depths), np.asarray(gaps))
    return (*result, np.asarray(contacts)) if return_contacts else result


def gate(base_t, candidate_t, frames, base_iou, candidate_iou, base_gap, candidate_gap, depths):
    displacement = np.linalg.norm(candidate_t[frames]-base_t[frames], axis=1)
    active = base_gap <= .02
    gap_delta = float(np.median(np.abs(candidate_gap[active]-base_gap[active]))) if active.any() else 0.
    checks = {'mean_displacement': float(displacement.mean()) <= .005,
              'p99_displacement': float(np.quantile(displacement, .99)) <= .03,
              'mean_mask_iou': float(np.mean(base_iou-candidate_iou)) <= .01+1e-12,
              'worst_mask_iou': float(np.max(base_iou-candidate_iou)) <= .05+1e-12,
              'contact_gap': gap_delta <= .01,
              'zero_hand_penetration': not np.any(depths > 0)}
    return {'passed': all(checks.values()), 'checks': checks,
            'mean_displacement_m': float(displacement.mean()), 'p99_displacement_m': float(np.quantile(displacement, .99)),
            'median_contact_gap_change_m': gap_delta, 'penetrating_points': int(np.count_nonzero(depths))}


def contact_targets(hands, translations, frames, active):
    """Keep the median observed hand-to-object offset per contiguous contact window."""
    centers = hands.mean(axis=1)
    target = translations.copy(); weight = np.zeros(len(translations))
    contact = np.flatnonzero(active)
    if not len(contact): return target, weight
    breaks = np.flatnonzero((np.diff(contact) != 1) | (np.diff(frames[contact]) != 1))+1
    for window in np.split(contact, breaks):
        offset = np.median(translations[frames[window]]-centers[window], axis=0)
        target[frames[window]] = centers[window]+offset
        weight[frames[window]] = 1.
    return target, weight


def object_candidate(bundle_path, mesh_path, evidence_path, output, lambdas=(3, 10, 30, 100, 300), *, lambda_policy='max_passing', lambda_cap=None):
    """Evidence contains full scored-frame masks, cameras, hand points AND normals.

    Coordinates must be the prediction world in metres. Every evidence artifact
    is hash-bound to its parent pair. Missing evidence cannot pass a candidate.
    """
    if lambda_policy not in ('max_passing', 'gcv_capped'): raise ValueError('Unknown lambda policy')
    if lambda_policy == 'gcv_capped' and lambda_cap is None: raise ValueError('GCV policy requires held-out surrogate evidence')
    configuration = {'lambdas': list(lambdas), 'method': 'recenter_huber_contact_pen_v2', 'lambda_policy': lambda_policy}
    previous = new_run(output, [bundle_path, mesh_path, evidence_path, *([lambda_cap] if lambda_cap else [])], configuration)
    if previous['state'] == 'CANDIDATE_REQUIRES_PACKED_VALIDATION':
        if (sha(Path(output)/Path(bundle_path).name) != previous['npz_sha256']
                or sha(Path(output)/Path(mesh_path).name) != previous['mesh_sha256']):
            raise ValueError('Completed candidate changed; preserve it for review')
        return previous
    b = dict(np.load(bundle_path, allow_pickle=False))
    with np.load(evidence_path, allow_pickle=False) as stream: e = dict(stream)
    if (str(e['parent_npz_sha256'].item()) != sha(bundle_path)
            or str(e['parent_mesh_sha256'].item()) != sha(mesh_path)
            or str(e['coordinate_frame'].item()) != 'prediction_world_meters'):
        raise ValueError('Evidence parent or frame convention mismatch')
    frames = np.asarray(e['frames'])
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    r, t, scale = b['object_rotation'], b['object_translation'], float(b['object_scale'])
    if r.shape[-2:] != (3, 3): raise ValueError('Candidate generation requires matrix rotations')
    neutrality = verify_recenter(v, f, r, t, scale)
    v, baseline, _ = recenter(v, f, r, t, scale)
    masks, hands, normals = e['masks'], e['hand_points'], e['hand_normals']
    human_masks = e.get('human_masks')
    if human_masks is None: human_masks = np.zeros_like(masks)
    if masks.ndim != 3 or masks.dtype.kind != 'b' or human_masks.shape != masks.shape or human_masks.dtype.kind != 'b':
        raise ValueError('Require matched boolean object/human evidence masks')
    if masks.shape[0] != len(frames) or hands.shape != normals.shape or hands.shape[0] != len(frames) or hands.shape[-1] != 3:
        raise ValueError('Incomplete per-scored-frame evidence')
    if not np.isfinite(hands).all() or not np.allclose(np.linalg.norm(normals, axis=-1), 1, atol=1e-3):
        raise ValueError('Missing or invalid full-hand normals')
    # Union, rather than replacing original points with their dilated counterparts.
    hands = np.concatenate([hands, hands+.002*normals], axis=1)
    def images(translation):
        return np.asarray([iou(silhouette(v, f, r[frame], translation[frame], scale,
                   e['intrinsics'][j] if e['intrinsics'].ndim == 3 else e['intrinsics'],
                   e['world_to_camera'][j], masks[j].shape) & ~human_masks[j],
                   masks[j] & ~human_masks[j]) for j, frame in enumerate(frames)])
    base_iou = images(baseline)
    _, _, base_gap, nearest_contact = penetration_constraints(v, f, r, baseline, scale, hands, frames, return_contacts=True)
    target, cw = contact_targets(nearest_contact[:, None], baseline, frames, base_gap <= .02)
    weights = np.ones(len(t)); weights[frames] = np.maximum(.05, base_iou)
    gcv = None
    if lambda_policy == 'gcv_capped':
        cap = load(lambda_cap)
        if cap.get('state') != 'HELD_OUT_SURROGATE_PASSED' or not cap.get('held_out_passed'):
            raise ValueError('Surrogate lambda cap lacks a passing held-out review')
        if 'fps' not in e or not np.isclose(float(e['fps']), cap['target_fps']):
            raise ValueError('Surrogate and evidence frame rates differ')
        for path, digest in cap['inputs_sha256'].items():
            if sha(path) != digest: raise ValueError('Surrogate evidence changed')
        gcv = gcv_lambda(baseline, frames, lambdas, weights)
        lambdas = [lam for lam in lambdas if lam <= min(2*gcv['lambda'], cap['lambda_cap'])]
        if not lambdas: raise ValueError('No lambda qualifies under the held-out cap')
    selected, selected_lambda, reviews = baseline.copy(), None, []
    for lam in sorted(set(lambdas)):
        candidate = smooth(baseline, frames, lam, weights, target, cw)
        try:
            active = {}
            for _ in range(10):
                constraints, depths, gap = penetration_constraints(v, f, r, candidate, scale, hands, frames)
                for frame, normal, bound in constraints:
                    active[(frame, tuple(np.round(normal, 6)), round(bound, 7))] = (frame, normal, bound)
                if not np.any(depths > 0): break
                # Historical cuts are conservative local choices. Infeasibility
                # rejects this lambda; no convergence/global-feasibility claim.
                candidate = smooth(baseline, frames, lam, weights, target, cw, list(active.values()))
            _, depths, gap = penetration_constraints(v, f, r, candidate, scale, hands, frames)
            if np.any(depths > 0):
                # Minimal per-frame rigid correction, followed by a re-fit with
                # repaired frames fixed. An infeasible anchor rejects the fit.
                constraints, _, _ = penetration_constraints(v, f, r, candidate, scale, hands, frames)
                fixed = []
                for j in np.flatnonzero(np.any(depths > 0, axis=1)):
                    frame = int(frames[j]); relevant = [(n, bound) for fr, n, bound in constraints if fr == frame]
                    matrix = sparse.csr_matrix(np.array([n for n, _ in relevant]))
                    moved = qp(sparse.eye(3), candidate[frame], matrix, np.full(len(relevant), -np.inf), np.array([b for _, b in relevant]))
                    for axis in np.eye(3):
                        fixed.extend([(frame, axis, float(axis@moved)), (frame, -axis, float(-axis@moved))])
                candidate = smooth(baseline, frames, lam, weights, target, cw, list(active.values())+fixed)
                _, depths, gap = penetration_constraints(v, f, r, candidate, scale, hands, frames)
            review = gate(baseline, candidate, frames, base_iou, images(candidate), base_gap, gap, depths)
            review['checks']['acceleration_nonincrease'] = acceleration(candidate, frames) <= acceleration(baseline, frames)+1e-10
            review['passed'] = all(review['checks'].values())
        except ValueError as exc:
            review = {'passed': False, 'error': str(exc)}
        review['lambda'] = lam; reviews.append(review)
        if review['passed']: selected, selected_lambda = candidate, lam
    result = dict(b); result['object_translation'] = selected.astype(t.dtype)
    if result['pose'][frames[0]].tobytes() != b['pose'][frames[0]].tobytes():
        raise ValueError('Alignment anchor changed')
    destination = Path(output)/Path(bundle_path).name
    np.savez_compressed(destination, **result)
    mesh_destination = Path(output)/Path(mesh_path).name
    trimesh.Trimesh(v, f, process=False).export(mesh_destination)
    # The exported GLB/NPZ are what the packer will read, not the float64 fit.
    serialized_mesh = trimesh.load(mesh_destination, force='mesh', process=False)
    v, f = np.asarray(serialized_mesh.vertices), np.asarray(serialized_mesh.faces)
    _, serialized_depth, serialized_gap = penetration_constraints(v, f, r, result['object_translation'], scale, hands, frames)
    stored_review = gate(baseline, result['object_translation'], frames, base_iou, images(result['object_translation']),
                        base_gap, serialized_gap, serialized_depth)
    stored_review['checks']['acceleration_nonincrease'] = acceleration(result['object_translation'], frames) <= acceleration(baseline, frames)+1e-10
    stored_review['passed'] = all(stored_review['checks'].values())
    if selected_lambda is not None and not stored_review['passed']:
        raise ValueError('Serialized candidate failed its evidence gates; preserve for review')
    removed = acceleration(baseline-selected, frames)
    review = {'selected_lambda': selected_lambda, 'lambda_policy': lambda_policy, 'gcv': gcv,
              'lambda_cap': load(lambda_cap)['lambda_cap'] if lambda_cap else None,
              'removed_acc_fraction': removed/max(acceleration(baseline, frames), 1e-12),
              'reviews': reviews, 'serialized_evidence_review': stored_review,
              'neutrality_before_serialization': neutrality,
              'parent_acceleration_cm': acceleration(t, frames), 'recentered_acceleration_cm': acceleration(baseline, frames),
              'candidate_acceleration_cm': acceleration(selected, frames), 'anchor_frame': int(frames[0]),
              'evaluated_frames': frames.tolist(),
              'npz_sha256': sha(destination), 'mesh_sha256': sha(mesh_destination),
              'scope': 'Observed-image and full supplied-hand proxy; host role subset and alignment scale are not known'}
    save(Path(output)/'review.json', review)
    complete(output, 'CANDIDATE_REQUIRES_PACKED_VALIDATION', **review)
    return review


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'mesh', 'evidence', 'output'): p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--lambdas', type=float, nargs='+', default=[3, 10, 30, 100, 300])
    p.add_argument('--lambda-policy', choices=['max_passing', 'gcv_capped'], default='max_passing', help='GCV remains opt-in until held-out validation')
    p.add_argument('--lambda-cap', type=Path)
    a = p.parse_args(); object_candidate(a.bundle, a.mesh, a.evidence, a.output, a.lambdas, lambda_policy=a.lambda_policy, lambda_cap=a.lambda_cap)


if __name__ == '__main__': main()


def component_candidate(mesh, episodes):
    """Mask-supported component ablation; never deletes geometry based on ACC alone.

    episodes entries carry bundle and hash-bound observation evidence arrays.
    An adopted variant must improve mean IoU by >=.05 in EVERY supplied episode.
    """
    components = sorted(mesh.split(only_watertight=False), key=lambda m: m.area, reverse=True)
    if len(components) == 1: return mesh.copy(), {'adopted': False, 'reason': 'single_component'}
    def per_frame(component, entry):
        b, e = entry['bundle'], entry['evidence']
        return [silhouette(component.vertices, component.faces, b['object_rotation'][frame], b['object_translation'][frame],
                           float(b['object_scale']), e['intrinsics'][j] if e['intrinsics'].ndim == 3 else e['intrinsics'],
                           e['world_to_camera'][j], e['masks'][j].shape) for j, frame in enumerate(e['frames'])]
    supported = [components[0]]
    for component in components[1:]:
        coverage = []
        for episode in episodes:
            for predicted, observed in zip(per_frame(component, episode), episode['evidence']['masks']):
                coverage.append(float(np.count_nonzero(predicted & observed)/max(1, np.count_nonzero(predicted))))
        if np.mean(coverage) >= .3: supported.append(component)
    candidates = {'full': mesh, 'largest': components[0], 'supported': trimesh.util.concatenate(supported)}
    scores = {name: [float(np.mean([iou(pred, obs) for pred, obs in zip(per_frame(candidate, episode), episode['evidence']['masks'])]))
                     for episode in episodes] for name, candidate in candidates.items()}
    passing = [name for name in ('largest', 'supported') if np.all(np.asarray(scores[name])-scores['full'] >= .05)]
    chosen = max(passing, key=lambda name: np.mean(scores[name])) if passing else 'full'
    return candidates[chosen].copy(), {'adopted': chosen != 'full', 'chosen': chosen, 'scores': scores,
                                      'scope': 'Observed full-frame silhouette gate, not hidden-reference CD-O'}
