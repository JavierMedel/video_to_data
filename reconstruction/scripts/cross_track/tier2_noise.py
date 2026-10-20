"""Tier-2-only motion/noise diagnostics. Spectral labels are hypotheses."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .common import load, save, sha, new_run, complete
from .temporal import smooth, smooth_rotations


def signal_statistics(values, fps, valid=None):
    values = np.asarray(values, float)
    if values.ndim != 2 or not np.isfinite(values).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Require finite T by D signal and positive fps')
    valid = np.ones(len(values), bool) if valid is None else np.asarray(valid, bool)
    if valid.shape != (len(values),): raise ValueError('Invalid visibility')
    ids = np.flatnonzero(valid)
    runs = np.split(ids, np.flatnonzero(np.diff(ids) != 1)+1)
    velocities, previous, following, accelerations, spectra = [], [], [], [], []
    grid = np.linspace(0, fps/2, 65)
    for run in runs:
        if len(run) < 4: continue
        x = values[run]; delta = np.diff(x, axis=0)
        velocities.append(delta); previous.append(delta[:-1]); following.append(delta[1:]); accelerations.append(np.diff(delta, axis=0))
        centered = x-np.linspace(x[0], x[-1], len(x))
        power = np.abs(np.fft.rfft(centered, axis=0))**2/len(x)
        frequency = np.fft.rfftfreq(len(x), 1/fps)
        spectra.append(np.stack([np.interp(grid, frequency, power[:, axis]) for axis in range(x.shape[1])], axis=1))
    if not velocities: return {'classification': 'unclear', 'usable_runs': 0, 'scope': 'No contiguous visible observations'}
    a, b = np.concatenate(previous), np.concatenate(following)
    a, b = a-a.mean(0), b-b.mean(0)
    denominator = float(np.sqrt(np.sum(a*a)*np.sum(b*b)))
    correlation = float(np.sum(a*b)/denominator) if denominator > 1e-14 else None
    label = 'unclear' if correlation is None else ('white' if correlation < -.3 else 'random_walk' if abs(correlation) < .15 else 'mixed')
    norms = np.linalg.norm(np.concatenate(accelerations), axis=1)
    return {'classification': label, 'increment_lag1_correlation': correlation, 'usable_runs': len(velocities),
            'second_difference_norm_quantiles': np.quantile(norms, [.1, .5, .9, .99]).tolist(),
            'frequency_hz': grid.tolist(), 'detrended_power_by_axis': np.mean(spectra, axis=0).tolist(),
            'scope': 'Noise hypothesis confounded by true motion; no clean Tier 1 data used'}


def verified_evidence(path, source, count):
    with np.load(path, allow_pickle=False) as raw: e = dict(raw)
    if str(e['source_manifest_sha256'].item()) != sha(Path(source)/'source_manifest.json'):
        raise ValueError('Kinematic evidence belongs to another source')
    if str(e['coordinate_frame'].item()) != 'released_track2_opencv_world': raise ValueError('Kinematic evidence frame mismatch')
    if not bool(e['observation_verified'].item()): raise ValueError('Rest/contact labels require observation verification')
    for key, shape in [('wrists', (count, 2, 3)), ('feet', (count, 2, 3)), ('resting', (count,)), ('stance', (count, 2)), ('contact', (count, 2))]:
        if e[key].shape != shape or not np.isfinite(e[key]).all(): raise ValueError('Invalid kinematic evidence: '+key)
    if any(e[k].dtype.kind != 'b' for k in ('resting', 'stance', 'contact')): raise ValueError('Require boolean observed labels')
    return e


def analyze(source, output, evidence=None):
    source, output = Path(source), Path(output)
    manifest = load(source/'source_manifest.json')
    if manifest.get('tier') != 2: raise ValueError('Noise report consumes only tier-owned Tier 2 inputs')
    for name, digest in manifest['output_sha256'].items():
        if sha(source/name) != digest: raise ValueError('Tier 2 source changed')
    new_run(output, [source/'source_manifest.json', *[source/n for n in manifest['output_sha256']], *([evidence] if evidence else [])], {'method': 'tier2_observation_noise_v1'})
    objects = np.load(source/'poses.npy', allow_pickle=False)
    with np.load(source/'soma_params.npz', allow_pickle=False) as b: root = b['transl']
    valid = np.ones(len(objects), bool); valid[manifest['interpolated_object_frames']] = False
    result = {'episode_index': manifest['episode_index'], 'tier': 2,
              'position': signal_statistics(objects[:, :3, 3], manifest['fps'], valid),
              'human_root': signal_statistics(root, manifest['fps']), 'resting': None}
    # Log increments avoid quaternion sign ambiguity; exclude visibility gaps.
    velocity = Rotation.from_matrix(objects[1:, :3, :3]@objects[:-1, :3, :3].transpose(0, 2, 1)).as_rotvec()
    result['angular_velocity'] = signal_statistics(velocity, manifest['fps'], valid[1:] & valid[:-1])
    if evidence:
        e = verified_evidence(evidence, source, len(objects))
        plane = np.asarray(load(source/'ground_plane.json')['plane'], float)
        if plane.shape != (4,) or not np.isfinite(plane).all() or not np.isclose(np.linalg.norm(plane[:3]), 1, atol=1e-5):
            raise ValueError('Invalid released ground plane')
        height = objects[:, :3, 3]@plane[:3]+plane[3]
        foot_height = e['feet']@plane[:3]+plane[3]
        result['resting'] = {'observed_frames': int(e['resting'].sum()),
             'object_height_std_m': float(np.std(height[e['resting']])) if e['resting'].any() else None,
             'stance_foot_height_std_m': float(np.std(foot_height[e['stance']])) if e['stance'].any() else None,
             'scope': 'Observed labels and own-tier FK; moving without hands is not assumed stationary'}
    save(output/'noise_report.json', result); complete(output, 'NOISE_DIAGNOSTIC_NOT_GROUND_TRUTH', **result)
    return result


def windows(mask):
    ids = np.flatnonzero(mask)
    return [] if not len(ids) else np.split(ids, np.flatnonzero(np.diff(ids) != 1)+1)


def constrained_motion(objects, human_root, evidence, lam, max_correction):
    """Observed rest/stance/contact targets; new outputs still need native retarget."""
    if not np.isfinite(max_correction) or max_correction <= 0: raise ValueError('Require an explicit positive correction budget')
    objects = np.asarray(objects, float); root = np.asarray(human_root, float); frames = np.arange(len(root))
    targets = root.copy(); weights = np.zeros(len(root))
    for foot in range(2):
        for run in windows(evidence['stance'][:, foot]):
            if len(run) < 3: continue
            delta = np.median(evidence['feet'][run, foot], axis=0)-evidence['feet'][run, foot]
            proposed = root[run]+delta
            targets[run] = (targets[run]*weights[run, None]+proposed)/(weights[run, None]+1)
            weights[run] += 1
    new_root = smooth(root, frames, lam, contact_target=targets, contact_weights=weights*10)
    position = objects[:, :3, 3]; target = position.copy(); w = np.zeros(len(root)); rotation = objects[:, :3, :3].copy()
    for run in windows(evidence['resting']):
        if len(run) < 3: continue
        target[run] = np.median(position[run], axis=0); w[run] = 10
        rotation[run] = Rotation.from_matrix(rotation[run]).mean().as_matrix()
    hands = evidence['wrists']+(new_root-root)[:, None]
    for hand in range(2):
        for run in windows(evidence['contact'][:, hand]):
            if len(run) < 3: continue
            offset = np.median(position[run]-hands[run, hand], axis=0)
            proposed = hands[run, hand]+offset
            target[run] = (target[run]*w[run, None]+10*proposed)/(w[run, None]+10); w[run] += 10
    result = objects.copy()
    result[:, :3, 3] = smooth(position, frames, lam, contact_target=target, contact_weights=w)
    result[:, :3, :3] = smooth_rotations(rotation, frames, lam)
    # Preserve the full source anchor after any constant-rotation targets.
    result[0] = objects[0]; new_root[0] = root[0]
    peak = max(float(np.linalg.norm(new_root-root, axis=1).max()), float(np.linalg.norm(result[:, :3, 3]-position, axis=1).max()))
    angular = Rotation.from_matrix(result[:, :3, :3]@objects[:, :3, :3].transpose(0, 2, 1)).magnitude()
    if peak > max_correction or np.max(angular) > .2:
        raise ValueError('Observed-constraint correction exceeded position/rotation budget')
    return result, new_root, {'maximum_position_correction_m': peak, 'maximum_rotation_correction_rad': float(np.max(angular))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--evidence', type=Path)
    p.add_argument('--cohort', action='store_true', help='Complete 30-episode Tier 2 source inventory and pooled hypotheses')
    a = p.parse_args()
    if not a.cohort: analyze(a.source, a.output, a.evidence)
    else:
        if a.evidence: raise ValueError('Per-episode observation evidence must be reviewed separately')
        analyze_cohort(a.source, a.output)


def analyze_cohort(source_root, output):
    paths = sorted(Path(source_root).glob('*/source_manifest.json')); output = Path(output)
    records = [load(p) for p in paths]
    if len(records) != 30 or {r['episode_index'] for r in records} != set(range(30)) or any(r.get('tier') != 2 for r in records):
        raise ValueError('Require all 30 distinct own-tier Tier 2 sources')
    new_run(output, paths, {'method': 'complete_tier2_noise_hypotheses_v1'})
    rows = [analyze(path.parent, output/f'episode_{record["episode_index"]:06d}') for path, record in zip(paths, records)]
    counts = {key: sum(row['position']['classification'] == key for row in rows) for key in ('white', 'random_walk', 'mixed', 'unclear')}
    result = {'episodes': 30, 'position_classification_counts': counts, 'rest_evidence_available': False,
              'scope': 'Pooled observational noise hypotheses; motion and drift are not identifiable from spectra alone'}
    save(output/'cohort_summary.json', result); complete(output, 'TIER2_NOISE_COHORT_DIAGNOSTIC', **result)
    return result
