"""Episode-held-out Track 1 lambda calibration; never tunes Tier 2/3 policies."""
import argparse
from pathlib import Path
import numpy as np
from .common import load, save, sha, new_run, complete, metric
from .temporal import smooth, gcv_lambda, stretches


def acceleration_error(predicted, reference, frames):
    official = metric('track_1', 'ACC-O')
    return float(np.concatenate([official.acceleration_error_per_frame(predicted[r], reference[r])
                                 for r in stretches(frames, len(reference)) if len(r) >= 3]).mean())


def study(manifest_path, output):
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = load(manifest_path)
    if manifest.get('purpose') != 'track1_lambda_calibration': raise ValueError('This corpus cannot select Tier 2/3 methods')
    entries = manifest['trajectories']; fps = float(manifest['target_fps'])
    lambdas = sorted(set(float(v) for v in manifest.get('lambdas', [1, 3, 10, 30, 100, 300, 1000])))
    if fps <= 0 or not lambdas or min(lambdas) <= 0: raise ValueError('Invalid FPS/lambda grid')
    splits = {name: [e for e in entries if e['split'] == name] for name in ('tuning', 'held_out')}
    if any(len(group) < 2 for group in splits.values()): raise ValueError('Require >=2 source episodes in each split')
    ids = [e['source_episode'] for e in entries]
    if len(set(ids)) != len(ids): raise ValueError('Source episodes overlap or are duplicated across surrogate splits')
    inputs = [manifest_path]
    for entry in entries:
        if entry.get('dataset_split') not in ('public_development', 'released_track2_tier1'):
            raise ValueError('Surrogate requires a permitted public clean corpus')
        if sha(entry['path']) != entry['sha256']: raise ValueError('Surrogate corpus digest changed')
        inputs.append(entry['path'])
    noise_path = Path(manifest['noise_parent_bundle'])
    if sha(noise_path) != manifest['noise_parent_sha256']: raise ValueError('Noise estimate parent changed')
    inputs.append(noise_path)
    new_run(output, inputs, {'manifest_sha256': sha(manifest_path), 'method': 'episode_held_out_lambda_v1'})
    with np.load(noise_path, allow_pickle=False) as b: noisy_parent = np.asarray(b['object_translation'], float)
    residual = np.diff(noisy_parent, n=2, axis=0)
    sigma = max(1e-7, float(np.median(np.linalg.norm(residual, axis=1))/np.sqrt(6)))
    rows = []
    rng = np.random.default_rng(42)
    for entry in entries:
        with np.load(entry['path'], allow_pickle=False) as raw:
            clean = np.asarray(raw['positions'], float)
        source_fps = float(entry['fps'])
        if clean.ndim != 2 or clean.shape[1] != 3 or len(clean) < 5 or not np.isfinite(clean).all() or source_fps <= 0:
            raise ValueError('Require finite T by 3 clean motion with known fps')
        times = np.arange(len(clean))/source_fps
        grid = np.arange(0, times[-1]+1e-10, 1/fps)
        clean = np.stack([np.interp(grid, times, clean[:, axis]) for axis in range(3)], axis=1)
        frames = np.arange(len(clean))
        for kind in ('white', 'correlated', 'random_walk', 'zero'):
            for level in ([0.] if kind == 'zero' else [.5, 1., 2.]):
                noise = rng.normal(size=clean.shape)
                if kind == 'correlated':
                    for i in range(1, len(noise)): noise[i] += .8*noise[i-1]
                elif kind == 'random_walk': noise = np.cumsum(noise, axis=0)
                noise *= level*sigma/max(float(np.sqrt(np.mean(noise**2))), 1e-12)
                noise[0] = 0.; observed = clean+noise
                gcv = gcv_lambda(observed, frames, lambdas)['lambda']
                # Use the actual robust final temporal fit, not only linear GCV.
                errors = {str(lam): acceleration_error(smooth(observed, frames, lam), clean, frames) for lam in lambdas}
                rows.append({'episode': entry['source_episode'], 'split': entry['split'], 'noise': kind, 'level': level,
                             'gcv_lambda': gcv, 'unsmoothed_error_m': acceleration_error(observed, clean, frames),
                             'errors_m': errors, 'oracle_error_m': min(errors.values())})
    # Tune cap only on tuning episodes. Zero-noise controls are reported separately.
    tuning = [r for r in rows if r['split'] == 'tuning' and r['noise'] != 'zero']
    qualifying = []
    for cap in lambdas:
        ratios = [r['errors_m'][str(cap)]/max(r['oracle_error_m'], 1e-12) for r in tuning]
        if float(np.quantile(ratios, .9)) <= 1.05: qualifying.append(cap)
    cap = max(qualifying) if qualifying else None
    held = [r for r in rows if r['split'] == 'held_out' and r['noise'] != 'zero']
    ratios, regressions = [], []
    if cap is not None:
        for row in held:
            selected = max(v for v in lambdas if v <= min(cap, 2*row['gcv_lambda']))
            error = row['errors_m'][str(selected)]
            row.update(selected_lambda=selected, selected_error_m=error)
            ratios.append(error/max(row['oracle_error_m'], 1e-12))
            regressions.append(error > row['unsmoothed_error_m']+1e-12)
    passed = bool(ratios) and float(np.median(ratios)) <= 1.1 and float(np.mean(regressions)) <= .05
    record = {'lambda_cap': cap, 'held_out_passed': passed, 'target_fps': fps, 'estimated_noise_sigma_m': sigma,
              'held_out_median_oracle_ratio': float(np.median(ratios)) if ratios else None,
              'held_out_regression_fraction': float(np.mean(regressions)) if regressions else None,
              'rows': rows, 'scope': 'Episode-held-out temporal surrogate; constrained image/contact candidate still needs review'}
    state = 'HELD_OUT_SURROGATE_PASSED' if passed else 'SURROGATE_REJECTED_KEEP_BASELINE'
    complete(output, state, **record)
    # Cap file retains input and implementation fingerprints, not only a number.
    save(output/'track1_lambda_cap.json', load(output/'execution.json'))
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); study(a.manifest, a.output)
