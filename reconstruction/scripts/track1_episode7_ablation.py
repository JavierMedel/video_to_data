"""Fixed 2x2 translation ablation from submitted v6; never packs or uploads.

Run in v2d_cari4d:blackwell with /repo mounted. --validate additionally runs
the existing all-scored-frame image/depth and actual-mesh hand checks.
Every branch keeps the v6 human arrays, rotations, scale and mesh unchanged.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

STEM = 'episode_000007'
BRANCHES = {
    'control': (False, 0.),
    'smoothing_only': (True, 0.),
    'depth_only': (False, .07),
    'combined': (True, .07),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def translations(source, smooth, offset, w2c):
    """Smooth world translations, then shift along each calibrated camera ray."""
    values = np.asarray(source, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError('Expected finite T x 3 translations')
    if smooth:
        from scipy.signal import savgol_filter
        if len(values) < 7:
            raise ValueError('Seven-frame smoothing requires at least seven frames')
        values = savgol_filter(values, 7, 2, axis=0, mode='interp')
    camera = values @ w2c[:3, :3].T + w2c[:3, 3]
    z = camera[:, 2].copy()
    if (z <= .1).any() or (z + offset <= .1).any():
        raise ValueError('Camera depth must remain above 0.1 m')
    if offset:
        camera *= ((z + offset) / z)[:, None]
        c2w = np.linalg.inv(w2c)
        values = camera @ c2w[:3, :3].T + c2w[:3, 3]
    return values.astype(source.dtype)


def verify_bundle(control, candidate):
    if set(control) != set(candidate):
        raise ValueError('Bundle keys changed')
    for key in control:
        if key != 'object_translation':
            np.testing.assert_array_equal(control[key], candidate[key], err_msg=key)
    value = candidate['object_translation']
    if value.shape != control['object_translation'].shape or value.dtype != control['object_translation'].dtype or not np.isfinite(value).all():
        raise ValueError('Invalid candidate translations')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/repo'))
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--mask-sheets', action='store_true', help='Render six source-frame mask comparisons for each changed branch')
    args = parser.parse_args()
    root = args.root.resolve()
    data = root / 'data/v2d_challenge/track_1'
    control_dir = data / 'cdo_v6_experiments/candidate_episodes'
    experiment = data / 'episode7_translation_ablation'
    reports = root / 'incident_reports/track1_improvement/episode7_translation_ablation'
    reports.mkdir(parents=True, exist_ok=True)
    control_path = control_dir / (STEM + '.npz')
    mesh_path = control_dir / (STEM + '_object.glb')
    v7_path = data / 'cdo_v7_experiments/candidate_episodes' / (STEM + '.npz')
    calibration_path = data / 'improvement_rollout_v1' / STEM / 'run/export' / STEM / 'edex'
    old_review_path = data / 'cdo_v7_experiments/hand_contact_candidates_v2' / STEM / 'review.json'
    control_hand_review = old_review_path.with_name('hand_penetration_review.json')
    inputs = [control_path, mesh_path, v7_path, calibration_path,
              root / 'incident_reports/track1_scored_frames.json', old_review_path,
              control_hand_review, v7_path.with_name(STEM + '_object.glb')]
    fingerprints = {str(p.relative_to(root)): digest(p) for p in inputs}
    control = dict(np.load(control_path, allow_pickle=False))
    v7 = dict(np.load(v7_path, allow_pickle=False))
    verify_bundle(control, v7)
    assert digest(v7_path.with_name(STEM + '_object.glb')) == digest(mesh_path)
    # This experiment is deliberately scoped to the identity-camera Episode 7.
    edex = json.loads(calibration_path.read_text())
    c2w = np.eye(4)
    c2w[:3, :4] = edex[0]['cameras'][0]['transform']
    np.testing.assert_allclose(c2w, np.eye(4), atol=1e-8)
    w2c = np.linalg.inv(c2w)
    sys.path.insert(0, str(root / 'modules/v2d_cari4d/lib/cari4d'))
    from prep.trajectory_diagnostics import rigid_motion_diagnostics
    import trimesh
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    vertices = mesh.vertices * float(control['object_scale'])
    old_review = json.loads(old_review_path.read_text())
    rows = {}
    for label, (smooth, offset) in BRANCHES.items():
        folder = experiment / label / STEM
        folder.mkdir(parents=True, exist_ok=True)
        candidate = dict(control)
        candidate['object_translation'] = translations(control['object_translation'], smooth, offset, w2c)
        verify_bundle(control, candidate)
        if label == 'combined':
            # V7 rounded between +.10m and -.03m stages. A direct +.07m
            # shift should reproduce it to within float32 rounding, not bits.
            residual = float(np.max(abs(candidate['object_translation'] - v7['object_translation'])))
            np.testing.assert_allclose(candidate['object_translation'], v7['object_translation'], atol=1e-6, rtol=0.)
        output = folder / (STEM + '.npz')
        if output.exists():
            saved = dict(np.load(output, allow_pickle=False))
            for key in candidate:
                np.testing.assert_array_equal(saved[key], candidate[key], err_msg=f'{label}/{key}')
        else:
            np.savez_compressed(output, **candidate)
        mesh_output = folder / mesh_path.name
        if not mesh_output.exists():
            shutil.copy2(mesh_path, mesh_output)
        assert digest(mesh_output) == digest(mesh_path)
        poses = np.tile(np.eye(4), (len(candidate['pose']), 1, 1))
        poses[:, :3, :3] = candidate['object_rotation']
        poses[:, :3, 3] = candidate['object_translation']
        motion = rigid_motion_diagnostics(poses, vertices)
        motion['translation_acceleration_cm_proxy'] = float(np.linalg.norm(np.diff(poses[:, :3, 3], n=2, axis=0), axis=1).mean() * 100)
        row = {'smoothing': smooth, 'window_frames': 7 if smooth else None,
               'polynomial_order': 2 if smooth else None, 'ray_depth_offset_m': offset,
               'motion': motion, 'candidate_npz_sha256': digest(output),
               'candidate_mesh_sha256': digest(mesh_output),
               'other_arrays_identical': True}
        if label == 'combined':
            row['max_translation_difference_from_v7_m'] = residual
        rows[label] = row
        # Supply the existing validator's motion/depth contract. The frame
        # split is inherited for comparison only; no candidate is fitted here.
        review = {'held_out_frames': old_review['held_out_frames'],
                  'geometry_review': {'held_out': {'depth_driven': bool(offset)}},
                  'summary': {'submitted_v6': {'sampled_vertex_acceleration_cm_proxy':
                              old_review['summary']['submitted_v6']['sampled_vertex_acceleration_cm_proxy']},
                              'final_candidate': motion},
                  'fixed_ablation': True, 'candidate_npz_sha256': digest(output)}
        (folder / 'review.json').write_text(json.dumps(review, indent=2) + '\n')
    manifest = {'episode': 7, 'control_version': 6, 'ground_truth_used': False,
                'inputs': fingerprints, 'branches': rows,
                'state': 'CANDIDATES_READY', 'official_scores': None,
                'scope': 'Fixed 2x2 ablation. Local diagnostics cannot establish official CD-O improvement. No packing or upload.'}
    manifest_path = reports / 'comparison.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    if args.validate:
        manifest['state'] = 'VALIDATING'
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        for label in BRANCHES:
            folder = experiment / label / STEM
            report = reports / label
            common = ['--root', str(root), '--episodes', '7', '--control-dir', str(control_dir),
                      '--candidate-dir', str(folder), '--report-dir', str(report)]
            for script, extra in [('track1_cdo_full_image_review.py', []),
                                  ('track1_cdo_hand_penetration_review.py', ['--iteration', '7', '--sample-count', '0',
                                   '--control-review', str(control_hand_review)])]:
                subprocess.run([sys.executable, str(root / 'scripts' / script), *common, *extra], check=True)
            rows[label]['image_depth_review'] = json.loads((folder / 'all_scored_image_review.json').read_text())
            rows[label]['hand_review'] = json.loads((folder / 'hand_penetration_review.json').read_text())
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        manifest['state'] = 'LOCAL_VALIDATION_COMPLETE'
    if args.mask_sheets:
        for label in BRANCHES:
            if label == 'control':
                continue
            folder = experiment / label / STEM
            subprocess.run([sys.executable, str(root / 'scripts/track1_cdo_render_candidate.py'),
                            '--root', str(root), '--iteration', '7', '--episode', '7', '--mask-sheet',
                            '--control-dir', str(control_dir), '--candidate-dir', str(folder),
                            '--candidate-label', label], check=True)
            rows[label]['mask_sheet_sha256'] = digest(folder / 'exact_candidate_comparison.jpg')
    assert all(digest(root / p) == value for p, value in fingerprints.items())
    manifest['source_hashes_preserved'] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    lines = ['# Episode 7 translation ablation', '',
             'Control: submitted v6. Only object translations vary. All diagnostics below are local proxies.', '',
             '| Branch | Ray shift (m) | Translation acceleration (cm) | Surface acceleration (cm) | Image IoU | Depth error (cm) | Hand / proximity |',
             '|---|---:|---:|---:|---:|---:|---|']
    for label, row in rows.items():
        image = row.get('image_depth_review'); hand = row.get('hand_review')
        iou = f"{image['summary']['candidate']['visible_iou_mean']:.6f}" if image else 'pending'
        depth = str(image['common_depth']['after_cm']) if image else 'pending'
        gates = f"{hand['hand_penetration_gate_pass']} / {hand['grasp_proximity_gate_pass']}" if hand else 'pending'
        lines.append(f"| {label} | {row['ray_depth_offset_m']:+.2f} | {row['motion']['translation_acceleration_cm_proxy']:.6f} | {row['motion']['sampled_vertex_acceleration_cm_proxy']:.6f} | {iou} | {depth} | {gates} |")
    lines += ['', f"State: {manifest['state']}. Source hashes preserved. Combined reproduces v7 within 1 micrometre.",
              '', 'No candidate is automatically promoted. Visual review and any official score remain separate from these local checks.',
              '', 'The unchanged control intentionally fails the meaningful-gain screen. Hand checks reuse the hash-matched v6 control summary; pointwise penetration deltas are unavailable with this cache. Depth comparisons use common pixels separately for each branch versus v6, so their support can differ.']
    (reports / 'results.md').write_text('\n'.join(lines) + '\n')
    print('EPISODE7_ABLATION', manifest['state'], str(manifest_path), flush=True)


if __name__ == '__main__':
    main()
