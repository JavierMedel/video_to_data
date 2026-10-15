"""Fit and validate isolated contact-constrained smoothing from submitted v6.

Every video frame constrains the fit. Episode 7 defaults preserve its original
experiment; --episode and --campaign select isolated all-episode outputs.
Scored-frame checks use the same predictions, not held-out data or GT scores.
No mesh, rotation, human parameter, constant depth shift, packing or upload.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from track1_contact_smoothing import feasible_options, objective, select_safe_path, summarize_selected_contacts
from track1_episode7_ablation import digest, verify_bundle, translations


def write_json(path, value):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/repo'))
    p.add_argument('--validate', action='store_true')
    p.add_argument('--render', action='store_true', help='Render complete comparison videos and six source mask overlays')
    p.add_argument('--episode', type=int, choices=range(30), default=7)
    p.add_argument('--campaign', help='Isolated all-episode campaign name')
    p.add_argument('--fit-hand-review', action='store_true', help='Reuse exact all-frame finite-option contact evaluations for the selected path')
    p.add_argument('--mask-sheet', action='store_true', help='Render source mask comparisons without full videos')
    a = p.parse_args()
    root = a.root.resolve(); data = root / 'data/v2d_challenge/track_1'
    control_dir = data / 'cdo_v6_experiments/candidate_episodes'
    STEM = f'episode_{a.episode:06d}'
    assert a.campaign is None or (Path(a.campaign).name == a.campaign and a.campaign not in ('.', '..'))
    assert a.episode == 7 or a.campaign is not None
    out = data / (a.campaign or 'episode7_contact_smoothing') / STEM
    report = root / 'incident_reports/track1_improvement' / (a.campaign or 'episode7_contact_smoothing')
    if a.campaign is not None: report = report / STEM
    out.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    source = control_dir / (STEM + '.npz'); mesh_path = control_dir / (STEM + '_object.glb')
    control = dict(np.load(source, allow_pickle=False))
    if a.campaign is None:
        smooth_path = data / 'episode7_translation_ablation/smoothing_only' / STEM / source.name
    else:
        smooth_path = out / 'proposal.npz'
        proposal = dict(control)
        proposal['object_translation'] = translations(control['object_translation'], True, 0., np.eye(4))
        if smooth_path.exists():
            saved = dict(np.load(smooth_path, allow_pickle=False))
            for key in proposal: np.testing.assert_array_equal(saved[key], proposal[key])
        else:
            np.savez_compressed(smooth_path, **proposal)
    spec_path = root / 'modules/v2d_cari4d/lib/cari4d/lib_mhr/assets/mhr_hand_surface_spec.npz'
    model_path = root / 'data/weights/track1_submission_mhr/mhr_model.pt'
    calibration = data / ('improvement_pilot_v1' if a.episode in (6, 18, 28) else 'improvement_rollout_v1') / STEM / 'run/export' / STEM / 'edex'
    np.testing.assert_allclose(json.loads(calibration.read_text())[0]['cameras'][0]['transform'], np.eye(4)[:3], atol=1e-8)
    inputs = [source, mesh_path, smooth_path, spec_path, model_path, calibration,
              root / 'modules/v2d_cari4d/lib/cari4d/lib_mhr/mesh_penetration_review.py',
              root / 'v2d_submission_kit/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py',
              root / 'incident_reports/track1_scored_frames.json']
    evaluator = 'eager'
    if a.campaign is not None:
        evaluator_path = out / 'evaluator.json'
        if evaluator_path.exists():
            evaluator = json.loads(evaluator_path.read_text())['implementation']
        elif not (out / 'contact_costs.npz').exists():
            config = data / a.campaign / 'runtime_config.json'
            evaluator = json.loads(config.read_text())['implementation'] if config.exists() else 'eager'
            write_json(evaluator_path, {'implementation': evaluator})
        if evaluator_path.exists(): inputs.append(evaluator_path)
    assert evaluator in ('eager', 'fused')
    if evaluator == 'fused':
        inputs.append(root / 'modules/v2d_cari4d/lib/cari4d/lib_mhr/fused_mesh_penetration_review.py')
    fingerprints = {str(path.relative_to(root)): digest(path) for path in inputs}
    smooth = dict(np.load(smooth_path, allow_pickle=False)); verify_bundle(control, smooth)
    options = np.stack([control['object_translation'], smooth['object_translation']], axis=1)
    count = len(options)
    manifest_path = report / 'status.json'
    manifest = {'episode': a.episode, 'inputs': fingerprints, 'state': 'FITTING_CONTACT_CONSTRAINTS',
                'evaluator_implementation': evaluator,
                'frames': count, 'ground_truth_used': False, 'constant_ray_depth_offset_m': 0.,
                'max_translation_displacement_m': .03, 'fidelity_weight': .001,
                'scope': 'All-video-frame fit with exact-mesh full-hand/FPS512 mean and proximity constraints; local diagnostics, not official scores.'}
    write_json(manifest_path, manifest)
    sys.path.insert(0, '/workspace/v2d_sam3d_body/lib')
    sys.path.insert(0, str(root / 'modules/v2d_cari4d/lib/cari4d'))
    sys.path.insert(0, str(root / 'v2d_submission_kit/v2d_submission_kit/tools/track1'))
    import torch
    import trimesh
    from scipy.spatial import cKDTree
    from mesh_to_mhr_params import MHR
    from lib_mhr.hand_surface_contact import load_mhr_hand_surface_spec
    from lib_mhr.mesh_penetration_review import MeshPenetrationReview
    if evaluator == 'fused':
        os.environ.setdefault('LOGNAME', f'uid{os.getuid()}')
        os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
        os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(data / a.campaign / 'runtime_cache/inductor'))
        os.environ.setdefault('TRITON_CACHE_DIR', str(data / a.campaign / 'runtime_cache/triton'))
        from lib_mhr.fused_mesh_penetration_review import FusedMeshPenetrationReview
        MeshPenetrationReview = FusedMeshPenetrationReview
    from prep.trajectory_diagnostics import rigid_motion_diagnostics

    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    scale = float(control['object_scale'])
    vertices = mesh.vertices * scale
    spec = load_mhr_hand_surface_spec(spec_path)
    hand_ids = spec.vertex_indices.reshape(-1)
    fps_ids = (spec.sample_local_indices + np.arange(2)[:, None] * spec.vertex_indices.shape[1]).reshape(-1)
    cache = out / 'contact_costs.npz'
    metadata = out / 'contact_costs_inputs.json'
    full = np.zeros((count, 2)); fps = full.copy(); proximity = full.copy(); done = np.zeros((count, 2), dtype=bool)
    if cache.exists():
        assert json.loads(metadata.read_text()) == fingerprints, 'Contact cost inputs changed'
        saved = dict(np.load(cache, allow_pickle=False))
        full, fps, proximity, done = [saved[key] for key in ('full', 'fps', 'proximity', 'done')]
        assert all(x.shape == (count, 2) for x in (full, fps, proximity, done))
    else:
        write_json(metadata, fingerprints)
    if not done.all():
        model = MHR(str(model_path), 'cuda', 4, 'float32')
        check = MeshPenetrationReview(mesh.vertices, mesh.faces)
        tree = cKDTree(np.asarray(mesh.vertices))
        with torch.inference_mode():
            human, _ = model.run(torch.as_tensor(control['pose'], device='cuda', dtype=torch.float64),
                                 torch.as_tensor(np.r_[control['scales'], control['shape']][None], device='cuda', dtype=torch.float64))
            hands = (human[:, hand_ids] / 1000.).double()
            del model, human
            rotation = torch.as_tensor(control['object_rotation'], device='cuda', dtype=torch.float64)
            for state in range(2):
                for start in range(0, count, 8):
                    indices = np.arange(start, min(start+8, count))
                    indices = indices[~done[indices, state]]
                    if not len(indices):
                        continue
                    translation = torch.as_tensor(options[indices, state], device='cuda', dtype=torch.float64)
                    local = torch.einsum('tpi,tij->tpj', hands[indices] - translation[:, None], rotation[indices]) / scale
                    depths = check.depths(local).cpu().numpy() * scale * 100.
                    nearest = tree.query(local[:, fps_ids].cpu().numpy().reshape(-1, 3), workers=4)[0]
                    full[indices, state] = depths.mean(axis=1)
                    fps[indices, state] = depths[:, fps_ids].mean(axis=1)
                    proximity[indices, state] = nearest.reshape(len(indices), len(fps_ids)).min(axis=1) * scale * 100.
                    done[indices, state] = True
                    temporary = cache.with_name('contact_costs.tmp.npz')
                    np.savez_compressed(temporary, full=full, fps=fps, proximity=proximity, done=done)
                    os.replace(temporary, cache)
                    manifest['completed_frame_options'] = int(done.sum()); write_json(manifest_path, manifest)
                    if start % 32 == 0:
                        print('CONTACT_SMOOTH_COSTS', state, start, count, flush=True)
        del hands, rotation, check
        torch.cuda.empty_cache()
    assert done.all()
    allowed = feasible_options(options, full, fps, proximity)
    # Use the same sampled mesh points as the independently reported motion proxy.
    sampled = vertices[::max(1, int(np.ceil(len(vertices)/1024)))]
    path, value = select_safe_path(options, allowed, control['object_rotation'], sampled.mean(axis=0))
    baseline_objective = objective(options, control['object_rotation'], sampled.mean(axis=0), np.zeros(count, dtype=int))
    assert value <= baseline_objective + 1e-12
    selected = options[np.arange(count), path]
    assert np.all(allowed[np.arange(count), path])
    candidate = dict(control); candidate['object_translation'] = selected.copy(); verify_bundle(control, candidate)
    candidate_path = out / source.name
    if candidate_path.exists():
        saved = dict(np.load(candidate_path, allow_pickle=False))
        for key in candidate:
            np.testing.assert_array_equal(candidate[key], saved[key])
    else:
        np.savez_compressed(candidate_path, **candidate)
    if not (out / mesh_path.name).exists():
        shutil.copy2(mesh_path, out / mesh_path.name)
    assert digest(out / mesh_path.name) == digest(mesh_path)
    np.savez_compressed(out / 'selection.npz', path=path, allowed=allowed)
    motions = {}
    for label, values in [('submitted_v6', options[:, 0]), ('final_candidate', selected)]:
        poses = np.tile(np.eye(4), (count, 1, 1)); poses[:, :3, :3] = control['object_rotation']; poses[:, :3, 3] = values
        motions[label] = rigid_motion_diagnostics(poses, vertices)
        motions[label]['translation_acceleration_cm_proxy'] = float(np.linalg.norm(np.diff(values.astype(float), n=2, axis=0), axis=1).mean()*100.)
    write_json(out / 'review.json', {'held_out_frames': [], 'geometry_review': {'held_out': {'depth_driven': False}},
                                   'summary': motions, 'scope': 'Contact-constrained fit uses every video frame; no held-out claim.'})
    manifest.update(state='CANDIDATE_READY', selected_smoothed_frames=int(path.sum()),
                    feasible_smoothed_frames=int(allowed[:, 1].sum()), summary=motions,
                    objective_before=baseline_objective, objective_after=value,
                    max_actual_translation_displacement_m=float(np.linalg.norm(selected-options[:, 0], axis=1).max()),
                    mean_depth_change_m=float((selected[:, 2]-options[:, 0, 2]).mean()),
                    candidate_npz_sha256=digest(candidate_path), candidate_mesh_sha256=digest(out / mesh_path.name),
                    fit_full_hand_mean_before_cm=float(full[:, 0].mean()),
                    fit_full_hand_mean_after_cm=float(full[np.arange(count), path].mean()))
    write_json(manifest_path, manifest)
    if a.validate:
        manifest['state'] = 'VALIDATING'; write_json(manifest_path, manifest)
        common = ['--root', str(root), '--episodes', str(a.episode), '--control-dir', str(control_dir), '--candidate-dir', str(out), '--report-dir', str(report)]
        baseline_hand = data / 'cdo_v7_experiments/hand_contact_candidates_v2' / STEM / 'hand_penetration_review.json'
        checks = [('track1_cdo_full_image_review.py', [])]
        if a.fit_hand_review:
            frames = np.array(json.loads((root / 'incident_reports/track1_scored_frames.json').read_text())[str(a.episode)])
            hand = summarize_selected_contacts(full, fps, proximity, done, path, frames)
            hand.update(episode=a.episode, frames=frames.tolist(), all_scored_frames=True, sample_count=len(frames),
                        human_arrays_identical=True, hand_vertex_count=len(hand_ids), fps_sample_vertex_count=len(fps_ids),
                        candidate_npz_sha256=digest(candidate_path), candidate_mesh_sha256=digest(out / mesh_path.name),
                        control_npz_sha256=digest(source), control_mesh_sha256=digest(mesh_path),
                        max_pointwise_depth_increase_cm=None, pointwise_nonincrease=None,
                        source_contact_costs_sha256=digest(cache), selection_sha256=digest(out / 'selection.npz'),
                        scope='Exact precomputed actual-mesh full-hand/FPS512 evaluations for every frame-option pair, selected without changing geometry. Reused fitting evidence, not an independent hand rerender or official PEN.')
            write_json(out / 'hand_penetration_review.json', hand)
        else:
            extra = ['--iteration', '7', '--sample-count', '0']
            if baseline_hand.exists(): extra += ['--control-review', str(baseline_hand)]
            checks.append(('track1_cdo_hand_penetration_review.py', extra))
        for script, extra in checks:
            subprocess.run([sys.executable, str(root / 'scripts' / script), *common, *extra], check=True)
        image = json.loads((out / 'all_scored_image_review.json').read_text())
        hand = json.loads((out / 'hand_penetration_review.json').read_text())
        manifest.update(state='LOCAL_VALIDATION_COMPLETE', image_review=image, hand_review=hand,
                        local_gates_pass=bool(image['image_gate_pass'] and image['depth_gate_pass'] and hand['hand_penetration_gate_pass'] and hand['grasp_proximity_gate_pass']))
        write_json(manifest_path, manifest)
    if a.render or a.mask_sheet:
        render = [sys.executable, str(root / 'scripts/track1_cdo_render_candidate.py'), '--root', str(root), '--iteration', '7', '--episode', str(a.episode),
                  '--control-dir', str(control_dir), '--candidate-dir', str(out), '--candidate-label', 'Contact-constrained smoothing', '--output-dir', str(out / 'videos')]
        subprocess.run([*render, '--mask-sheet'], check=True)
        if a.render:
            if not (out / 'videos/validation.json').exists():
                subprocess.run(render, check=True)
            manifest['video_validation'] = json.loads((out / 'videos/validation.json').read_text())
        write_json(manifest_path, manifest)
    assert all(digest(root / path) == value for path, value in fingerprints.items())
    manifest['source_hashes_preserved'] = True; write_json(manifest_path, manifest)
    print('CONTACT_SMOOTH_COMPLETE', manifest['state'], json.dumps(motions), flush=True)


if __name__ == '__main__':
    main()
