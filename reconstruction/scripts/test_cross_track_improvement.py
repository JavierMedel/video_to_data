"""Cross-track numerical correctness, provenance and decision-boundary checks."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy import sparse
from scipy.spatial.transform import Rotation
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_track.common import new_run, sha, load, metric, KIT, REPO
from cross_track.temporal import smooth, qp, acceleration, smooth_rotations
from cross_track.track1 import gate, penetration_constraints, contact_targets
from cross_track.track3 import support_plane, bottleneck
from cross_track.policy import select, halving, training_job, diagnostics, checkpoint_metadata, read_rollout
from cross_track.submission import quota, snapshot, competitions
from track1_metric_harness import verify_recenter, recenter


def test_recenter_preserves_world_surface_for_rotating_scaled_offcenter_mesh():
    mesh = trimesh.creation.box(extents=[.1, .3, .2]); mesh.vertices += [.3, -.2, .1]
    r = Rotation.random(9, random_state=42).as_matrix(); t = np.random.default_rng(42).normal(size=(9, 3))
    review = verify_recenter(mesh.vertices, mesh.faces, r, t, 1.4)
    assert review['max_world_sample_error_m'] < 1e-12
    vertices, translation, _ = recenter(mesh.vertices, mesh.faces, r, t, 1.4)
    assert np.allclose(1.4*np.einsum('tij,vj->tvi', r, vertices)+translation[:, None],
                       1.4*np.einsum('tij,vj->tvi', r, mesh.vertices)+t[:, None])
    assert not np.allclose(t, translation)


def test_huber_fit_suppresses_jitter_without_bridging_unscored_gaps():
    y = np.c_[np.arange(30)*.01, np.sin(np.arange(30))* .01, np.zeros(30)]
    frames = np.r_[np.arange(10), np.arange(20, 30)]
    x = smooth(y, frames, 30)
    assert np.array_equal(x[10:20], y[10:20])
    assert np.array_equal(x[0], y[0])
    assert acceleration(x, frames) < acceleration(y, frames)


def test_constrained_qp_satisfies_anchor_and_halfspaces():
    result = qp(sparse.eye(2), np.array([3., -2.]), sparse.eye(2), [0., 1.], [1., 1.])
    assert np.allclose(result, [1, 1], atol=1e-7)
    with pytest.raises(ValueError): qp(sparse.eye(1), [0.], sparse.eye(1), [2.], [1.])


def test_contiguous_acceleration_matches_official_endpoint_weighting():
    x = np.random.default_rng(42).normal(size=(20, 2, 3))
    frames = np.r_[np.arange(7), np.arange(11, 20)]
    official = metric('track_1', 'ACC-O')
    expected = np.concatenate([official.acceleration_error_per_frame(x[r], np.zeros_like(x[r])) for r in (np.arange(7), np.arange(11, 20))]).mean()*100
    assert acceleration(x, frames) == pytest.approx(expected)


def test_jitter_sensitivity_uses_three_dimensional_norm_not_scalar_rms():
    sigma = .005; jitter = np.random.default_rng(42).normal(0, sigma, (20000, 3))
    # E||N(0,6 sigma^2 I3)|| = sqrt(6)*sigma*2*sqrt(2/pi).
    expected = np.sqrt(6)*sigma*2*np.sqrt(2/np.pi)*100
    assert acceleration(jitter, np.arange(len(jitter))) == pytest.approx(expected, rel=.02)


def test_so3_fit_handles_quaternion_sign_and_pi_branch_and_pins_anchor():
    angle = np.linspace(2.9, 3.4, 9)+np.array([0, .02, -.02, .03, -.02, .01, -.01, .02, 0])
    original = Rotation.from_rotvec(np.c_[np.zeros(9), np.zeros(9), angle]).as_matrix()
    fitted = smooth_rotations(original, np.arange(9), 3)
    assert np.array_equal(original[0], fitted[0])
    assert np.allclose(np.linalg.det(fitted), 1)
    def curvature(r):
        v = Rotation.from_matrix(r[1:]@r[:-1].transpose(0, 2, 1)).as_rotvec()
        return np.linalg.norm(np.diff(v, axis=0))
    assert curvature(fitted) < curvature(original)


def test_penetration_constraint_moves_object_away_from_hand():
    mesh = trimesh.creation.box(extents=[.2, .2, .2])
    r = np.eye(3)[None]; t = np.zeros((1, 3)); hands = np.array([[[.09, 0, 0]]])
    constraints, depth, gap = penetration_constraints(mesh.vertices, mesh.faces, r, t, 1., hands, [0])
    assert depth[0, 0] == pytest.approx(.01)
    _, normal, bound = constraints[0]
    assert normal[0] > .99 and bound < 0
    assert normal@np.array([-.02, 0, 0]) <= bound


def test_evidence_gates_reject_new_penetration_even_when_acceleration_improves():
    t = np.zeros((10, 3)); frames = np.arange(10)
    args = (t, t, frames, np.ones(10), np.ones(10), np.zeros(10), np.zeros(10))
    assert gate(*args, np.zeros((10, 3)))['passed']
    assert not gate(*args, np.full((10, 3), 1e-8))['passed']


def test_contact_targets_do_not_span_noncontiguous_scored_runs():
    t = np.zeros((6, 3)); t[4:] = [1, 0, 0]
    targets, w = contact_targets(np.zeros((4, 2, 3)), t, np.array([0, 1, 4, 5]), np.ones(4, bool))
    assert np.allclose(targets[:2], 0) and np.allclose(targets[4:], [1, 0, 0])


def test_version_directory_rejects_changed_inputs(tmp_path):
    source = tmp_path/'parent'; source.write_text('immutable')
    new_run(tmp_path/'version', [source], {'seed': 42})
    source.write_text('changed')
    with pytest.raises(ValueError): new_run(tmp_path/'version', [source], {'seed': 42})


def record(episode, sp=0, mp=0, auc=.1, error=10):
    return {'episode_index': episode, 'scope': 'exact_released_tier1', 'closeness': error,
            'metrics': {'spider_sr': sp, 'maniptrans_sr': mp, 'add_auc': auc, 'mppe_cm': error}}


def test_checkpoint_selection_obeys_success_priority_and_complete_roster():
    candidates = [record(i) for i in range(30)]+[record(2, sp=1, auc=.01), record(2, auc=.9)]
    selected = select(candidates)
    assert selected['episodes']['2']['metrics']['spider_sr'] == 1
    with pytest.raises(ValueError): select(candidates[:20])
    candidates[0]['scope'] = 'tier_internal_proxy'
    with pytest.raises(ValueError): select(candidates)


def test_halving_excludes_solved_high_auc_episodes():
    assert halving([record(0, sp=1, mp=1, auc=.8, error=0), record(1, error=.2), record(2, error=.1)], 1) == [2]


def test_native_job_plans_use_actual_nine_and_eleven_stage_curricula(tmp_path, monkeypatch):
    import shutil
    from cross_track import policy
    config = Path('robotic_grounding/flash_chord/src/flash_chord/configs')
    for name in ('experiment/g1_recon_body_flash_sac.yaml', 'experiment/sharpa_flash_sac.yaml',
                 'curriculum/recon_body_flash_sac.yaml', 'curriculum/tracking_250m_mixed_reset.yaml'):
        target = tmp_path/config/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO/config/name, target)
    monkeypatch.setattr(policy, 'REPO', tmp_path)
    reference = tmp_path/'robotic_grounding/data/reference'
    # Only the builder is used; it reads and hashes this tiny fixture, no native process.
    reference.parent.mkdir(parents=True, exist_ok=True); reference.write_text('reference')
    try:
        t2 = training_job('track_2_tier1', 1, reference, reference.parent/'job2', 20000768)
        t3 = training_job('track_3', 1, reference, reference.parent/'job3', 20000768)
        assert len(t2['curriculum_thresholds']) == 9
        assert len(t3['curriculum_thresholds']) == 11
        assert 'sim.fps=20' in t3['overrides']
        assert t2['curriculum_thresholds'][-1] == 20000768
        assert 'training.resume=true' not in t2['overrides']
    finally: reference.unlink()


def test_plane_fit_rejects_outliers_without_pca_object_gravity():
    rng = np.random.default_rng(42); points = rng.normal(size=(120, 3)); points[:100, 2] = .3+rng.normal(0, .001, 100)
    fit = support_plane(points)
    assert fit['inliers'] >= 100
    assert fit['normal'][2] > .999 and abs(fit['offset']+.3) < .002


def test_dev_decomposition_requires_matched_four_episode_arms():
    rows = [{'arm': arm, 'episode_index': i, 'metrics': {'spider_sr': 0, 'add_auc': .1}} for arm in 'OMRA' for i in range(4)]
    assert bottleneck(rows)['priority'] == 'policy_training'
    with pytest.raises(ValueError): bottleneck(rows[:-1])


def test_weekly_gate_counts_pending_and_fails_when_boundary_is_unknown():
    now = datetime(2026, 10, 20, tzinfo=timezone.utc)
    history = {'slug': 'board', 'observed_utc': now.isoformat(), 'history_complete': True,
               'submissions': [{'date': (now-timedelta(hours=i)).isoformat(), 'state': 'PENDING'} for i in range(5)]}
    with pytest.raises(ValueError, match='unresolved'): quota(history, {}, now)
    rules = {'weekly_window': 'rolling_7_days', 'weekly_window_source_quote': 'Fixture rule text',
             'weekly_window_source_url': 'https://example.org/rules', 'rules_captured_utc': now.isoformat()}
    with pytest.raises(ValueError, match='limit'): quota(history, rules, now)
    history['submissions'].pop()
    assert quota(history, rules, now)['used'] == 4


def test_read_only_snapshot_uses_all_23_configured_slugs(tmp_path):
    class Api:
        def competition_leaderboard_view(self, slug, page_size):
            assert page_size == 100
            return [{'score': '0.2'}]
        def competition_submissions(self, slug, page_number, page_size):
            assert page_number == 1 and page_size == 100
            return []
    assert len(competitions()) == 23
    result = snapshot(tmp_path/'snapshot', Api())
    assert len(result['competitions']) == 23
    assert len(list((tmp_path/'snapshot').glob('v2d*.json'))) == 23


def test_full_rollout_key_grid_rejects_duplicates(tmp_path):
    rows = []
    for frame in range(3):
        row = {'reference_frame': frame, 'environment_id': 0, 'object_body_id': 0, 'environment_valid': True, 'object_body_name': 'object'}
        for prefix in ('achieved', 'reference'):
            row.update({prefix+'_position_'+a: 0. for a in 'xyz'})
            row.update({prefix+'_quaternion_'+a: float(a == 'w') for a in 'xyzw'})
        rows.append(row)
    path = tmp_path/'trajectory.parquet'
    pq.write_table(pa.Table.from_pylist(rows), path)
    assert len(read_rollout(path)[0]) == 3
    pq.write_table(pa.Table.from_pylist(rows+[rows[-1]]), path)
    with pytest.raises(ValueError, match='Duplicate'): read_rollout(path)


def test_serialized_object_candidate_keeps_human_anchor_and_hash_bound_resume(tmp_path):
    from cross_track.track1 import object_candidate
    from cross_track.image import silhouette
    mesh_path = tmp_path/'episode_000000_object.glb'
    m = trimesh.creation.box(extents=[.2, .2, .2]); m.vertices += [.02, .01, 0]; m.export(mesh_path)
    m = trimesh.load(mesh_path, force='mesh', process=False)
    n = 5; r = np.tile(np.eye(3), (n, 1, 1)); t = np.tile([0., 0, 2.], (n, 1)); k = np.array([[80., 0, 32], [0, 80, 32], [0, 0, 1.]])
    bundle = tmp_path/'episode_000000.npz'
    pose = np.zeros((n, 136), dtype=np.float32)
    np.savez_compressed(bundle, pose=pose, scales=np.zeros(68), shape=np.zeros(45), object_rotation=r,
                        object_translation=t, object_scale=1.)
    masks = np.array([silhouette(m.vertices, m.faces, rotation, translation, 1., k, np.eye(4), (64, 64)) for rotation, translation in zip(r, t)])
    evidence = tmp_path/'evidence.npz'
    np.savez_compressed(evidence, parent_npz_sha256=sha(bundle), parent_mesh_sha256=sha(mesh_path),
                        coordinate_frame='prediction_world_meters', frames=np.arange(n), masks=masks, intrinsics=k,
                        world_to_camera=np.tile(np.eye(4), (n, 1, 1)), hand_points=np.tile([[[1., 1, 2.]]], (n, 1, 1)),
                        hand_normals=np.tile([[[1., 0, 0.]]], (n, 1, 1)))
    review = object_candidate(bundle, mesh_path, evidence, tmp_path/'version', lambdas=[3])
    assert review['selected_lambda'] == 3
    assert np.array_equal(np.load(tmp_path/'version'/bundle.name)['pose'], pose)
    object_candidate(bundle, mesh_path, evidence, tmp_path/'version', lambdas=[3])
    (tmp_path/'version'/bundle.name).write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'): object_candidate(bundle, mesh_path, evidence, tmp_path/'version', lambdas=[3])


def test_track3_upload_gate_includes_separate_mesh_board(tmp_path, monkeypatch):
    from cross_track import submission
    packet = tmp_path/'packet'; packet.write_bytes(b'packet')
    rules = tmp_path/'rules.json'; rules.write_text('{}')
    snapshot_dir = tmp_path/'snapshots'; snapshot_dir.mkdir()
    for track, _, slug, _ in competitions():
        if track in ('track_3', 'track_3_mesh'): (snapshot_dir/(slug+'.json')).write_text(json.dumps({'slug': slug}))
    monkeypatch.setattr(submission, 'quota', lambda history, rules: {'used': 0})
    result = submission.submit(packet, 'track_3', rules, snapshot_dir, tmp_path/'upload')
    assert len(result['checks']) == 6 and 'v2d-challenge-track3-cd-o' in result['checks']
    assert not (tmp_path/'upload/kaggle_receipt.json').exists()


def test_ambiguous_upload_is_blocked_before_any_retry(tmp_path):
    from cross_track.submission import submit
    packet = tmp_path/'packet'; packet.write_bytes(b'packet')
    rules = tmp_path/'rules.json'; rules.write_text('{}')
    report = tmp_path/'upload'; report.mkdir(); (report/'upload_manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='ambiguous'): submit(packet, 'track_3', rules, tmp_path, report, execute=True)


def test_tier3_preparation_needs_video_reconstruction_lineage(tmp_path):
    from cross_track.tier_inputs import validate_root
    (tmp_path/'reconstruction_manifest.json').write_text(json.dumps({'tier': 3, 'input_source': 'reconstruction', 'outputs_sha256': {}}))
    with pytest.raises(ValueError, match='lineage'): validate_root(tmp_path, 3)


def test_mesh_method_selection_uses_paired_dev_scores_and_fifteen_percent_gain():
    from cross_track.track3 import select_mesh_methods
    rows = [{'split': 'development', 'object_class': 'cup', 'episode_index': i, 'method': method, 'cd_o_cm': score}
            for i in (1, 2) for method, score in [('baseline', .5), ('clean', .4), ('mild', .48)]]
    assert select_mesh_methods(rows)['cup']['method'] == 'clean'
    rows[0]['split'] = 'evaluation'
    with pytest.raises(ValueError): select_mesh_methods(rows)


def test_observed_support_plane_alignment_is_shared_and_not_object_pca(tmp_path):
    import run_track3_full_campaign as campaign
    bundles = []
    for index in range(2):
        folder = tmp_path/f'object_{index}'; folder.mkdir(); bundles.append(folder)
        trimesh.creation.box(extents=[.1, .1, .2]).export(folder/'mesh.obj')
        pose = np.eye(4)[None]; pose[0, :3, 3] = [.2*index, .1, .4]
        np.savez_compressed(folder/'result.npz', object_to_world_transform=pose)
    x, y = np.meshgrid(np.linspace(-1, 1, 6), np.linspace(-1, 1, 6))
    points = np.c_[x.ravel(), y.ravel(), np.full(x.size, -.2)]
    path = tmp_path/'support.npz'
    np.savez_compressed(path, points=points, up=[0, 0, 1], coordinate_frame='reconstruction_world_meters')
    evidence = {'development_gate_passed': True, 'points_npz': str(path), 'points_sha256': sha(path)}
    world = campaign.shared_alignment(bundles, tmp_path/'alignment.json', evidence)
    assert np.allclose((points@world[:3, :3].T+world[:3, 3])[:, 2], 0, atol=1e-8)
    assert load(tmp_path/'alignment.json')['shared_by_all_objects']
    evidence['development_gate_passed'] = False
    with pytest.raises(ValueError, match='development'): campaign.shared_alignment(bundles, tmp_path/'bad.json', evidence)


def test_assembly_retains_all_parent_bytes_when_no_candidates_promoted(tmp_path):
    from cross_track.packaging import assemble_track1
    parent = tmp_path/'parent'; parent.mkdir()
    for i in range(30):
        # Retained pairs are copied verbatim; the official packer validates their
        # contents later, so this fixture does not need a fabricated full decode.
        (parent/f'episode_{i:06d}.npz').write_bytes(f'npz {i}'.encode())
        (parent/f'episode_{i:06d}_object.glb').write_bytes(f'mesh {i}'.encode())
    packet = tmp_path/'parent.parquet'; packet.write_bytes(b'parent packet')
    decisions = tmp_path/'decisions.json'; decisions.write_text(json.dumps({str(i): {} for i in range(30)}))
    scored = tmp_path/'scored.json'; scored.write_text(json.dumps({str(i): [0, 1, 2] for i in range(30)}))
    result = assemble_track1(parent, packet, decisions, scored, tmp_path/'version', minimum_promoted=0)
    assert result['promoted'] == [] and result['retained_parent'] == list(range(30))
    for path in parent.iterdir(): assert sha(path) == sha(tmp_path/'version/submission_episodes'/path.name)
