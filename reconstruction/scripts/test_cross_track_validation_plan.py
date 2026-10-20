"""Regressions for real output ownership, paging, evidence and calibration."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_track.common import save, sha, load
from cross_track.ownership import own_output, validate_outputs, validate_owner
from cross_track.submission import paged_rows
from cross_track.temporal import gcv_lambda


def job_at(path, arm='fresh'):
    return {'track': 'track_2_tier2', 'episode_index': 1, 'arm': arm, 'output_dir': str(path),
            'target_steps': 5000192, 'reference_sha256': 'reference', 'parent_checkpoint_sha256': None,
            'reward_profile': 'baseline', 'overrides': ['training.total_environment_steps=5000192']}


def test_foreign_checkpoints_cannot_be_adopted_by_another_report(tmp_path):
    output = tmp_path/'new_arm'; output.mkdir(); (output/'state_5000192.safetensors').write_bytes(b'foreign')
    with pytest.raises(ValueError, match='identity'): validate_owner(job_at(output))
    with pytest.raises(ValueError, match='identity'):
        with own_output(job_at(output)): pytest.fail('foreign output adopted')


def test_output_owner_allows_interruption_but_rejects_arm_or_budget_changes(tmp_path):
    output = tmp_path/'new_arm'; job = job_at(output)
    with own_output(job): (output/'state_1024.safetensors').write_bytes(b'interrupted')
    with own_output(job): assert (output/'.run_identity.json').exists()
    for changed in ({**job, 'arm': 'warm-start'}, {**job, 'target_steps': 10000384}):
        with pytest.raises(ValueError, match='another run'): validate_owner(changed)


def test_output_lock_rejects_competing_controllers(tmp_path):
    job = job_at(tmp_path/'same')
    with own_output(job):
        with pytest.raises(BlockingIOError):
            with own_output(job): pytest.fail('parallel output ownership allowed')


def test_duplicate_nested_and_symlinked_baseline_outputs_are_rejected(tmp_path):
    with pytest.raises(ValueError, match='Duplicate'):
        validate_outputs([job_at(tmp_path/'job'), job_at(tmp_path/'job')])
    with pytest.raises(ValueError, match='overlapping'):
        validate_outputs([job_at(tmp_path/'job'), job_at(tmp_path/'job/child')])
    baseline = tmp_path/'baseline'; baseline.mkdir()
    link = tmp_path/'innocent'; link.symlink_to(baseline, target_is_directory=True)
    with pytest.raises(ValueError, match='baseline'): validate_owner(job_at(link/'episode'))


def test_explicit_resume_uses_a_new_directory_and_preserves_parent(tmp_path):
    parent = tmp_path/'baseline'; parent.mkdir(); (parent/'state').write_bytes(b'parent')
    job = {**job_at(tmp_path/'resumed', 'resume'), 'parent_checkpoint_sha256': sha(parent/'state')}
    with own_output(job): assert not list((tmp_path/'resumed').glob('state*'))
    assert (parent/'state').read_bytes() == b'parent'


def test_submission_ancestry_is_protected_even_with_innocent_names(tmp_path):
    packet = tmp_path/'scored'; packet.mkdir(); save(packet/'kaggle_receipt.json', {})
    with pytest.raises(ValueError, match='submission lineage'): validate_owner(job_at(packet/'new'))


def test_numbered_pagination_counts_more_than_one_full_page():
    class Api:
        def competition_submissions(self, slug, page_number=1, page_size=100):
            return [{'ref': str(v)} for v in range((page_number-1)*100, min(page_number*100, 205))]
    assert len(paged_rows(Api(), 'board')) == 205


def test_token_envelope_pagination_and_repeated_token_rejection():
    class Api:
        def competition_submissions(self, slug, page_token='', page_size=100):
            return {'submissions': [{'ref': 'first' if not page_token else 'second'}],
                    'next_page_token': 'next' if not page_token else ''}
    assert [r['ref'] for r in paged_rows(Api(), 'board')] == ['first', 'second']
    class Bad:
        def competition_submissions(self, slug, page_token='', page_size=100):
            return {'submissions': [{'ref': page_token or 'start'}], 'next_page_token': 'stuck'}
    with pytest.raises(ValueError, match='Repeated'): paged_rows(Bad(), 'board')


def test_discarded_tokens_and_duplicate_attempts_fail_closed():
    class Discarded:
        def competition_submissions(self, slug, page_token='', page_size=100): return []
    with pytest.raises(ValueError, match='discarded'): paged_rows(Discarded(), 'board')
    class Duplicate:
        def competition_submissions(self, slug, page_number=1, page_size=100): return [{'ref': 'same'}]*100
    with pytest.raises(ValueError, match='Duplicate'): paged_rows(Duplicate(), 'board')


def test_installed_sdk_adapter_keeps_response_tokens_without_authentication():
    from contextlib import contextmanager
    requests = []
    class Api:
        def competition_submissions(self, slug, page_token='', page_number=-1, page_size=20):
            pytest.fail('SDK high-level method drops the continuation token')
        @contextmanager
        def build_kaggle_client(self):
            def fetch(request):
                requests.append(request)
                token = request.page_token
                return SimpleNamespace(submissions=[{'ref': token or 'first'}], next_page_token='' if token else 'second')
            yield SimpleNamespace(competitions=SimpleNamespace(competition_api_client=SimpleNamespace(list_submissions=fetch)))
    assert len(paged_rows(Api(), 'board')) == 2
    assert requests[1].page_token == 'second'


def test_gcv_degrees_of_freedom_match_dense_anchored_reference():
    rng = np.random.default_rng(1); y = rng.normal(size=(12, 3)); w = np.linspace(.4, 1., 12); lam = 10.
    result = gcv_lambda(y, np.arange(12), [lam], w)['scores'][0]
    d = np.diff(np.eye(12), n=2, axis=0); h = np.diag(w)+lam*d.T@d
    expected = (1+np.trace(np.linalg.solve(h[1:, 1:], np.diag(w[1:]))))*3
    assert result['effective_df'] == pytest.approx(expected, rel=1e-12)
    with pytest.raises(ValueError): gcv_lambda(y, np.arange(12), [0.])


def test_gcv_is_independent_of_unscored_values():
    y = np.random.default_rng(42).normal(size=(20, 3)); frames = np.r_[0:7, 13:20]
    first = gcv_lambda(y, frames, [3, 10, 30]); y[7:13] += 1000
    assert gcv_lambda(y, frames, [3, 10, 30]) == first


def test_tier2_noise_diagnostic_distinguishes_synthetic_jitter_from_random_walk():
    from cross_track.tier2_noise import signal_statistics
    rng = np.random.default_rng(42); white = rng.normal(size=(5000, 3))
    assert signal_statistics(white, 30)['classification'] == 'white'
    assert signal_statistics(np.cumsum(white, axis=0), 30)['classification'] == 'random_walk'


def test_observed_rest_constraints_reduce_drift_and_keep_source_anchor():
    from cross_track.tier2_noise import constrained_motion
    n = 30; obj = np.repeat(np.eye(4)[None], n, axis=0); obj[:, 0, 3] = np.linspace(0, .015, n)
    root = np.zeros((n, 3)); evidence = {'stance': np.zeros((n, 2), bool), 'feet': np.zeros((n, 2, 3)),
          'wrists': np.zeros((n, 2, 3)), 'resting': np.ones(n, bool), 'contact': np.zeros((n, 2), bool)}
    result, changed_root, review = constrained_motion(obj, root, evidence, 10., .03)
    assert np.std(result[:, 0, 3]) < .5*np.std(obj[:, 0, 3])
    assert np.array_equal(result[0], obj[0]) and np.array_equal(changed_root[0], root[0])
    with pytest.raises(ValueError, match='budget'): constrained_motion(obj, root, evidence, 10., .0001)


def test_camera_calibration_recovers_a_known_camera_on_held_out_frames():
    pytest.importorskip('cv2')
    from cross_track.tier2_camera import project, calibrate
    rng = np.random.default_rng(42); points = rng.uniform(-.4, .4, (12, 12, 3)); points[..., 2] += 2
    params = np.r_[np.log([700., 720.]), [320, 240], [.1, -.05, .02], [.1, -.1, .5]]
    pixels, _ = project(points, params)
    result = calibrate(points, pixels, np.ones((12, 12), bool), np.arange(12), (480, 640))
    assert result['held_out_passed'] and result['held_out_median_error_px'] < 1e-5
    assert np.allclose(np.asarray(result['intrinsics'])[:2, :2].diagonal(), [700, 720], atol=.01)
    planar = points.copy(); planar[..., 2] = 2.
    with pytest.raises(ValueError, match='degenerate'): calibrate(planar, pixels, np.ones((12, 12), bool), np.arange(12), (480, 640))


def test_visible_iou_uses_human_occlusion_for_both_masks():
    from cross_track.track1_evidence import visible_iou
    obj = np.zeros((8, 8), bool); obj[2:6, 2:6] = True
    human = np.zeros_like(obj); human[:, :4] = True
    rendered = obj.copy(); rendered[2:6, 1] = True
    assert visible_iou(rendered, obj, human) == 1.


def test_surrogate_requires_episode_disjoint_splits(tmp_path):
    from cross_track.surrogate import study
    entries = [{'source_episode': 'same', 'split': split} for split in ('tuning', 'tuning', 'held_out', 'held_out')]
    manifest = tmp_path/'manifest.json'; save(manifest, {'purpose': 'track1_lambda_calibration', 'target_fps': 30, 'trajectories': entries})
    with pytest.raises(ValueError, match='overlap'): study(manifest, tmp_path/'study')


def test_gcv_candidate_requires_a_passing_held_out_cap(tmp_path):
    from cross_track.track1 import object_candidate
    with pytest.raises(ValueError, match='held-out'): object_candidate('unused', 'unused', 'unused', tmp_path, lambda_policy='gcv_capped')


def test_runner_resumes_only_its_own_interrupted_checkpoint(tmp_path, monkeypatch):
    from cross_track import jobs
    monkeypatch.setattr(jobs, 'REPO', tmp_path)
    mount = tmp_path/'robotic_grounding'; mount.mkdir()
    reference = mount/'reference'; reference.write_bytes(b'reference')
    output = mount/'data/new_arm/episode_000001'
    job = {**job_at(output), 'reference': str(reference), 'reference_sha256': sha(reference),
           'overrides': ['output_dir=/repo/data/new_arm/episode_000001', 'task.parquet=/repo/reference', 'training.total_environment_steps=5000192']}
    manifest = mount/'jobs.json'; save(manifest, [job]); report = mount/'reports/queue'
    calls = []
    def train(command, log):
        calls.append(command)
        if len(calls) == 1:
            save(output/'state_1024.safetensors', {'environment_steps': '1024'})
            raise RuntimeError('simulated interruption')
        assert 'training.resume=true' in command
        assert 'training.initial_policy_path=null' in command
        assert any('state_1024.safetensors' in str(v) for v in command)
        save(output/'state_5000192.safetensors', {'environment_steps': '5000192'})
        save(output/'policy_5000192.safetensors', {'environment_steps': '5000192'})
    monkeypatch.setattr(jobs, 'checked_command', train)
    monkeypatch.setattr(jobs, 'checkpoint_metadata', load)
    with pytest.raises(RuntimeError, match='interruption'): jobs.run_jobs(manifest, report, True)
    jobs.run_jobs(manifest, report, True)
    jobs.run_jobs(manifest, report, True)
    assert len(calls) == 2 and load(report/'job_0000/execution.json')['state'] == 'TRAINED'
    save(manifest, [{**job, 'arm': 'warm-start'}])
    with pytest.raises(ValueError, match='another run'): jobs.run_jobs(manifest, mount/'other_report', True)


def test_retarget_feasibility_checks_named_limits_and_support(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]/'robotic_grounding/scripts'))
    from retarget_track2_tier1 import verify_feasibility
    import pyarrow as pa
    import pyarrow.parquet as pq
    source = tmp_path/'source_episodes/demo'; source.mkdir(parents=True)
    save(source/'source_manifest.json', {'interpolated_object_frames': []})
    (source/'support.usda').write_text('#usda 1.0\n')
    parquet = tmp_path/'whole_body/challenge/sequence_id=demo/robot_name=g1/data.parquet'
    parquet.parent.mkdir(parents=True)
    stage = parquet.parents[3]/'reconstructed_stage/demo_support.usda'; stage.parent.mkdir()
    stage.write_bytes((source/'support.usda').read_bytes())
    urdf = tmp_path/'robot.urdf'; urdf.write_text('<robot><joint name="joint"><limit lower="-1" upper="1"/></joint></robot>')
    row = {'robot_joint_names': ['joint'], 'robot_joint_positions': [[0.], [.5]],
           'robot_root_position': [[0., 0, 0]]*2, 'robot_root_wxyz': [[1., 0, 0, 0]]*2}
    pq.write_table(pa.Table.from_pylist([row]), parquet)
    assert verify_feasibility(parquet, source, urdf)['joint_limits_checked'] == 1
    row['robot_joint_positions'][1][0] = 2.
    pq.write_table(pa.Table.from_pylist([row]), parquet)
    with pytest.raises(ValueError, match='limits violated'): verify_feasibility(parquet, source, urdf)


def test_mhr_parity_report_rejects_synthetic_or_stale_sources(tmp_path):
    from cross_track.mhr_asset import verify_report
    asset = tmp_path/'asset'; asset.write_bytes(b'participant asset')
    report = tmp_path/'report.json'; save(report, {'passed': True, 'frames': 100, 'episodes': 10,
         'asset_sha256': sha(asset), 'maximum_vertex_error_m': 1e-7, 'maximum_joint_error_m': 1e-7,
         'samples': []})
    with pytest.raises(ValueError, match='coverage'): verify_report(report, asset)
    save(report, {'passed': False})
    with pytest.raises(ValueError, match='three-way'): verify_report(report, asset)


def test_dev_jobs_verify_release_membership_and_preserve_version(tmp_path, monkeypatch):
    from cross_track import track3, policy
    monkeypatch.setattr(track3, 'REPO', tmp_path)
    reference = tmp_path/'reference'; reference.write_bytes(b'native fixture')
    def builder(track, episode, reference, output, steps):
        return {**job_at(output), 'track': track, 'episode_index': episode, 'target_steps': steps}
    monkeypatch.setattr(policy, 'training_job', builder)
    data = {'split': 'development', 'episodes': [{'episode_index': i, 'arms': {a: {'reference': str(reference), 'reference_sha256': sha(reference)} for a in 'OMRA'}} for i in range(4)]}
    paths = {}
    for name, folder, ids in [('development_metadata', 'public', range(4)), ('evaluation_metadata', 'evaluation', range(10, 14))]:
        path = tmp_path/f'reconstruction/data/v2d_challenge/track_3/{folder}/meta/episodes.jsonl'
        path.parent.mkdir(parents=True); path.write_text(''.join(json.dumps({'episode_index': i})+'\n' for i in ids))
        data[name] = {'path': str(path), 'sha256': sha(path)}; paths[name] = path
    manifest = tmp_path/'dev.json'; save(manifest, data)
    output = tmp_path/'dev_v1'
    assert len(track3.dev_jobs(manifest, output)) == 16
    data['new_method'] = 'changed'; save(manifest, data)
    with pytest.raises(ValueError, match='different'): track3.dev_jobs(manifest, output)
    paths['evaluation_metadata'].write_text(json.dumps({'episode_index': 3})+'\n')
    data['evaluation_metadata']['sha256'] = sha(paths['evaluation_metadata']); save(manifest, data)
    with pytest.raises(ValueError, match='overlap evaluation'): track3.dev_jobs(manifest, tmp_path/'dev_v2')


def test_noise_cohort_refuses_incomplete_or_cross_tier_inputs(tmp_path):
    from cross_track.tier2_noise import analyze_cohort
    with pytest.raises(ValueError, match='all 30'): analyze_cohort(tmp_path/'sources', tmp_path/'report')
    for i in range(30): save(tmp_path/f'sources/episode_{i}/source_manifest.json', {'episode_index': i, 'tier': 1 if i == 16 else 2})
    with pytest.raises(ValueError, match='own-tier'): analyze_cohort(tmp_path/'sources', tmp_path/'report')


def test_rolling_quota_also_requires_a_documented_official_boundary():
    from cross_track.submission import quota
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    history = {'history_complete': True, 'observed_utc': now.isoformat(), 'submissions': [], 'slug': 'board'}
    with pytest.raises(ValueError, match='official source'):
        quota(history, {'weekly_window': 'rolling_7_days'}, now)


def test_diagnostic_lambda_control_cannot_be_promoted_to_a_packet(tmp_path):
    from cross_track.packaging import assemble_track1
    parent = tmp_path/'parent'; parent.mkdir()
    for i in range(30):
        for suffix in ('.npz', '_object.glb'): (parent/f'episode_{i:06d}{suffix}').write_bytes(b'immutable parent')
    packet = tmp_path/'parent.parquet'; packet.write_bytes(b'parent packet')
    candidate = tmp_path/'candidate'; candidate.mkdir()
    save(candidate/'execution.json', {'state': 'CANDIDATE_REQUIRES_PACKED_VALIDATION',
         'evaluated_frames': [0, 1, 2], 'lambda_policy': 'max_passing'})
    decisions = {str(i): {} for i in range(30)}; decisions['0'] = {'object': str(candidate)}
    save(tmp_path/'reviews.json', decisions); save(tmp_path/'scored.json', {str(i): [0, 1, 2] for i in range(30)})
    with pytest.raises(ValueError, match='diagnostic control'):
        assemble_track1(parent, packet, tmp_path/'reviews.json', tmp_path/'scored.json', tmp_path/'assembled', minimum_promoted=0)


def test_host_reviews_resolve_full_repo_container_lineage_without_ambiguity(tmp_path, monkeypatch):
    from cross_track import common
    monkeypatch.setattr(common, 'REPO', tmp_path)
    source = tmp_path/'data/parent.npz'; source.parent.mkdir(); source.write_bytes(b'parent')
    digest = sha(source)
    record = {'inputs_sha256': {'/repo/data/parent.npz': digest}}
    assert common.sha('/repo/data/parent.npz') == digest
    assert common.input_digest(record, source) == digest
    record['inputs_sha256'][str(source)] = digest
    with pytest.raises(ValueError, match='unambiguous'): common.input_digest(record, source)
