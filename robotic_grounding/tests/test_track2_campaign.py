"""Submission invariants for the all-episode G1 baseline."""
import importlib.util
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import run_track2_tier1_campaign as campaign


def test_scaled_curriculum_preserves_all_nine_stages_and_zero_assistance():
    values = campaign.scaled_thresholds(5000192)
    assert len(values) == 9 and values[-1] == 5000192
    assert all(v % 1024 == 0 for v in values)
    assert all(a < b for a, b in zip(values, values[1:]))
    assert abs(values[5] / 5000192 - .6) < .001
    assert campaign.scaled_thresholds(campaign.FULL_STEPS)[-1] == campaign.FULL_STEPS
    with pytest.raises(ValueError): campaign.scaled_thresholds(5000000)


def rollout(frames, valid=True):
    return pa.table({'environment_id': [0] * len(frames), 'reference_frame': frames,
                     'object_body_id': [0] * len(frames), 'environment_valid': [valid] * len(frames),
                     'achieved_position_x': [0.] * len(frames)})


def test_exact_coverage_rejects_missing_duplicated_invalid_worlds():
    campaign.validate_rollout(rollout([0, 1, 2]), {'frames': 3})
    for table in (rollout([0, 0, 2]), rollout([0, 1]), rollout([0, 1, 2], valid=False)):
        with pytest.raises(ValueError): campaign.validate_rollout(table, {'frames': 3})


def test_track2_native_launcher_explicitly_selects_isolated_mount_layout(tmp_path):
    command = campaign.native(tmp_path, 'scripts.train_flash_sac', ['scene.world_count=1024'])
    assert command[command.index('--track') + 1] == 'track_2_tier1'
    assert command[-1] == 'scene.world_count=1024'


def test_four_metric_receipt_is_complete_for_track2(tmp_path, monkeypatch):
    shared = campaign.common
    monkeypatch.setattr(shared, 'REPORT', tmp_path)
    packet = tmp_path / 'submission.csv'; packet.write_text('actual packet')
    calls, updates = [], []
    def run(command, log):
        calls.append(command)
        assert command[command.index('--track') + 1] == 'track_2_tier1'
        if '--dry-run' not in command:
            (tmp_path / 'kaggle_receipt.json').write_text(json.dumps({'submissions': {
                name: {'state': 'COMPLETE', 'ref': i + 1, 'public_score': '0.0'}
                for i, name in enumerate(['add_auc', 'spider_sr', 'maniptrans_sr', 'mppe_cm'])}}))
    monkeypatch.setattr(shared, 'run', run)
    shared.upload_packed(packet, lambda **kw: updates.append(kw), track='track_2_tier1')
    assert len(calls) == 2 and updates[-1]['state'] == 'SCORED'
    assert updates[-1]['uploaded'] is True
    assert 'G1+Dex3' in calls[-1][calls[-1].index('-m') + 1]


def test_training_quotes_hydra_reference_paths_with_equals(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, 'DATA', tmp_path / 'data')
    monkeypatch.setattr(campaign, 'REPORT', tmp_path / 'reports')
    monkeypatch.setattr(campaign, 'verify_input', lambda _: (tmp_path / 'reference.parquet', {'object': 'cone'}))
    monkeypatch.setattr(campaign, 'cpath', lambda _: '/repo/sequence_id=recording/data.parquet')
    def capture(command, log):
        assert 'sim.njmax=4096' in command and 'sim.nconmax=1024' in command
        override = next(v for v in command if v.startswith('task.parquet='))
        assert json.loads(override.split('=', 1)[1]) == '/repo/sequence_id=recording/data.parquet'
        raise RuntimeError('Captured before GPU execution')
    monkeypatch.setattr(campaign, 'run', capture)
    with pytest.raises(RuntimeError, match='Captured before GPU'):
        campaign.train_evaluate({'episode_index': 17})


@pytest.fixture
def pilot_campaign(tmp_path, monkeypatch):
    """Exercise real campaign ownership, checkpoint and rollout reuse without GPU jobs."""
    for name in ('STEPS', 'TIER', 'DATA', 'REPORT', 'POLICY_SUBDIR', 'RESULTS_SUBDIR',
                 'EXPORT_SUBDIR', 'LEGACY_OUTPUTS'):
        monkeypatch.setattr(campaign, name, getattr(campaign, name))
    for name in ('REPORT', 'PROFILE', 'TOTAL_STEPS'):
        monkeypatch.setattr(campaign.common, name, getattr(campaign.common, name))
    config = Path('flash_chord/src/flash_chord/configs/curriculum/recon_body_flash_sac.yaml')
    config_text = (campaign.ROOT / config).read_text()
    monkeypatch.setattr(campaign, 'ROOT', tmp_path)
    (tmp_path / config).parent.mkdir(parents=True)
    (tmp_path / config).write_text(config_text)
    inventory = tmp_path / 'inventory.json'
    inventory.write_text(json.dumps({'tier': 2, 'episodes': [
        {'episode_index': i, 'frames': 3 + i} for i in range(30)]}))
    data = tmp_path / 'data'
    references = tmp_path / 'references'
    references.mkdir()
    for i in range(30): (references / f'{i}.parquet').write_bytes(f'reference {i}'.encode())
    monkeypatch.setattr(campaign, 'verify_input', lambda episode: (
        references / f'{episode["episode_index"]}.parquet', {'object': 'cone'}))
    calls = {'training': [], 'evaluation': [], 'packing': []}

    def local(path):
        return tmp_path / Path(path).relative_to('/repo')

    def run(command, log):
        overrides = dict(item.split('=', 1) for item in command if '=' in item)
        if 'scripts.train_flash_sac' in command:
            output = local(overrides['output_dir'])
            i = int(output.name.split('_')[-1])
            calls['training'].append(i)
            output.mkdir(parents=True, exist_ok=True)
            (output / f'policy_{campaign.STEPS}.safetensors').write_bytes(f'test actor {i}'.encode())
        elif 'scripts.evaluate_policy' in command:
            checkpoint = local(overrides['evaluation.checkpoint'])
            i = int(checkpoint.parent.name.split('_')[-1])
            calls['evaluation'].append(i)
            table = rollout(list(range(3 + i))).append_column('object_body_name', pa.array(['object'] * (3 + i)))
            table = table.replace_schema_metadata({b'flash_chord.checkpoint_sha256': campaign.sha(checkpoint).encode()})
            trajectory = local(overrides['evaluation.object_trajectories_output'])
            trajectory.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, trajectory)
            local(overrides['evaluation.metrics_output']).write_text(json.dumps({
                'add_auc': .1, 'spider_sr': .2, 'maniptrans_sr': .3, 'mppe_cm': 4.5}))
        else:
            assert Path(command[1]).name == 'pack_submission.py'
            calls['packing'].append(command)
            Path(command[command.index('-o') + 1]).write_text('test packet; no real submission')
    monkeypatch.setattr(campaign, 'run', run)
    base = ['run_track2_tier1_campaign.py', '--tier', '2', '--version', 'v1',
            '--data-root', str(data), '--inventory', str(inventory)]
    report = tmp_path / 'incident_reports/track2_tier2/v1_campaign'
    return base, data, report, inventory, calls


def test_pilot_to_full_reuses_checkpoint_evaluation_and_preserves_report(pilot_campaign, monkeypatch):
    base, data, report, _, calls = pilot_campaign
    monkeypatch.setattr(sys, 'argv', [*base, '--episodes', '7', '3'])
    campaign.main()
    assert calls == {'training': [3, 7], 'evaluation': [3, 7], 'packing': []}
    assert not (data / 'submission_v1').exists()
    state = json.loads((report / 'execution.json').read_text())
    assert state['state'] == 'PILOT_EVALUATED_FOR_REVIEW'
    assert state['pilot_episodes'] == [3, 7]
    assert state['submission_ready'] is state['uploaded'] is False
    pilot_path = Path(state['pilot_report_path'])
    pilot_bytes = pilot_path.read_bytes()
    pilot = json.loads(pilot_bytes)
    assert pilot['elapsed_seconds'] >= 0
    assert len(pilot['episode_results']) == 2
    for result in pilot['episode_results']:
        assert result['metrics'] == {'add_auc': .1, 'spider_sr': .2, 'maniptrans_sr': .3, 'mppe_cm': 4.5}
        execution = result['evaluation']['execution']
        assert execution['training_performed'] and execution['evaluation_performed']
        assert not execution['evaluation_reused']
        assert execution['training_seconds'] >= 0 and execution['evaluation_seconds'] >= 0
    signature = (report / 'campaign_inputs.json').read_bytes()
    protected = [data / f'policies_v1/episode_{i:06d}/{filename}' for i in (3, 7)
                 for filename in ('.run_identity.json', 'policy_5000192.safetensors')]
    protected += [data / f'results_v1/episode_{i:06d}.{suffix}' for i in (3, 7) for suffix in ('parquet', 'json')]
    protected_bytes = {path: path.read_bytes() for path in protected}
    monkeypatch.setattr(sys, 'argv', base)
    campaign.main()
    assert sorted(calls['training']) == sorted(calls['evaluation']) == list(range(30))
    assert len(calls['packing']) == 1
    assert (report / 'campaign_inputs.json').read_bytes() == signature
    assert pilot_path.read_bytes() == pilot_bytes
    assert all(path.read_bytes() == content for path, content in protected_bytes.items())
    state = json.loads((report / 'execution.json').read_text())
    assert state['state'] == 'PACKED_FOR_REVIEW' and state['submission_ready']
    assert len(state['evaluated']) == 30 and 'pilot_episodes' not in state


def test_repeated_pilot_keeps_original_measurement_and_records_cache_hit(pilot_campaign, monkeypatch):
    base, _, report, _, calls = pilot_campaign
    monkeypatch.setattr(sys, 'argv', [*base, '--episodes', '7'])
    campaign.main()
    first = Path(json.loads((report / 'execution.json').read_text())['pilot_report_path'])
    original = first.read_bytes()
    campaign.main()
    second = Path(json.loads((report / 'execution.json').read_text())['pilot_report_path'])
    assert first != second and first.read_bytes() == original
    assert calls == {'training': [7], 'evaluation': [7], 'packing': []}
    execution = json.loads(second.read_text())['episode_results'][0]['evaluation']['execution']
    assert execution['evaluation_reused']
    assert not execution['training_performed'] and not execution['evaluation_performed']
    assert execution['training_seconds'] == execution['evaluation_seconds'] == 0


@pytest.mark.parametrize('extra', [
    ['--episodes', '-1'], ['--episodes', '30'], ['--episodes', '1', '1'],
    ['--episodes', '1', '--upload'],
])
def test_invalid_pilot_rejected_before_lock_or_report(pilot_campaign, monkeypatch, extra):
    base, data, report, _, calls = pilot_campaign
    monkeypatch.setattr(sys, 'argv', [*base, *extra])
    with pytest.raises(ValueError): campaign.main()
    assert not report.exists() and not data.exists()
    assert not any(calls.values())


def test_pilot_preserves_scored_legacy_baseline(pilot_campaign, monkeypatch, tmp_path):
    _, _, _, _, calls = pilot_campaign
    report = tmp_path / 'incident_reports/track2_baseline/short_campaign'
    report.mkdir(parents=True)
    baseline = report / 'execution.json'
    baseline.write_bytes(b'{"state":"SCORED","uploaded":true}\n')
    monkeypatch.setattr(sys, 'argv', ['campaign', '--episodes', '7'])
    with pytest.raises(ValueError, match='preserve the scored Tier 1 baseline'): campaign.main()
    assert baseline.read_bytes() == b'{"state":"SCORED","uploaded":true}\n'
    assert not (report / 'campaign.lock').exists()
    assert not any(calls.values())


@pytest.mark.parametrize('protection', ['alias', 'receipt'])
def test_pilot_cannot_alias_protected_or_submitted_outputs(pilot_campaign, monkeypatch, protection):
    base, data, report, _, calls = pilot_campaign
    if protection == 'alias':
        base[base.index('--version') + 1] = 'short'
    else:
        report.mkdir(parents=True)
        (report / 'kaggle_receipt.json').write_text('{}')
    monkeypatch.setattr(sys, 'argv', [*base, '--episodes', '7'])
    with pytest.raises(ValueError, match='Preserve baseline|submission lineage'): campaign.main()
    assert not data.exists() and not (report / 'campaign.lock').exists()
    assert not any(calls.values())


@pytest.mark.parametrize('bad_roster', ['missing', 'duplicate', 'replacement'])
def test_pilot_requires_complete_unique_inventory(pilot_campaign, monkeypatch, bad_roster):
    base, _, report, inventory, calls = pilot_campaign
    payload = json.loads(inventory.read_text())
    if bad_roster == 'missing': payload['episodes'].pop()
    elif bad_roster == 'duplicate': payload['episodes'].append(payload['episodes'][0])
    else: payload['episodes'][-1] = payload['episodes'][0]
    inventory.write_text(json.dumps(payload))
    monkeypatch.setattr(sys, 'argv', [*base, '--episodes', '7'])
    with pytest.raises(ValueError, match='exactly 30 unique episodes'): campaign.main()
    assert not report.exists() and not any(calls.values())


def test_pilot_failure_retains_report_and_never_packs(pilot_campaign, monkeypatch):
    base, _, report, _, calls = pilot_campaign
    def fail(_): raise RuntimeError('Test evaluator failure')
    monkeypatch.setattr(campaign, 'recoverable_train', fail)
    monkeypatch.setattr(sys, 'argv', [*base, '--episodes', '7'])
    with pytest.raises(RuntimeError, match='Test evaluator failure'): campaign.main()
    state = json.loads((report / 'execution.json').read_text())
    pilot = json.loads(Path(state['pilot_report_path']).read_text())
    assert state['state'] == pilot['state'] == 'STOPPED_FOR_REVIEW'
    assert pilot['failed'] == {'7': 'Test evaluator failure'}
    assert not pilot['submission_ready'] and not any(calls.values())
