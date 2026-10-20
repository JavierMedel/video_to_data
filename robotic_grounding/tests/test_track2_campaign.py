"""Submission invariants for the all-episode G1 baseline."""
import importlib.util
import json
from pathlib import Path
import sys

import pyarrow as pa
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
