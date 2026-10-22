"""Reproduce the final Tier 2 packet integrity audit without evaluating policies."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO / 'reconstruction/scripts'), str(REPO / 'robotic_grounding/scripts')]
import run_track2_tier1_campaign as campaign
from cross_track.ownership import validate_owner
from cross_track.policy import checkpoint_metadata


def main():
    campaign.TIER = 2
    campaign.DATA = REPO / 'robotic_grounding/data/track2/tier2/v1'
    campaign.STEPS = 5000192
    data = campaign.DATA
    reports = REPO / 'robotic_grounding/incident_reports/track2_tier2/v1_campaign'
    inventory = json.loads((reports.parent / 'inventory_v1.json').read_text())
    packet = data / 'submission_v1/track2_tier2.csv'
    frame = pd.read_csv(packet)
    episode_ids = frame.row_id.str.split('/').str[1].str.removeprefix('e').astype(int)
    assert set(episode_ids) == set(range(30)) and not frame.row_id.duplicated().any()
    rows = []
    for episode in inventory['episodes']:
        i = episode['episode_index']
        stem = f'episode_{i:06d}'
        reference, _ = campaign.verify_input(episode)
        receipt = json.loads((reports / stem / 'policy.json').read_text())
        checkpoint = Path(receipt['checkpoint_path'])
        metadata = checkpoint_metadata(checkpoint)
        config = json.loads(metadata['resolved_config_json'])
        assert receipt['state'] == 'EVALUATED'
        assert metadata['checkpoint_kind'] == 'policy'
        assert int(metadata['environment_steps']) == campaign.STEPS
        assert config['training']['total_environment_steps'] == campaign.STEPS
        assert config['scene']['world_count'] == 1024
        assert config['task']['parquet'] == campaign.cpath(reference)
        output = data / 'policies_v1' / stem
        job = {'track': 'track_2_tier2', 'episode_index': i, 'arm': 'fresh',
               'reference_sha256': campaign.sha(reference), 'output_dir': str(output),
               'target_steps': campaign.STEPS, 'reward_profile': 'baseline',
               'overrides': {'campaign_sha256': campaign.sha(Path(campaign.__file__)),
                             'thresholds': campaign.scaled_thresholds(campaign.STEPS)}}
        validate_owner(job)
        for key, path in [('trajectory_sha256', data / 'results_v1' / f'{stem}.parquet'),
                          ('metrics_sha256', data / 'results_v1' / f'{stem}.json'),
                          ('export_sha256', data / 'results_v1_submission' / f'{stem}.parquet'),
                          ('checkpoint_sha256', checkpoint), ('reference_sha256', reference)]:
            assert campaign.sha(path) == receipt[key], (i, key)
        table = pq.ParquetFile(data / 'results_v1' / f'{stem}.parquet').read()
        campaign.validate_rollout(table, episode)
        native = json.loads((data / 'results_v1' / f'{stem}.json').read_text())
        assert native['non_finite_worlds'] == []
        for phase in ['training', 'evaluation']:
            assert json.loads((reports / stem / phase / 'exit.json').read_text())['exit_code'] == 0
            assert not (reports / stem / phase / 'physics_buffer_failure.json').exists()
        part = frame[episode_ids == i]
        assert part.checkpoint_sha256.nunique() == 1
        assert part.checkpoint_sha256.iloc[0] == campaign.sha(checkpoint)
        peak = max(float(line.split(',')[2]) for line in
                   (reports / stem / 'training/gpu.csv').read_text().splitlines())
        rows.append({'episode_index': i, 'source_frames': episode['frames'],
                     'native_rows': len(table), 'packet_rows': len(part),
                     'checkpoint_sha256': campaign.sha(checkpoint),
                     'trained_steps': int(metadata['environment_steps']),
                     'curriculum_stage': metadata.get('curriculum_stage'),
                     'final_voc_scale': metadata.get('voc_scale'),
                     'training_peak_sampled_mib': peak, 'run_identity_verified': True})
    assert len(rows) == 30 and sum(row['packet_rows'] for row in rows) == len(frame)
    poses = frame[['pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w']].to_numpy()
    assert np.isfinite(poses).all()
    assert np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1, atol=2e-6)
    assert frame.code_commit_url.nunique() == 1
    report = {'state': 'PACKET_AND_ALL_30_POLICY_LINEAGES_VERIFIED',
              'packet': str(packet), 'packet_sha256': campaign.sha(packet),
              'rows': len(frame), 'native_frames': sum(row['native_rows'] for row in rows),
              'episode_count': len(rows), 'producing_commit_url': frame.code_commit_url.iloc[0],
              'peak_sampled_training_mib': max(row['training_peak_sampled_mib'] for row in rows),
              'episodes': sorted(rows, key=lambda row: row['episode_index']),
              'scope': 'Coverage, finite values, source/actor hashes, saved configs, native exits, and output ownership; no checkpoint selection'}
    Path(__file__).with_name('packet_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'episodes'}, indent=2))


if __name__ == '__main__':
    main()
