"""Paths and artifact inventory for the isolated Track 1 improvement rollout."""
import json
from pathlib import Path

PILOTS = (6, 18, 28)
BATCHES = ((7, 8, 19, 20, 27, 29), tuple(range(6)) + tuple(range(9, 15)), (15, 16, 17) + tuple(range(21, 27)))
ORDER = tuple(i for batch in BATCHES for i in batch)


def paths(root, episode):
    root = Path(root)
    data = root / 'data/v2d_challenge/track_1'
    stem = f'episode_{episode:06d}'
    if episode < 6:
        baseline = data / 'cari4d_runs/track1_submission' / stem / stem
    else:
        baseline = data / 'cari4d_runs/track1_resume' / stem / stem
    if episode == 1:
        baseline = data / 'cari4d_runs/episode_000001_stability' / stem
    if episode == 15:
        baseline = data / 'cari4d_runs/episode_000015_test/2026-09-10_16-14-06_foam_grass_block_rotate_02'
    if episode == 16:
        baseline = data / stem
        export = next((baseline/'cari4d_export').glob('*/wild_export.json')).parent
        depth = baseline/'cari4d_aligned_depth.h5'
        init = baseline/'cari4d_sam3d_body.pt'
        intrinsics = baseline/'moge/cari4d_intrinsics.joblib'
    else:
        export = next((baseline/'export').glob('*/wild_export.json')).parent
        depth = baseline/'depth/moge2/aligned.h5'
        init = baseline/'mhr_init'/(stem+'.pkl')
        intrinsics = baseline/'depth/moge2/intrinsics.pkl'
        if episode == 15:
            init = baseline/'mhr_init'/(baseline.name+'.pkl')
    return {
        'root': root, 'data': data, 'stem': stem, 'baseline_run': baseline,
        'baseline_export': export, 'depth': depth, 'init': init,
        'intrinsics': intrinsics,
        'video': data/'videos/chunk-000/observation.images.exo_camera'/(stem+'.mp4'),
        'work': data/'improvement_rollout_v1'/stem,
        'report_dir': root/'incident_reports/track1_improvement/rollout',
        'baseline_pairs': data/'improvement_pilot_v1/candidate_episodes',
    }


def episode_metadata(root):
    path = Path(root)/'data/v2d_challenge/track_1/meta/episodes_metadata.jsonl'
    return {r['episode_index']: r for r in map(json.loads, path.read_text().splitlines())}
