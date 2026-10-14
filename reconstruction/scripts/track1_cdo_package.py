"""Select evidence-backed object changes and build one immutable Track 1 package."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/v2d_challenge/track_1'
REPORT = ROOT / 'incident_reports/track1_improvement/cdo_v4'
KIT = ROOT / 'v2d_submission_kit/v2d_submission_kit'
p = argparse.ArgumentParser()
p.add_argument('--commit', help='Git commit URL or hash; required when packing')
p.add_argument('--review-only', action='store_true')
a = p.parse_args()
sys.path.insert(0, str(KIT))
from v2dlb.mhr_submission import EPISODE_FRAME, _rotations
from submit.v2d_submit import validate_submission


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


decisions = {}
for episode in (1, 2, 12, 14, 19, 20):
    stem = f'episode_{episode:06d}'
    selected = None
    for group in ('pose_candidates', 'hoop_candidates', 'hoop_geometry', 'physics_hoop', 'physics_hoop_v2', 'contact_projection'):
        candidate = DATA / 'cdo_v4_experiments' / group / stem
        if (candidate / 'review.json').exists():
            selected = candidate
    assert selected is not None
    review = json.loads((selected / 'review.json').read_text())
    contact = json.loads((selected / 'contact_review.json').read_text())
    assert contact['candidate_npz_sha256'] == sha(selected / (stem + '.npz'))
    assert contact['candidate_mesh_sha256'] == sha(selected / (stem + '_object.glb'))
    old, new = review['summary']['submitted_v3'], review['summary']['final_candidate']
    gates = {'silhouette': new['visible_iou_mean'] >= max(.3, old['visible_iou_mean']-.02),
             'translation_motion': new['translation_acceleration_cm_proxy'] <= max(.1, old['translation_acceleration_cm_proxy']*1.1),
             'surface_motion': new['sampled_vertex_acceleration_cm_proxy'] <= max(.1, old['sampled_vertex_acceleration_cm_proxy']*1.1),
             'signed_penetration': contact['contact_gate_pass'], 'human_arrays_unchanged': contact['human_parameters_identical']}
    passed = all(gates.values())
    decisions[str(episode)] = {'verdict': 'PROMOTE' if passed else 'RETAIN_V3', 'gates': gates,
                               'source': str(selected), 'summary': {'submitted_v3': old, 'candidate': new},
                               'contact_summary': contact['summary'], 'visual_review_complete': False,
                               'review_sha256': sha(selected/'review.json'), 'contact_sha256': sha(selected/'contact_review.json')}
out = REPORT / 'promotion_decisions.json'
if a.review_only:
    assert not out.exists(), 'Preserve recorded review decisions'
    out.write_text(json.dumps(decisions, indent=2) + '\n')
    print(json.dumps(decisions, indent=2))
    raise SystemExit(0)
assert a.commit, '--commit is required to package'
approved = json.loads(out.read_text())
for episode, decision in decisions.items():
    recorded = approved[episode]
    assert all(recorded[k] == decision[k] for k in ('verdict', 'gates', 'source', 'review_sha256', 'contact_sha256'))
    assert recorded['visual_review_complete'] and recorded.get('reason'), episode
promoted = [int(i) for i, d in approved.items() if d['verdict'] == 'PROMOTE']
assert promoted
episodes = DATA / 'cdo_v4_experiments/candidate_episodes'
archive = DATA / 'submission_packages/track1_cdo_v4.parquet'
assert not episodes.exists() and not archive.exists(), 'Preserve existing package'
episodes.mkdir(parents=True)
v3 = DATA / 'improvement_rollout_v1/candidate_episodes'
v3_hashes = json.loads((ROOT / 'incident_reports/track1_improvement/rollout/candidate_validation.json').read_text())['artifact_sha256']
sample = pd.read_parquet(KIT / 'data/track_1_sample_submission.parquet', columns=['row_id'])
parts = sample.row_id.str.split('_', expand=True)
hashes = {}; changed = []
for episode in range(30):
    stem = f'episode_{episode:06d}'
    source = Path(approved[str(episode)]['source']) if episode in promoted else v3
    for suffix in ('.npz', '_object.glb'):
        assert sha(v3/(stem+suffix)) == v3_hashes[stem+suffix], 'Submitted v3 artifact changed'
        original = source / (stem+suffix); destination = episodes / original.name
        shutil.copy2(original, destination)
        hashes[destination.name] = sha(destination)
        assert hashes[destination.name] == sha(original)
    with np.load(episodes / (stem+'.npz'), allow_pickle=False) as d, np.load(v3/(stem+'.npz'), allow_pickle=False) as prior:
        count = len(d['pose'])
        for key, shape in {'pose': (count, 136), 'scales': (68,), 'shape': (45,), 'object_rotation': (count, 3, 3), 'object_translation': (count, 3)}.items():
            assert d[key].shape == shape and np.isfinite(d[key]).all(), (episode, key)
        for key in ('pose', 'scales', 'shape'):
            assert np.array_equal(d[key], prior[key]), (episode, key)
        assert float(d['object_scale'].reshape(())) > 0 and np.isfinite(d['object_scale']).all()
        _rotations(d['object_rotation'])
        selected_frames = parts.loc[(parts[1].astype(int)==episode) & (parts[2].astype(int)!=EPISODE_FRAME), 2].astype(int)
        assert count > selected_frames.max()
        if not np.array_equal(d['object_translation'], prior['object_translation']) or not np.array_equal(d['object_rotation'], prior['object_rotation']):
            changed.append(episode)
    mesh = trimesh.load(episodes/(stem+'_object.glb'), force='mesh', process=False)
    assert len(mesh.faces) and len(mesh.vertices) and mesh.area > 0 and np.isfinite(mesh.vertices).all()
assert changed == sorted(promoted)
commit = a.commit if a.commit.startswith('https://') else 'https://github.com/JavierMedel/video_to_data/commit/' + a.commit
command = [sys.executable, 'eval_reconstruction.py', '--episodes', str(episodes), '--sample', 'data/track_1_sample_submission.parquet', '--commit', commit, '--out', str(archive)]
with (REPORT/'official_pack.log').open('w') as log:
    subprocess.run(command, cwd=KIT, stdout=log, stderr=subprocess.STDOUT, check=True)
frame = validate_submission(archive, 'track_1', json.loads((KIT/'config/tracks.json').read_text()))
assert set(frame.row_id) == set(sample.row_id) and len(frame) == len(sample)
result = {'status': 'LOCAL_VALIDATION_PASS', 'episodes': 30, 'rows': len(frame), 'promoted_episodes': sorted(promoted),
          'all_human_arrays_identical_to_v3': True, 'archive': str(archive), 'sha256': sha(archive), 'commit': commit,
          'artifact_sha256': hashes, 'uploaded': False}
(REPORT/'candidate_validation.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k != 'artifact_sha256'}, indent=2), flush=True)
