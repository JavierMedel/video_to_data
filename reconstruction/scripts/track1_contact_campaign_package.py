"""Stage the reviewed 30-episode candidate; optionally pack with real provenance.

Never uploads. Only final reviewed promotions replace v6. The official packer
must preserve every baseline numeric row except promoted object translations.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh

from track1_episode7_ablation import verify_bundle


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--campaign', default='contact_smoothing_v8')
    p.add_argument('--commit', help='Actual producing commit; omit to stage and validate pairs only')
    a = p.parse_args(); root = a.root.resolve()
    assert Path(a.campaign).name == a.campaign and a.campaign not in ('.', '..')
    data = root/'data/v2d_challenge/track_1'; report = root/'incident_reports/track1_improvement'/a.campaign
    campaign = json.loads((report/'campaign.json').read_text())
    review = json.loads((report/'selection_review.json').read_text())
    assert campaign['state'] == review['state'] == 'COMPLETE'
    assert set(review['episodes']) == set(map(str, range(30))) and not review['pending_visual_review']
    assert all(d['verdict'] in ('PROMOTE', 'RETAIN_V6') for d in review['episodes'].values())
    assert review['promoted'], 'No justified change to submit'
    controls = data/'cdo_v6_experiments/candidate_episodes'
    episodes = data/a.campaign/'submission_episodes'; episodes.mkdir(parents=True, exist_ok=True)
    kit = root/'v2d_submission_kit/v2d_submission_kit'; sys.path.insert(0, str(kit))
    from v2dlb.mhr_submission import EPISODE_FRAME, _rotations, array_order
    from submit.v2d_submit import validate_submission
    sample = pd.read_parquet(kit/'data/track_1_sample_submission.parquet', columns=['row_id'])
    parts = sample.row_id.str.split('_', expand=True)
    assert set(parts[1].astype(int)) == set(range(30))
    hashes = {}; changed = []
    for episode in range(30):
        stem = f'episode_{episode:06d}'; decision = review['episodes'][str(episode)]
        source = root/decision['source'] if decision['verdict']=='PROMOTE' else controls
        if decision['verdict']=='PROMOTE':
            assert all(decision['gates'].values())
            assert sha(root/decision['status']) == decision['status_sha256']
        for suffix in ('.npz', '_object.glb'):
            name = stem+suffix; assert sha(controls/name) == campaign['control_sha256'][name]
            destination = episodes/name
            if destination.exists(): assert sha(destination) == sha(source/name), 'Preserve existing staged artifacts'
            else: shutil.copy2(source/name, destination)
            hashes[name] = sha(destination)
        control = dict(np.load(controls/(stem+'.npz'), allow_pickle=False))
        candidate = dict(np.load(episodes/(stem+'.npz'), allow_pickle=False)); verify_bundle(control, candidate)
        count = len(candidate['pose'])
        for key, shape in {'pose': (count,136), 'scales': (68,), 'shape': (45,), 'object_rotation': (count,3,3), 'object_translation': (count,3)}.items():
            assert candidate[key].shape == shape and np.isfinite(candidate[key]).all(), (episode,key)
        assert np.isfinite(candidate['object_scale']).all() and float(candidate['object_scale']) > 0
        _rotations(candidate['object_rotation'])
        frames = parts.loc[(parts[1].astype(int)==episode)&(parts[2].astype(int)!=EPISODE_FRAME),2].astype(int)
        assert len(frames) and count > frames.max()
        assert hashes[stem+'_object.glb'] == campaign['control_sha256'][stem+'_object.glb']
        mesh = trimesh.load(episodes/(stem+'_object.glb'), force='mesh', process=False)
        assert len(mesh.faces) and len(mesh.vertices) and np.isfinite(mesh.vertices).all() and mesh.area > 0
        if not np.array_equal(control['object_translation'], candidate['object_translation']): changed.append(episode)
    assert changed == review['promoted']
    result = {'state': 'STAGED_VALIDATED', 'episodes': 30, 'expected_rows': len(sample), 'changed_episodes': changed,
              'control_version': 6, 'all_non_translation_arrays_identical': True, 'all_meshes_identical': True,
              'source_hashes_preserved': True, 'artifact_sha256': hashes, 'selection_review_sha256': sha(report/'selection_review.json'),
              'uploaded': False}
    (report/'package_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    if a.commit:
        assert re.fullmatch(r'[0-9a-f]{40}', a.commit), 'Require a real full commit hash'
        subprocess.run(['git','cat-file','-e',a.commit+'^{commit}'],cwd=root,check=True)
        critical = ['scripts/track1_contact_campaign.py','scripts/track1_contact_campaign_package.py','scripts/track1_contact_campaign_review.py',
                    'scripts/track1_episode7_contact_smoothing.py','scripts/track1_episode7_ablation.py','scripts/track1_contact_smoothing.py',
                    'scripts/track1_cdo_full_image_review.py','scripts/track1_cdo_hand_penetration_review.py','scripts/track1_cdo_render_candidate.py',
                    'modules/v2d_cari4d/lib/cari4d/lib_mhr/fused_mesh_penetration_review.py']
        subprocess.run(['git','ls-files','--error-unmatch',*critical],cwd=root,check=True,stdout=subprocess.DEVNULL)
        subprocess.run(['git','diff','--quiet',a.commit,'--',*critical],cwd=root,check=True)
        commit = 'https://github.com/JavierMedel/video_to_data/commit/'+a.commit
        archive = data/'submission_packages'/f'track1_{a.campaign}.parquet'
        if not archive.exists():
            temporary = archive.with_name(archive.stem+'.tmp.parquet')
            assert not temporary.exists(), 'Inspect preserved interrupted pack before retrying'
            with (report/'official_pack.log').open('w') as log:
                subprocess.run([sys.executable,'eval_reconstruction.py','--episodes',str(episodes),'--sample','data/track_1_sample_submission.parquet',
                                '--commit',commit,'--out',str(temporary)],cwd=kit,stdout=log,stderr=subprocess.STDOUT,check=True)
            temporary.rename(archive)
        frame = validate_submission(archive,'track_1',json.loads((kit/'config/tracks.json').read_text()))
        assert set(frame.row_id) == set(sample.row_id) and len(frame) == len(sample)
        assert frame.code_commit_url.nunique() == 1 and frame.code_commit_url.iloc[0] == commit
        baseline = pd.read_parquet(data/'submission_packages/track1_cdo_v6.parquet').set_index('row_id')
        packed = frame.set_index('row_id').loc[baseline.index]
        # Only object_translation rows may differ. This also detects incidental
        # mesh simplification, packing or human conversion changes.
        translation_index = array_order().index('object_translation')
        allowed_rows = np.asarray(baseline.index.str.startswith(tuple(f't1_{e:06d}_' for e in changed)) & baseline.index.str.contains(f'_{translation_index}_'), dtype=bool)
        difference = ~np.all(baseline[['x','y','z']].to_numpy() == packed[['x','y','z']].to_numpy(),axis=1)
        assert not difference[~allowed_rows].any(), 'Unintended packed values changed'
        result.update(state='OFFICIAL_PACK_VALIDATED',rows=len(frame),archive=str(archive),sha256=sha(archive),commit=commit,
                      packed_non_translation_rows_identical=True,changed_packed_rows=int(difference.sum()))
        (report/'package_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_sha256'},indent=2))


if __name__ == '__main__': main()
