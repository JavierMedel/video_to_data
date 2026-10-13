"""Pack visually approved improvements over the immutable second submission."""
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
from track1_rollout_common import ORDER, PILOTS, paths

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/v2d_challenge/track_1'
KIT=ROOT/'v2d_submission_kit/v2d_submission_kit'
REPORT=ROOT/'incident_reports/track1_improvement/rollout'
p=argparse.ArgumentParser(); p.add_argument('--commit',required=True); a=p.parse_args()
sys.path.insert(0,str(KIT))
from v2dlb.mhr_submission import EPISODE_FRAME, _rotations
from submit.v2d_submit import validate_submission

decisions=json.loads((REPORT/'promotion_decisions.json').read_text())
assert set(map(int,decisions))==set(ORDER), 'Every remaining episode needs a recorded review decision'
episodes=DATA/'improvement_rollout_v1/candidate_episodes'
archive=DATA/'submission_packages/track1_improvement_rollout_v1.parquet'
assert not episodes.exists() and not archive.exists(), 'Preserve any existing candidate package'
sources={}; promoted=[]
for i in range(30):
    q=paths(ROOT,i) if i not in PILOTS else None
    source=DATA/'improvement_pilot_v1/candidate_episodes'
    if q:
        decision=decisions[str(i)]
        assert decision['verdict'] in ('PROMOTE','RETAIN_SECOND_SUBMISSION')
        assert decision['visual_review_complete'] is True and decision['reason']
        if decision['verdict']=='PROMOTE':
            for filename in ('comparison.json','human_contact_review.json'):
                assert hashlib.sha256((q['work']/filename).read_bytes()).hexdigest()==decision['evidence_sha256'][filename]
            obj=json.loads((q['work']/'comparison.json').read_text())['summary']
            human=json.loads((q['work']/'human_contact_review.json').read_text())['summary']
            assert obj['refined']['negative_center_depth_frames']==0
            assert obj['refined']['visible_iou_mean']>=max(.3,obj['baseline']['visible_iou_mean']-.02)
            assert obj['refined']['translation_acceleration_cm_proxy']<=max(.1,obj['baseline']['translation_acceleration_cm_proxy']*1.1)
            assert human['candidate']['human_visible_iou_mean']>=human['baseline']['human_visible_iou_mean']-.03
            source=q['work']/'submission_episodes'; promoted.append(i)
    stem=f'episode_{i:06d}'
    for suffix in ('.npz','_object.glb'): assert (source/(stem+suffix)).exists()
    sources[i]=source
assert promoted, 'No supported improvements to submit'
episodes.mkdir(parents=True)
sample=pd.read_parquet(KIT/'data/track_1_sample_submission.parquet',columns=['row_id'])
parts=sample.row_id.str.split('_',expand=True)
hashes={}
for i,source in sources.items():
    stem=f'episode_{i:06d}'
    for suffix in ('.npz','_object.glb'):
        path=episodes/(stem+suffix); shutil.copy2(source/path.name,path)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==hashlib.sha256((source/path.name).read_bytes()).hexdigest()
        hashes[path.name]=digest
    with np.load(episodes/(stem+'.npz'),allow_pickle=False) as d:
        count=len(d['pose'])
        for key,shape in {'pose':(count,136),'scales':(68,),'shape':(45,),'object_rotation':(count,3,3),'object_translation':(count,3)}.items():
            assert d[key].shape==shape and np.isfinite(d[key]).all(), (i,key)
        assert np.isfinite(d['object_scale']).all() and float(d['object_scale'].reshape(()))>0
        scored=parts.loc[(parts[1].astype(int)==i)&(parts[2].astype(int)!=EPISODE_FRAME),2].astype(int)
        assert count>scored.max()
        _rotations(d['object_rotation'])
    mesh=trimesh.load(episodes/(stem+'_object.glb'),force='mesh')
    assert len(mesh.faces) and len(mesh.vertices) and np.isfinite(mesh.vertices).all() and mesh.area>0
command=[sys.executable,'eval_reconstruction.py','--episodes',str(episodes),'--sample','data/track_1_sample_submission.parquet','--commit','https://github.com/JavierMedel/video_to_data/commit/'+a.commit,'--out',str(archive)]
with (REPORT/'official_pack.log').open('w') as log: subprocess.run(command,cwd=KIT,stdout=log,stderr=subprocess.STDOUT,check=True)
frame=validate_submission(archive,'track_1',json.loads((KIT/'config/tracks.json').read_text()))
assert set(frame.row_id)==set(sample.row_id)
result={'status':'LOCAL_VALIDATION_PASS','rows':len(frame),'episodes':30,'promoted_episodes':promoted,'retained_episodes':[i for i in range(30) if i not in promoted],'archive':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'commit':a.commit,'artifact_sha256':hashes,'uploaded':False}
(REPORT/'candidate_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='artifact_sha256'}),flush=True)
