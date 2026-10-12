"""Wait for reviewed pilot results and pack a local candidate. Never uploads."""
import argparse
import datetime
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/v2d_challenge/track_1'; PILOT=DATA/'improvement_pilot_v1'
KIT=ROOT/'v2d_submission_kit/v2d_submission_kit'; REPORT=ROOT/'incident_reports/track1_improvement/progress.md'
p=argparse.ArgumentParser(); p.add_argument('--wait-runner-pid',type=int,required=True); p.add_argument('--commit',required=True); a=p.parse_args()
def record(message):
    with REPORT.open('a') as f: f.write('\n'+datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+message+'\n')
    print(message,flush=True)
record('Local candidate builder waiting for all three pilot comparisons. No Kaggle uploader is called.')
while True:
    try: running=b'track1_pilot_reconstruct.py' in Path(f'/proc/{a.wait_runner_pid}/cmdline').read_bytes()
    except FileNotFoundError: running=False
    if not running: break
    time.sleep(30)
comparisons={}
for i in (6,18,28):
    path=PILOT/f'episode_{i:06d}/comparison.json'
    if not path.exists(): record(f'BLOCKED: pilot comparison missing for episode {i}. No candidate packed.'); raise SystemExit(1)
    c=json.loads(path.read_text()); s=c['summary']; new=s['refined']; old=s['baseline']
    if new['visible_iou_mean']<max(0.3,old['visible_iou_mean']+0.1) or new['negative_center_depth_frames'] or new['translation_acceleration_cm_proxy']>old['translation_acceleration_cm_proxy']:
        record(f'REVIEW REQUIRED: episode {i} does not pass conservative local promotion checks. No candidate packed.'); raise SystemExit(1)
    comparisons[str(i)]=c
episodes=PILOT/'candidate_episodes'
if episodes.exists(): record('Candidate episode directory already exists; preserving it.'); raise SystemExit(1)
episodes.mkdir()
for i in range(30):
    stem=f'episode_{i:06d}'
    source=PILOT/stem/'submission_episodes' if i in (6,18,28) else DATA/'submission_episodes'
    for suffix in ('.npz','_object.glb'): shutil.copy2(source/(stem+suffix),episodes/(stem+suffix))
    d=dict(np.load(episodes/(stem+'.npz'),allow_pickle=False))
    for key in ('pose','scales','shape','object_rotation','object_translation','object_scale'): assert np.isfinite(d[key]).all(), (i,key)
archive=DATA/'submission_packages/track1_improvement_pilot_v1.parquet'
if archive.exists(): raise FileExistsError(archive)
cmd=[sys.executable,'eval_reconstruction.py','--episodes',str(episodes),'--sample','data/track_1_sample_submission.parquet','--commit','https://github.com/JavierMedel/video_to_data/commit/'+a.commit,'--out',str(archive)]
log=PILOT/'official_pack.log'
with log.open('w') as f: result=subprocess.run(cmd,cwd=KIT,stdout=f,stderr=subprocess.STDOUT)
if result.returncode: record('Official packer failed; see '+str(log)); raise SystemExit(result.returncode)
sys.path.insert(0,str(KIT))
from submit.v2d_submit import validate_submission
frame=validate_submission(archive,'track_1',json.loads((KIT/'config/tracks.json').read_text()))
result={'status':'LOCAL_VALIDATION_PASS','rows':len(frame),'episodes':30,'improved_episodes':[6,18,28],'baseline_episodes_preserved':27,'archive':str(archive),'commit':a.commit,'uploaded':False,'comparisons':{i:c['summary'] for i,c in comparisons.items()}}
(PILOT/'candidate_validation.json').write_text(json.dumps(result,indent=2))
record('LOCAL CANDIDATE READY: '+json.dumps(result))
