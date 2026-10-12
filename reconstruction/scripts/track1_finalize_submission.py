"""Validate and upload one authorized first submission after the runner exits.

Never retries an upload. The exclusive authorization marker and uploader receipt
preserve ambiguous outcomes across interruptions.
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import trimesh

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/v2d_challenge/track_1'
KIT = ROOT/'v2d_submission_kit/v2d_submission_kit'
REPORT = ROOT/'incident_reports/track1_submission_progress.md'
OUT = DATA/'submission_packages'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(KIT))
from v2dlb.mhr_submission import EPISODE_FRAME, _rotations

def record(message):
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with REPORT.open('a') as f: f.write(f'\n{stamp} {message}\n')
    print(message,flush=True)

def command(args,label):
    logfile=OUT/(label+'.log')
    with logfile.open('a') as f:
        result=subprocess.run(args,cwd=KIT,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:
        excerpt='\n'.join(logfile.read_text(errors='replace').splitlines()[-20:])
        record(f'FINALIZATION BLOCKED {label}; exit {result.returncode}\n```\n{excerpt}\n```')
        raise SystemExit(result.returncode)
    record(f'PASS {label}; log {logfile.relative_to(ROOT)}')

p=argparse.ArgumentParser()
p.add_argument('--wait-runner-pid',type=int,required=True)
p.add_argument('--commit',required=True)
a=p.parse_args()
commit_url='https://github.com/JavierMedel/video_to_data/commit/'+a.commit
record(f'Finalizer watching runner PID {a.wait_runner_pid}; upload remains gated on all 30 pairs and official validation.')
last_count=-1
while True:
    count=sum((DATA/'submission_episodes'/f'episode_{i:06d}.npz').exists() and (DATA/'submission_episodes'/f'episode_{i:06d}_object.glb').exists() for i in range(30))
    if count != last_count:
        record(f'Artifact pairs present: {count}/30 (final validation pending)'); last_count=count
    try: running=b'track1_resume_execution.py' in Path(f'/proc/{a.wait_runner_pid}/cmdline').read_bytes()
    except FileNotFoundError: running=False
    if not running: break
    time.sleep(30)

sample=pd.read_parquet(KIT/'data/track_1_sample_submission.parquet',columns=['row_id'])
parts=sample.row_id.str.split('_',expand=True)
expected=set(parts[1].astype(int))
assert expected==set(range(30)), 'Official episode set changed'
for i in range(30):
    path=DATA/'submission_episodes'/f'episode_{i:06d}.npz'
    mesh_path=path.with_name(f'episode_{i:06d}_object.glb')
    if not path.exists() or not mesh_path.exists():
        record(f'FINALIZATION BLOCKED: missing pair for episode {i:06d}. No upload attempted.'); raise SystemExit(1)
    d=dict(np.load(path,allow_pickle=False)); t=len(d['pose'])
    assert d['pose'].shape==(t,136) and d['scales'].shape==(68,) and d['shape'].shape==(45,), path
    assert d['object_rotation'].shape==(t,3,3) and d['object_translation'].shape==(t,3), path
    for key in ('pose','scales','shape','object_rotation','object_translation','object_scale'):
        assert np.isfinite(d[key]).all(), (path,key)
    assert float(d['object_scale'].reshape(()))>0, path
    scored=parts.loc[(parts[1].astype(int)==i)&(parts[2].astype(int)!=EPISODE_FRAME),2].astype(int)
    assert t>scored.max(), (path,t,scored.max())
    _rotations(d['object_rotation'])
    mesh=trimesh.load(mesh_path,force='mesh')
    assert len(mesh.faces) and len(mesh.vertices) and np.isfinite(mesh.vertices).all() and mesh.area>0, mesh_path
record('PASS all 30 NPZ/GLB pairs: required shapes, scored-frame coverage, finite values, rotations, positive scales, finite triangle geometry.')
archive=OUT/'track1_first.parquet'
receipt=OUT/'track1_first.kaggle-receipt.json'
guard=OUT/'track1_first.upload-authorized-used.json'
if guard.exists() or receipt.exists():
    record('Existing upload authorization marker or receipt found; refusing another upload. Poll the existing receipt.'); raise SystemExit(1)
command([sys.executable,'eval_reconstruction.py','--episodes',str(DATA/'submission_episodes'),'--sample','data/track_1_sample_submission.parquet','--commit',commit_url,'--out',str(archive)],'official_pack')
command([sys.executable,'tools/preflight.py','--track','track_1'],'official_preflight')
command([sys.executable,'submit/v2d_submit.py',str(archive),'--track','track_1','-m','First CARI4D Blackwell reconstruction','--dry-run'],'official_dry_run')
with guard.open('x') as f:
    json.dump({'commit_url':commit_url,'file':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'authorization':'User authorized one first Track 1 leaderboard version after official validation.'},f,indent=2)
record(f'Official validation succeeded; final file {archive}. Consuming the one authorized version across all five Track 1 metric competitions.')
command([sys.executable,'submit/v2d_submit.py',str(archive),'--track','track_1','-m','First CARI4D Blackwell reconstruction','--wait','--receipt',str(receipt)],'kaggle_upload')
record('Kaggle receipt: '+receipt.read_text())
