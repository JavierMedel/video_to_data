"""Prepare reviewed inputs and run all 27 remaining episodes sequentially."""
import json
import subprocess
import sys
import time
from pathlib import Path
from track1_rollout_common import ORDER, paths

ROOT=Path(__file__).resolve().parents[1]
preflight=ROOT/'incident_reports/track1_improvement/rollout/cache_preflight.json'
while not preflight.exists(): time.sleep(30)
assert all(row['status']=='PASS' for row in json.loads(preflight.read_text())), 'Resolve cache preflight failures before the rollout'
references={0:0,3:3,6:6,9:10,12:12,15:15,18:18,21:21,24:24,27:28}
corrections={int(i) for i in json.loads((ROOT/'scripts/track1_rollout_prompts.json').read_text())}
for i in (0,3,10,12,15,21,24):
    while not (paths(ROOT,i)['work']/'mesh/mesh_validation.json').exists(): time.sleep(30)
for i in sorted(corrections):
    while not (paths(ROOT,i)['work']/'corrected_mask_approval.json').exists(): time.sleep(30)
    assert json.loads((paths(ROOT,i)['work']/'corrected_mask_approval.json').read_text())['verdict']=='PASS'
decisions={r['episode']:r for r in json.loads((ROOT/'incident_reports/track1_improvement/rollout/mask_review_decisions.json').read_text())}
for i in ORDER:
    q=paths(ROOT,i)
    if (q['work']/'review_approval.json').exists(): continue
    reference=references[(i//3)*3]
    base=['docker','run','--rm','--user','1000:1000','--entrypoint','python','-e','PYTHONUNBUFFERED=1','-v',f'{ROOT}:/repo','v2d_cari4d:blackwell','/repo/scripts/track1_rollout_prepare.py','--root','/repo','--episodes',str(i),'--mesh-episode',str(reference),'--review-note',decisions[i]['note']]
    if reference not in (6,18,28): base += ['--mesh-source',f'/repo/data/v2d_challenge/track_1/improvement_rollout_v1/episode_{reference:06d}/mesh/object_metric.glb']
    if i in corrections: base.append('--corrected-masks')
    else: assert decisions[i]['mask_review']=='PASS_SAMPLED_VISUAL_REVIEW'
    subprocess.run(base,check=True)
subprocess.run([sys.executable,str(ROOT/'scripts/track1_rollout_run.py'),'--episodes',*[str(i) for i in ORDER],'--allow-episode-16'],check=True)
