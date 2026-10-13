"""Read-only progress snapshot; run without GPU or Docker access."""
import argparse
import json
from pathlib import Path
from track1_rollout_common import ORDER

p=argparse.ArgumentParser(); p.add_argument('--compact',action='store_true'); a=p.parse_args()
root=Path(__file__).resolve().parents[1]
rollout=root/'data/v2d_challenge/track_1/improvement_rollout_v1'
report=root/'incident_reports/track1_improvement/rollout'
converted=[]; compared=[]; reviewed=[]; prepared=[]; logs=[]
for i in ORDER:
    stem=f'episode_{i:06d}'; work=rollout/stem
    if (work/'review_approval.json').exists(): prepared.append(i)
    if (work/'submission_episodes'/(stem+'.npz')).exists() and (work/'submission_episodes'/(stem+'_object.glb')).exists(): converted.append(i)
    if (work/'comparison.json').exists(): compared.append(i)
    if (work/'human_contact_review.json').exists(): reviewed.append(i)
    for stage in ('pipeline','convert','compare','human_contact','sam2','sam3d'):
        p=work/(stage+'.log')
        if p.exists(): logs.append((p.stat().st_mtime,i,stage,p))
result={'total_remaining_episodes':27,'inputs_prepared':prepared,'converted':converted,'object_comparisons_complete':compared,'human_contact_reviews_complete':reviewed}
if (report/'status.json').exists(): result['runner_status']=json.loads((report/'status.json').read_text())
if logs:
    timestamp,i,stage,p=max(logs)
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-2000)); tail=f.read().decode(errors='replace').replace('\r','\n').splitlines()
    result['latest_gpu_log']={'episode':i,'stage':stage,'mtime_epoch':timestamp,'last_lines':tail[-3:]}
if a.compact:
    result={**{k:len(result[k]) for k in ('inputs_prepared','converted','object_comparisons_complete','human_contact_reviews_complete')},'total':27,'runner':result.get('runner_status',{}),'latest_gpu_log':result.get('latest_gpu_log',{})}
    result['runner'].pop('requested',None)
    result['runner']['completed_count']=len(result['runner'].pop('completed',[]))
    print(json.dumps(result))
else: print(json.dumps(result,indent=2))
