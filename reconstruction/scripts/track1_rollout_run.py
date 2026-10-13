"""Sequential resumable rollout stages, with explicit candidate comparisons."""
import argparse
import datetime
import json
import subprocess
from pathlib import Path
from track1_rollout_common import paths

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(); p.add_argument('--episodes',type=int,nargs='+',required=True)
p.add_argument('--allow-episode-16',action='store_true'); a=p.parse_args()
report_dir=ROOT/'incident_reports/track1_improvement/rollout'; report_dir.mkdir(parents=True,exist_ok=True)
progress=report_dir/'progress.md'; status_file=report_dir/'status.json'
container_root='/repo/data/v2d_challenge/track_1/improvement_rollout_v1'
base=['docker','run','--rm','--gpus','all','--user','1000:1000','--entrypoint','python','-e','PYTHONUNBUFFERED=1','-e','XDG_CACHE_HOME=/tmp/v2d-rollout-cache','-v',f'{ROOT}:/repo','-v',f'{ROOT}/modules/v2d_cari4d/lib:/workspace/v2d_cari4d/lib:ro','v2d_cari4d:blackwell']
completed=[]
for i in a.episodes:
    if i==16 and not a.allow_episode_16: raise SystemExit('Episode 16 requires approved isolated rerun')
    q=paths(ROOT,i); work=q['work']
    review=json.loads((work/'review_approval.json').read_text()); assert review['verdict']=='PASS'
    if (work/'comparison.json').exists() and (work/'human_contact_review.json').exists():
        assert (work/'submission_episodes'/(q['stem']+'.npz')).exists() and (work/'submission_episodes'/(q['stem']+'_object.glb')).exists()
        completed.append(i)
        print('ROLLOUT_COMPLETED_CANDIDATE_PRESERVED',i,flush=True)
        continue
    flag=['--allow-episode-16'] if i==16 else []
    jobs=[('pipeline',base+['/repo/scripts/track1_pilot_reconstruct_container.py','--episode',str(i),'--work-root',container_root]+flag),
          ('convert',base+['/repo/scripts/track1_convert_refined.py','--root','/repo','--episode',str(i),'--run',f'{container_root}/{q["stem"]}/run','--out-dir',f'{container_root}/{q["stem"]}/submission_episodes','--progress-report','/repo/incident_reports/track1_improvement/rollout/progress.md']+flag),
          ('compare',base+['/repo/scripts/track1_pilot_compare.py','/repo',str(i),'--work-root',container_root,'--baseline-pairs','/repo/data/v2d_challenge/track_1/improvement_pilot_v1/candidate_episodes','--report-dir','/repo/incident_reports/track1_improvement/rollout']),
          ('human_contact',base+['/repo/scripts/track1_rollout_human_contact.py','--root','/repo','--episode',str(i)])]
    for stage,command in jobs:
        if stage=='pipeline' and (work/'comparison.json').exists(): continue
        if stage=='convert' and (work/'submission_episodes'/(q['stem']+'.npz')).exists() and (work/'submission_episodes'/(q['stem']+'_object.glb')).exists(): continue
        if stage=='compare' and (work/'comparison.json').exists(): continue
        if stage=='human_contact' and (work/'human_contact_review.json').exists(): continue
        status={'state':'RUNNING','episode':i,'stage':stage,'completed':completed,'requested':a.episodes,'updated':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        status_file.write_text(json.dumps(status,indent=2)+'\n')
        with progress.open('a') as f: f.write(f'\nSTART {q["stem"]} {stage}: '+json.dumps(command)+'\n')
        with (work/(stage+'.log')).open('a') as log: result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            status.update(state='FAILED',exit_code=result.returncode)
            status_file.write_text(json.dumps(status,indent=2)+'\n')
            raise SystemExit(result.returncode)
        with progress.open('a') as f: f.write(f'\nCOMPLETE {q["stem"]} {stage}\n')
        print('ROLLOUT_STAGE_COMPLETE',i,stage,flush=True)
    completed.append(i)
status={'state':'BATCH_COMPLETE','completed':completed,'requested':a.episodes,'updated':datetime.datetime.now(datetime.timezone.utc).isoformat()}
status_file.write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status),flush=True)
