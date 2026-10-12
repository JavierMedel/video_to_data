"""Sequential pilot reconstruction; never uploads or writes baseline arrays."""
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT/'data/v2d_challenge/track_1/improvement_pilot_v1'
REPORT=ROOT/'incident_reports/track1_improvement/progress.md'
for i in (6,28,18):
    stem=f'episode_{i:06d}'; work=PILOT/stem
    assert (work/'mesh/object_metric.glb').exists(), 'Reviewed regenerated mesh required'
    base=['docker','run','--rm','--gpus','all','--user','1000:1000','--entrypoint','python','-e','PYTHONUNBUFFERED=1','-e','XDG_CACHE_HOME=/tmp/v2d-pilot-cache','-v',f'{ROOT}:/repo','-v',f'{ROOT}/modules/v2d_cari4d/lib:/workspace/v2d_cari4d/lib:ro','v2d_cari4d:blackwell']
    jobs=[('pipeline',base+['/repo/scripts/track1_pilot_reconstruct_container.py','--episode',str(i)]),('convert',base+['/repo/scripts/track1_convert_refined.py','--root','/repo','--episode',str(i),'--run',f'/repo/data/v2d_challenge/track_1/improvement_pilot_v1/{stem}/run','--out-dir',f'/repo/data/v2d_challenge/track_1/improvement_pilot_v1/{stem}/submission_episodes','--progress-report','/repo/incident_reports/track1_improvement/progress.md'])]
    jobs.append(('compare',base+['/repo/scripts/track1_pilot_compare.py','/repo',str(i)]))
    for label,cmd in jobs:
        if label=='convert' and (work/'submission_episodes'/(stem+'.npz')).exists() and (work/'submission_episodes'/(stem+'_object.glb')).exists():
            print('PILOT_CONVERSION_PRESERVED',stem,flush=True)
            continue
        with REPORT.open('a') as f: f.write(f'\nSTART {stem} {label}\n`'+ ' '.join(cmd)+'`\n')
        with (work/(label+'.log')).open('a') as log: result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            excerpt='\n'.join((work/(label+'.log')).read_text(errors='replace').splitlines()[-25:])
            with REPORT.open('a') as f: f.write(f'\nFAILED {stem} {label}: exit {result.returncode}\n```\n{excerpt}\n```\n')
            raise SystemExit(result.returncode)
        with REPORT.open('a') as f: f.write(f'\nCOMPLETE {stem} {label}\n')
        print('PILOT_STAGE_COMPLETE',stem,label,flush=True)
