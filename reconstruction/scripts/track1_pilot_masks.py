"""Sequential corrected SAM2 pilot; baseline masks are never overwritten."""
import json
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT/'data/v2d_challenge/track_1/improvement_pilot_v1'
REPORT=ROOT/'incident_reports/track1_improvement/progress.md'
REPORT.parent.mkdir(exist_ok=True)
config=json.loads((ROOT/'scripts/track1_pilot_prompts.json').read_text())
for i in (6,28,18):
    stem=f'episode_{i:06d}'; work=PILOT/stem
    work.mkdir(parents=True,exist_ok=True)
    prompt=work/'object_prompts.json'
    if not prompt.exists(): prompt.write_text(json.dumps(config[str(i)]['sam2'],indent=2))
    assert json.loads(prompt.read_text())==config[str(i)]['sam2'], 'Changed prompt requires a new candidate directory'
    files=list((work/'object_masks/1').glob('*.png'))
    if len(files)==config[str(i)]['expected_frames'] and (work/'mask_review.json').exists():
        print('REVIEWED_MASKS_PRESERVED',stem,flush=True)
        continue
    command=['docker','run','--rm','--gpus','all','--user','1000:1000','--entrypoint','python','-e','PYTHONUNBUFFERED=1','-v',f'{ROOT}:/repo','-v',f'{ROOT}/modules/v2d_sam2/lib:/workspace/v2d_sam2/lib:ro','v2d_sam2:blackwell','-m','v2d.sam2.lib.video_to_masks','--video_path',f'/repo/data/v2d_challenge/track_1/videos/chunk-000/observation.images.exo_camera/{stem}.mp4','--prompts_path',f'/repo/data/v2d_challenge/track_1/improvement_pilot_v1/{stem}/object_prompts.json','--masks_dir',f'/repo/data/v2d_challenge/track_1/improvement_pilot_v1/{stem}/object_masks','--weights_dir','/repo/data/weights/sam2']
    with REPORT.open('a') as f: f.write('\nSTART '+stem+' corrected masks\n`'+' '.join(command)+'`\n')
    with (work/'sam2.log').open('a') as log: result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    with REPORT.open('a') as f: f.write(f'\n{stem} masks exit {result.returncode}\n')
    if result.returncode: raise SystemExit(result.returncode)
    print('MASKS_COMPLETE',stem,flush=True)
