"""Correct only reviewed wrong-object prompts and prepare review sheets."""
import argparse
import json
import subprocess
import time
from pathlib import Path
from track1_rollout_common import paths

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(); p.add_argument('--episodes',type=int,nargs='+',required=True)
p.add_argument('--wait-for-episode',type=int)
p.add_argument('--wait-for-human-review',type=int); a=p.parse_args()
if a.wait_for_human_review is not None:
    while not (paths(ROOT,a.wait_for_human_review)['work']/'human_contact_review.json').exists(): time.sleep(30)
    time.sleep(2)
if a.wait_for_episode is not None:
    while not (paths(ROOT,a.wait_for_episode)['work']/'comparison.json').exists(): time.sleep(30)
    time.sleep(2)
config=json.loads((ROOT/'scripts/track1_rollout_prompts.json').read_text())
for i in a.episodes:
    q=paths(ROOT,i); work=q['work']; work.mkdir(parents=True,exist_ok=True)
    prompt=work/'object_prompts.json'; expected=config[str(i)]['sam2']
    if prompt.exists(): assert json.loads(prompt.read_text())==expected, 'New prompt requires a new candidate directory'
    else: prompt.write_text(json.dumps(expected,indent=2)+'\n')
    prefix=f'/repo/data/v2d_challenge/track_1/improvement_rollout_v1/{q["stem"]}'
    command=['docker','run','--rm','--gpus','all','--user','1000:1000','--entrypoint','python','-e','PYTHONUNBUFFERED=1','-v',f'{ROOT}:/repo','-v',f'{ROOT}/modules/v2d_sam2/lib:/workspace/v2d_sam2/lib:ro','v2d_sam2:blackwell','-m','v2d.sam2.lib.video_to_masks','--video_path',f'/repo/data/v2d_challenge/track_1/videos/chunk-000/observation.images.exo_camera/{q["stem"]}.mp4','--prompts_path',prefix+'/object_prompts.json','--masks_dir',prefix+'/object_masks','--weights_dir','/repo/data/weights/sam2']
    with (work/'sam2.log').open('a') as log: subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (work/'mask_review.log').open('a') as log:
        subprocess.run(['docker','run','--rm','--user','1000:1000','--entrypoint','python','-v',f'{ROOT}:/repo','v2d_cari4d:blackwell','/repo/scripts/track1_rollout_review.py','--root','/repo','--episodes',str(i),'--corrected'],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('ROLLOUT_CORRECTED_MASKS_READY_FOR_REVIEW',i,flush=True)
