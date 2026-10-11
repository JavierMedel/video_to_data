"""Sequential durable recovery runner. Never executes Episode 16."""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/v2d_challenge/track_1'
REPORT = ROOT/'incident_reports/track1_submission_progress.md'
LOGS = DATA/'logs/resume_execution'
LOGS.mkdir(parents=True,exist_ok=True)

def record(message):
    with REPORT.open('a') as f:
        f.write('\n'+datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+message+'\n')
    print(message,flush=True)

def execute(command, label):
    path = LOGS/(label+'.log')
    record('START '+label+'; command: `'+ ' '.join(command)+'`; log: '+str(path.relative_to(ROOT)))
    with path.open('a') as log:
        code = subprocess.call(command,stdout=log,stderr=subprocess.STDOUT)
    if code:
        excerpt = '\n'.join(path.read_text(errors='replace').splitlines()[-25:])
        record(f'FAILED {label}; exit {code}\n```\n{excerpt}\n```')
        raise SystemExit(code)
    record('COMPLETE '+label)

def convert(i,run):
    if i == 16: raise RuntimeError('Protected episode')
    target = DATA/'submission_episodes'/f'episode_{i:06d}.npz'
    mesh = target.with_name(f'episode_{i:06d}_object.glb')
    if target.exists() and mesh.exists(): return
    execute(['docker','run','--rm','--gpus','all','--user','1000:1000','--entrypoint','python','-e','PYTHONUNBUFFERED=1','-e','HOME=/tmp','-v',f'{ROOT}:/repo','v2d_cari4d:blackwell','/repo/scripts/track1_convert_refined.py','--root','/repo','--episode',str(i),'--run','/repo/'+str(run.relative_to(ROOT))],f'episode_{i:06d}_convert')

def reconstruct(i):
    if i == 16: raise RuntimeError('Protected episode')
    work = DATA/f'episode_{i:06d}'
    if i == 15:
        video = work/'2026-09-10_16-14-06_foam_grass_block_rotate_02.0.color.mp4'
        mesh = work/'sam3d/object.glb'; weights = ROOT/'data/weights/cari4d_episode15_overlay'
        output = DATA/'cari4d_runs/episode_000015_test'
    else:
        video = work/f'episode_{i:06d}.0.color.mp4'
        source = DATA/'videos/chunk-000/observation.images.exo_camera'/f'episode_{i:06d}.mp4'
        if not video.exists(): video.hardlink_to(source)
        mesh = DATA/f'episode_{i//3*3:06d}_candidate_mesh/object_metric.glb'
        if i == 17: mesh = DATA/'episode_000016/sam3d/mesh/object.glb'
        weights = ROOT/'data/weights'
        output = DATA/f'cari4d_runs/track1_resume/episode_{i:06d}'
    assert mesh.is_file(), str(mesh)
    masks = work/'cari4d_masks.h5'
    output.mkdir(parents=True,exist_ok=True)
    command = ['docker','run','--rm','--gpus','all','--user','1000:1000','--entrypoint','python','-e','HOME=/tmp','-e','PYTHONUNBUFFERED=1','-e','V2D_TRACK1_DEFER_RENDER=1',
        '-v',f'{ROOT}/scripts:/recovery:ro','-v',f'{ROOT}/modules/v2d_cari4d/lib:/workspace/v2d_cari4d/lib:ro',
        '-v',f'{video.parent}:/data/video_path','-v',f'{mesh.parent}:/data/object_mesh_path','-v',f'{weights.parent}:/data/weights_path','-v',f'{output.parent}:/data/output_dir',
        'v2d_cari4d:blackwell','/recovery/track1_run_episode_container.py',
        '--video','/data/video_path/'+video.name,'--masks','/data/video_path/'+masks.name,'--mesh','/data/object_mesh_path/'+mesh.name,
        '--weights','/data/weights_path/'+weights.name,'--output','/data/output_dir/'+output.name]
    execute(command,f'episode_{i:06d}_pipeline')
    convert(i,output/video.name.removesuffix('.0.color.mp4'))

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--episodes',nargs='+',type=int,required=True); args=p.parse_args()
    for i in args.episodes:
        if i == 16: continue
        target=DATA/'submission_episodes'/f'episode_{i:06d}.npz'
        if target.exists() and target.with_name(f'episode_{i:06d}_object.glb').exists(): continue
        if i in (3,4,5): convert(i,DATA/f'cari4d_runs/track1_submission/episode_{i:06d}/episode_{i:06d}')
        else: reconstruct(i)
    record('Requested episode batch completed; no Kaggle upload performed by this runner.')
