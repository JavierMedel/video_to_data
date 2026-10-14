"""Run isolated, identically seeded one-factor refinement experiments."""
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'incident_reports/track1_improvement/cdo_v4'
DATA = ROOT / 'data/v2d_challenge/track_1'
REPORT.mkdir(parents=True, exist_ok=True)
episode = 12
source = DATA / f'improvement_rollout_v1/episode_{episode:06d}'
original = json.loads((source / 'run/.stages/07_refinement.json').read_text())['signature']['command']
seed_code = 'import random, numpy as np, torch, runpy; random.seed(42); np.random.seed(42); torch.manual_seed(42); runpy.run_module("learning.training.mhr_opt_refineout", run_name="__main__")'
variants = [('control_100_batch64', 100, 64, False), ('rotation_100_batch64', 100, 64, True), ('steps300_batch64', 300, 64, False)]
results = []
for name, steps, batch, rotation in variants:
    host = DATA / f'cdo_v4_experiments/refinement/episode_{episode:06d}' / name
    host.mkdir(parents=True, exist_ok=True)
    container = Path('/repo') / host.relative_to(ROOT)
    if (host / 'refined.pth').exists():
        raise FileExistsError(f'Preserve existing experiment: {host}')
    command = original[3:].copy()
    for flag, value in {'--out': str(container / 'refined.pth'), '--postopt-checkpoint': str(container / 'checkpoint.pth'), '--num-steps': str(steps), '--batch-size': str(batch)}.items():
        command[command.index(flag) + 1] = value
    if rotation:
        command.append('--optimize-object-rotation')
    docker = ['docker', 'run', '--rm', '--gpus', 'all', '--user', '1000:1000', '--entrypoint', 'python', '-e', 'PYTHONUNBUFFERED=1', '-e', 'PYTHONPATH=/repo/modules/v2d_cari4d/lib/cari4d:/workspace/v2d_sam3d_body/lib:/workspace/v2d_foundation_pose/lib/FoundationPose', '-e', 'MHR_ASSETS_ROOT=/repo/data/weights/sam3d_body', '-e', 'XDG_CACHE_HOME=/tmp/v2d-cdo-ablation', '-e', 'PYOPENGL_PLATFORM=egl', '-e', 'OPENCV_IO_ENABLE_OPENEXR=1', '-e', 'HDF5_USE_FILE_LOCKING=FALSE', '-e', 'MPLCONFIGDIR=/tmp/matplotlib', '-v', f'{ROOT}:/repo', '-v', f'{ROOT}/modules/v2d_cari4d/lib:/workspace/v2d_cari4d/lib:ro', 'v2d_cari4d:blackwell', '-u', '-c', seed_code] + command
    state = {'state': 'RUNNING', 'episode': episode, 'variant': name, 'seed': 42, 'steps': steps, 'batch_size': batch, 'optimize_rotation': rotation, 'started': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'command': docker, 'results': results, 'source_stage_sha256': hashlib.sha256((source / 'run/.stages/07_refinement.json').read_bytes()).hexdigest(), 'comparison_scope': 'one-factor variants against identically seeded control; repository defaults are not a verified leaderboard-baseline reproduction'}
    (REPORT / 'ablation_status.json').write_text(json.dumps(state, indent=2) + '\n')
    print('ABLATION_START', name, flush=True)
    with (host / 'refinement.log').open('a') as log:
        result = subprocess.run(docker, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        state.update(state='FAILED', exit_code=result.returncode)
        (REPORT / 'ablation_status.json').write_text(json.dumps(state, indent=2) + '\n')
        raise SystemExit(result.returncode)
    results.append({'variant': name, 'state': 'REFINEMENT_COMPLETE_REVIEW_PENDING', 'output': str(host / 'refined.pth')})
    print('ABLATION_COMPLETE', name, flush=True)
state.update(state='REFINEMENT_BATCH_COMPLETE_REVIEW_PENDING', results=results, completed=datetime.datetime.now(datetime.timezone.utc).isoformat())
(REPORT / 'ablation_status.json').write_text(json.dumps(state, indent=2) + '\n')
