"""Convert preserved CARI4D geometry with the official Track 1 converter."""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--episode', type=int, required=True)
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--out-dir', type=Path)
parser.add_argument('--progress-report', type=Path)
args = parser.parse_args()
if args.episode == 16:
    raise SystemExit('Episode 16 protected: conversion refused')
root = args.root
sys.path.insert(0, str(root/'modules/v2d_cari4d/lib/cari4d'))
sys.path.insert(0, '/workspace/v2d_sam3d_body/lib')
sys.path.insert(0, str(root/'v2d_submission_kit/v2d_submission_kit/tools/track1'))
os.environ['MHR_ASSETS_ROOT'] = str(root/'data/weights/sam3d_body')
from lib_mhr import MHRLayer, MHR_PARAM_DIMS
from mesh_to_mhr_params import convert

stem = f'episode_{args.episode:06d}'
out = args.out_dir or root/'data/v2d_challenge/track_1/submission_episodes'
target = out/(stem+'.npz'); mesh_target = out/(stem+'_object.glb')
if target.exists() or mesh_target.exists():
    raise SystemExit('Refusing to overwrite existing submission artifacts')
bundle = torch.load(args.run/'inference/refined.pth', map_location='cpu', weights_only=False)
frames = np.array([int(f) for f in bundle['frames']])
assert np.array_equal(frames, np.arange(len(frames))), 'Expected every video frame in order'
layer = MHRLayer.from_mhr_assets(mhr_assets_root=os.environ['MHR_ASSETS_ROOT'], device='cuda')
vertices = []
with torch.no_grad():
    for start in range(0, len(frames), 16):
        params = {k:torch.as_tensor(bundle['pr'][k][start:start+16], device='cuda', dtype=torch.float32) for k in MHR_PARAM_DIMS}
        vertices.append(layer.mhr_forward_vertices(params).cpu().numpy())
verts = np.concatenate(vertices)
del layer
torch.cuda.empty_cache()
converted = convert(verts, str(root/'data/weights/track1_submission_mhr/mhr_model.pt'), device='cuda', precision='float32', model_batch=128)
converted['report'] = json.dumps(converted['report'], sort_keys=True)
pose = np.asarray(bundle['pr'].get('pose_abs_postopt', bundle['pr']['pose_abs']))
converted.update(object_rotation=pose[:,:3,:3], object_translation=pose[:,:3,3], object_scale=np.array(1.0,dtype=np.float32))
assert all(np.isfinite(converted[k]).all() for k in ['pose','scales','shape','object_rotation','object_translation','object_scale'])
mesh_source = next((args.run/'export').glob('*/object_mesh/output_aligned.glb'))
out.mkdir(exist_ok=True)
tmp = target.with_suffix('.tmp.npz')
np.savez_compressed(tmp, **converted)
tmp.replace(target)
import shutil
shutil.copy2(mesh_source, mesh_target)
with (args.progress_report or root/'incident_reports/track1_submission_progress.md').open('a') as log:
    log.write(f'\nConverted {stem} from preserved refined output: {len(frames)} frames. Official converter report: {converted["report"]}\n')
print(f'CONVERSION_COMPLETE {stem} {target}', flush=True)
