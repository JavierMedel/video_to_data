"""Locate rotation/depth failures by stage without changing saved predictions."""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, default=Path('/repo'))
p.add_argument('--episodes', type=int, nargs='+', default=[1, 2, 12, 14, 19, 20])
a = p.parse_args()
sys.path.insert(0, str(a.root / 'modules/v2d_cari4d/lib/cari4d'))
from prep.trajectory_diagnostics import rigid_motion_diagnostics

data = a.root / 'data/v2d_challenge/track_1'
report = a.root / 'incident_reports/track1_improvement/cdo_v4'
rows = []
for episode in a.episodes:
    stem = f'episode_{episode:06d}'
    work = data / 'improvement_rollout_v1' / stem / 'run'
    mesh = trimesh.load(work / 'export' / stem / 'object_mesh/output_aligned.glb', force='mesh', process=False)
    tracker = pickle.loads((work / 'foundationpose' / (stem + '.pkl')).read_bytes())
    stages = {'tracker': np.asarray(tracker['obj_pose_world'])}
    for name in ('coconet', 'refined'):
        bundle = torch.load(work / 'inference' / (name + '.pth'), map_location='cpu', weights_only=False)
        stages[name] = np.asarray(bundle['pr'].get('pose_abs_postopt', bundle['pr']['pose_abs']))
    details = {}
    for name, poses in stages.items():
        rot = poses[:, :3, :3]
        details[name] = {**rigid_motion_diagnostics(poses, mesh.vertices),
                        'orthogonality_error_max': float(abs(rot @ rot.transpose(0, 2, 1) - np.eye(3)).max()),
                        'determinant_min': float(np.linalg.det(rot).min()),
                        'center_depth_median_m': float(np.median(poses[:, 2, 3]))}
    rows.append({'episode': episode, 'frames': len(poses), 'mesh_extents_m': mesh.extents.tolist(), 'stages': details})
    print(json.dumps(rows[-1]), flush=True)
(report / 'pose_stage_audit.json').write_text(json.dumps(rows, indent=2) + '\n')
