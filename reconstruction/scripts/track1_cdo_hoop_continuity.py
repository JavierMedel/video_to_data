"""Score continuous-axis hoop candidates using actual mesh silhouettes."""
import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
import trimesh

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, default=Path('/repo'))
p.add_argument('--episodes', type=int, nargs='+', default=[2, 1])
a = p.parse_args()
sys.path.insert(0, '/workspace/v2d_foundation_pose/lib/FoundationPose')
sys.path.insert(0, str(a.root / 'modules/v2d_cari4d/lib/cari4d'))
from prep.temporal_pose_selection import circular_hoop_rotations
from prep.trajectory_diagnostics import rigid_motion_diagnostics
from prep.mhr_export_utils import read_mask, read_rgb, camera_calibration, load_edex
from prep.run_foundationpose_mhr_export import _render_candidate_masks, _scaled_image_grid
from prep.foundationpose_pose_selection import measure_visible_silhouettes

data = a.root / 'data/v2d_challenge/track_1'
scored = json.loads((a.root / 'incident_reports/track1_scored_frames.json').read_text())
metadata = {r['episode_index']: r for r in map(json.loads, (data / 'meta/episodes_metadata.jsonl').read_text().splitlines())}
with torch.inference_mode():
    for episode in a.episodes:
        assert metadata[episode]['object'] == 'hula_hoop', 'Do not apply hoop symmetry to handles'
        stem = f'episode_{episode:06d}'
        parent = data / 'cdo_v4_experiments/pose_candidates' / stem
        out = data / 'cdo_v4_experiments/hoop_candidates' / stem
        assert not (out / (stem + '.npz')).exists(), 'Preserve existing hoop candidate'
        out.mkdir(parents=True, exist_ok=True)
        bundle = dict(np.load(parent / (stem + '.npz'), allow_pickle=False))
        mesh = trimesh.load(parent / (stem + '_object.glb'), force='mesh', process=False)
        assert mesh.extents[0] < .15 * min(mesh.extents[1:]) and abs(mesh.extents[1] / mesh.extents[2] - 1.) < .05
        vertices = mesh.vertices * float(bundle['object_scale'])
        renderer = SimpleNamespace(mesh_tensors={'pos': torch.as_tensor(vertices, device='cuda', dtype=torch.float32), 'faces': torch.as_tensor(mesh.faces, device='cuda', dtype=torch.int32)}, glctx=dr.RasterizeCudaContext())
        poses = np.tile(np.eye(4), (len(bundle['pose']), 1, 1))
        poses[:, :3, :3] = bundle['object_rotation']; poses[:, :3, 3] = bundle['object_translation']
        changed = poses.copy(); changed[:, :3, :3] = circular_hoop_rotations(poses[:, :3, :3])
        export = data / 'improvement_rollout_v1' / stem / 'run/export' / stem
        k, w2c = camera_calibration(load_edex(export), 0)
        frames = np.array(scored[str(episode)])
        frames = frames[np.linspace(0, len(frames)-1, min(33, len(frames))).round().astype(int)]
        summaries = {}; panels = []
        for label, sequence in [('parent', poses), ('hoop_continuity', changed)]:
            ious = []
            for frame in frames:
                human = read_mask(export, 'human', 0, f'{frame:06d}'); obj = read_mask(export, 'object', 0, f'{frame:06d}')
                small_k, shape = _scaled_image_grid(k, obj.shape, 768)
                resize = lambda x: cv2.resize(x.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                human, obj = resize(human), resize(obj)
                mask = _render_candidate_masks(renderer, w2c[None] @ sequence[frame:frame+1], small_k, shape)
                ious.append(float(measure_visible_silhouettes(mask, human, obj).iou[0]))
                if frame in frames[np.linspace(0, len(frames)-1, 6).round().astype(int)]:
                    rgb = cv2.resize(read_rgb(export, 0, f'{frame:06d}'), (shape[1], shape[0]))
                    rgb[obj] = (rgb[obj]*.6 + np.array([0, 255, 0])*.4).astype(np.uint8)
                    contours, _ = cv2.findContours(mask[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(rgb, contours, -1, (255, 60, 60), 2)
                    rgb = cv2.resize(rgb, (512, 384))
                    cv2.putText(rgb, f'{label} f{frame} IoU {ious[-1]:.3f}', (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                    panels.append(rgb)
            summaries[label] = {'visible_iou_mean': float(np.mean(ious)), 'translation_acceleration_cm_proxy': float(np.linalg.norm(np.diff(sequence[:, :3, 3], n=2, axis=0), axis=1).mean()*100), **rigid_motion_diagnostics(sequence, vertices)}
        accepted = summaries['hoop_continuity']['visible_iou_mean'] >= summaries['parent']['visible_iou_mean'] - .02
        review = json.loads((parent / 'review.json').read_text())
        review.update(hoop_symmetry_review={'accepted': bool(accepted), 'summary': summaries, 'scope': 'Circular hoop only; actual mesh scored; no handle symmetry assumptions'})
        if accepted:
            bundle['object_rotation'] = changed[:, :3, :3].astype(np.float32)
            review['summary']['final_candidate'] = summaries['hoop_continuity']
        np.savez_compressed(out / (stem + '.npz'), **bundle)
        shutil.copy2(parent / (stem + '_object.glb'), out / (stem + '_object.glb'))
        review['candidate_npz'] = str(out / (stem + '.npz'))
        (out / 'review.json').write_text(json.dumps(review, indent=2) + '\n')
        cv2.imwrite(str(out / 'comparison.jpg'), cv2.cvtColor(np.concatenate([np.concatenate(panels[:6], axis=1), np.concatenate(panels[6:], axis=1)], axis=0), cv2.COLOR_RGB2BGR))
        print('HOOP_CONTINUITY_COMPLETE', episode, json.dumps(summaries), 'accepted', bool(accepted), flush=True)
