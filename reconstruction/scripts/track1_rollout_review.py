"""Read-only cache inventory and multi-frame mask sheets for visual review."""
import argparse
import json
import sys
from pathlib import Path
import cv2
import h5py
import numpy as np
from track1_rollout_common import ORDER, episode_metadata, paths

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--episodes', type=int, nargs='+', default=list(ORDER))
p.add_argument('--corrected',action='store_true')
a = p.parse_args()
metadata = episode_metadata(a.root)
scored = json.loads((a.root/'incident_reports/track1_scored_frames.json').read_text())
inventory = []
for i in a.episodes:
    q = paths(a.root, i)
    q['report_dir'].mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(q['video']))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert total > max(scored[str(i)])
    frames = sorted(set(np.linspace(0, total-1, 6).round().astype(int).tolist() +
                        np.array(scored[str(i)])[np.linspace(0, len(scored[str(i)])-1, 3).round().astype(int)].tolist()))
    panels, records = [], []
    with h5py.File(q['data']/q['stem']/'cari4d_masks.h5', 'r') as f:
        group = f[next(iter(f.keys()))]
        for frame in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, rgb = cap.read()
            assert ok, (i, frame)
            mask = group[f'{frame:06d}-k0.obj_rend_mask.png'][()] > 0
            if a.corrected:
                mask=cv2.imread(str(q['work']/'object_masks/1'/f'{frame:06d}.png'),cv2.IMREAD_GRAYSCALE)
                assert mask is not None, (i,frame)
                mask=mask>0
            overlay = rgb.copy()
            overlay[mask] = (overlay[mask]*.45 + np.array([0,255,0])*.55).astype(np.uint8)
            panel = cv2.resize(overlay, (512,384))
            cv2.putText(panel, f'{i:02d} {metadata[i]["object"]} f{frame}', (8,22), cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
            panels.append(panel)
            yy, xx = np.nonzero(mask)
            records.append({'frame': frame, 'pixels': int(mask.sum()), 'bbox': [int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())] if len(xx) else None})
    cap.release()
    while len(panels) % 3:
        panels.append(np.zeros_like(panels[0]))
    sheet = np.concatenate([np.concatenate(panels[j:j+3], axis=1) for j in range(0,len(panels),3)], axis=0)
    label='corrected' if a.corrected else 'baseline'
    cv2.imwrite(str(q['report_dir']/(q['stem']+'_'+label+'_masks.jpg')), sheet)
    files = {key: {'path': str(q[key]), 'available': q[key].is_file(), 'bytes': q[key].stat().st_size if q[key].is_file() else 0} for key in ('video','depth','init','intrinsics')}
    row = {'episode': i, 'object': metadata[i]['object'], 'frames': total, 'files': files, 'mask_samples': records, 'visual_review': 'PENDING'}
    inventory.append(row)
    (q['report_dir']/(q['stem']+('_corrected_inventory.json' if a.corrected else '_inventory.json'))).write_text(json.dumps(row,indent=2)+'\n')
    print('ROLLOUT_REVIEW_PREPARED', i, total, {k:v['available'] for k,v in files.items()}, flush=True)
(q['report_dir']/('corrected_inventory.json' if a.corrected else 'inventory.json')).write_text(json.dumps(inventory,indent=2)+'\n')
