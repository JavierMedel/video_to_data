"""Prepare depth-grounded mesh inputs from visually reviewed video masks."""
import argparse
import json
import sys
from pathlib import Path
import cv2
import h5py
import numpy as np
from track1_rollout_common import paths

p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
p.add_argument('--episode',type=int,required=True); p.add_argument('--frame',type=int,default=0)
p.add_argument('--extract-only',action='store_true'); a=p.parse_args()
q=paths(a.root,a.episode); work=q['work']; work.mkdir(parents=True,exist_ok=True)
cap=cv2.VideoCapture(str(q['video'])); cap.set(cv2.CAP_PROP_POS_FRAMES,a.frame); ok,rgb=cap.read(); cap.release(); assert ok
inputs=work/'mesh_inputs'; inputs.mkdir(exist_ok=True)
cv2.imwrite(str(inputs/'rgb.png'),rgb)
if a.extract_only:
    print(str(inputs/'rgb.png')); raise SystemExit(0)
sys.path.insert(0,str(a.root/'modules/v2d_cari4d/lib/cari4d'))
from prep.mhr_export_utils import read_depth_m
with h5py.File(q['data']/q['stem']/'cari4d_masks.h5','r') as f:
    group=f[next(iter(f.keys()))]
    mask=group[f'{a.frame:06d}-k0.obj_rend_mask.png'][()]>0
    human=group[f'{a.frame:06d}-k0.person_mask.png'][()]>0
corrected=work/'object_masks/1'/f'{a.frame:06d}.png'
if corrected.exists(): mask=cv2.imread(str(corrected),cv2.IMREAD_GRAYSCALE)>0
depth=read_depth_m(q['baseline_export'],0,f'{a.frame:06d}',depth_root=q['depth'])
meta=json.loads((q['baseline_export']/'wild_export.json').read_text()); k=np.asarray(meta['intrinsics'])
visible=mask&~human&np.isfinite(depth)&(depth>0.05)
assert visible.sum()>=32, 'Select a frame with enough visible object depth'
yy,xx=np.meshgrid(np.arange(depth.shape[0]),np.arange(depth.shape[1]),indexing='ij')
points=np.stack([(xx-k[0,2])*depth/k[0,0],(yy-k[1,2])*depth/k[1,1],depth],axis=-1).astype(np.float32)
np.save(inputs/'pointmap.npy',points)
cv2.imwrite(str(inputs/'mask.png'),mask.astype(np.uint8)*255)
(inputs/'intrinsics.json').write_text(json.dumps({'fx':float(k[0,0]),'fy':float(k[1,1]),'cx':float(k[0,2]),'cy':float(k[1,2]),'width':rgb.shape[1],'height':rgb.shape[0]},indent=2))
(inputs/'review.json').write_text(json.dumps({'episode':a.episode,'frame':a.frame,'visible_pixels':int(visible.sum()),'observed_median_depth_m':float(np.median(depth[visible])),'ground_truth_used':False},indent=2))
print('ROLLOUT_MESH_INPUTS_READY',a.episode,a.frame,flush=True)
