"""Expose thin-part coverage in original masks with labelled source crops."""
import argparse
import json
from pathlib import Path
import cv2
import h5py
import numpy as np
from track1_rollout_common import paths

p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--episodes',type=int,nargs='+',required=True)
p.add_argument('--corrected',action='store_true'); a=p.parse_args()
for i in a.episodes:
    q=paths(a.root,i); q['work'].mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(q['video'])); ok,rgb=cap.read(); cap.release(); assert ok
    with h5py.File(q['data']/q['stem']/'cari4d_masks.h5','r') as f:
        group=f[next(iter(f.keys()))]; mask=group['000000-k0.obj_rend_mask.png'][()]>0
    if a.corrected: mask=cv2.imread(str(q['work']/'object_masks/1/000000.png'),cv2.IMREAD_GRAYSCALE)>0
    yy,xx=np.nonzero(mask); assert len(xx)
    x0=max(0,int(xx.min())-90); x1=min(rgb.shape[1],int(xx.max())+91)
    y0=max(0,int(yy.min())-75); y1=min(rgb.shape[0],int(yy.max())+76)
    crop=rgb[y0:y1,x0:x1]; visible=mask[y0:y1,x0:x1]
    overlay=crop.copy(); overlay[visible]=(overlay[visible]*.45+np.array([0,255,0])*.55).astype(np.uint8)
    sheet=cv2.resize(np.concatenate([crop,overlay],axis=1),None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST)
    label='corrected' if a.corrected else 'original'
    cv2.putText(sheet,f'ep {i} crop origin ({x0},{y0}), scale 3; raw / {label} mask',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
    cv2.imwrite(str(q['report_dir']/(q['stem']+('_corrected_closeup.jpg' if a.corrected else '_reference_closeup.jpg'))),sheet)
    reference=q['work']/'reference'; reference.mkdir(exist_ok=True); cv2.imwrite(str(reference/'frame_000000.png'),rgb)
    (reference/('corrected_crop.json' if a.corrected else 'crop.json')).write_text(json.dumps({'crop':[x0,y0,x1,y1],'mask_bbox':[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())]},indent=2))
    print('ROLLOUT_REFERENCE_CLOSEUP',i,[x0,y0,x1,y1],flush=True)
