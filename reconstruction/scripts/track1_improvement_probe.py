"""Estimate pilot mesh scale from image masks and human-aligned depth."""
import json,sys
from pathlib import Path
import numpy as np
import cv2
import torch
import trimesh
root=Path(sys.argv[1]); data=root/'data/v2d_challenge/track_1'
sys.path.insert(0,str(root/'modules/v2d_cari4d/lib/cari4d'))
from prep.mhr_export_utils import read_depth_m,read_mask,read_rgb
out=root/'incident_reports/track1_improvement'; out.mkdir(exist_ok=True)
results=[]
for i in (6,18,28):
    stem=f'episode_{i:06d}'; run=data/'cari4d_runs/track1_resume'/stem/stem
    export=run/'export'/stem
    meta=json.loads((export/'wild_export.json').read_text()); k=np.array(meta['intrinsics'])
    b=torch.load(run/'inference/refined.pth',map_location='cpu',weights_only=False)
    mesh=trimesh.load(export/'object_mesh/output_aligned.glb',force='mesh')
    v=np.asarray(mesh.vertices)[::max(1,len(mesh.vertices)//3000)]
    frames=json.loads((root/'incident_reports/track1_scored_frames.json').read_text())[str(i)]
    selected=np.array(frames)[np.linspace(0,len(frames)-1,12).round().astype(int)]
    rows=[]; panels=[]
    for frame in selected:
        name=f'{frame:06d}'; mask=read_mask(export,'object',0,name); human=read_mask(export,'human',0,name)
        depth=read_depth_m(export,0,name,depth_root=run/'depth/moge2/aligned.h5')
        visible=mask&~human&(depth>0.05)&np.isfinite(depth)
        visible=cv2.erode(visible.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        if visible.sum()<30: continue
        yy,xx=np.nonzero(mask); z=float(np.median(depth[visible])); center=np.array([np.median(xx),np.median(yy)])
        translation=np.r_[(center-k[:2,2])*z/np.diag(k)[:2],z]
        rotation=np.asarray(b['in']['pose_abs'])[frame,:3,:3]
        pts=v@rotation.T+translation
        uv=pts@k.T; uv=uv[:,:2]/uv[:,2:]
        observed=np.array([xx.min(),yy.min(),xx.max(),yy.max()])
        projected=np.r_[uv.min(0),uv.max(0)]
        ratio=(observed[2:]-observed[:2])/np.maximum(1,projected[2:]-projected[:2])
        row={'frame':int(frame),'depth_m':z,'translation_from_mask_depth':translation.tolist(),'scale_ratio_xy':ratio.tolist(),'visible_fraction':float(visible.sum()/mask.sum())}
        rows.append(row)
        if len(panels)<3:
            rgb=read_rgb(export,0,name).copy()
            cv2.rectangle(rgb,tuple(observed[:2].astype(int)),tuple(observed[2:].astype(int)),(0,255,0),3)
            cv2.rectangle(rgb,tuple(projected[:2].astype(int)),tuple(projected[2:].astype(int)),(255,0,0),3)
            cv2.putText(rgb,f'{stem} frame {frame} depth {z:.2f} m ratios {ratio[0]:.2f},{ratio[1]:.2f}',(20,35),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            panels.append(cv2.resize(rgb,(768,576)))
    result={'episode':i,'source_extents_m':mesh.extents.tolist(),'samples':rows,'suggested_scale_ratio':float(np.median([np.sqrt(np.prod(r['scale_ratio_xy'])) for r in rows]))}
    results.append(result)
    cv2.imwrite(str(out/(stem+'_probe.jpg')),cv2.cvtColor(np.concatenate(panels,axis=1),cv2.COLOR_RGB2BGR))
    print(json.dumps(result),flush=True)
(out/'calibration_probe.json').write_text(json.dumps(results,indent=2))
