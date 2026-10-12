"""Create segmentation review sheets and depth-grounded mesh inputs."""
import json,sys
from pathlib import Path
import cv2
import numpy as np
root=Path(sys.argv[1]); i=int(sys.argv[2]); stem=f'episode_{i:06d}'
sys.path.insert(0,str(root/'modules/v2d_cari4d/lib/cari4d'))
from prep.mhr_export_utils import read_depth_m
data=root/'data/v2d_challenge/track_1'; work=data/'improvement_pilot_v1'/stem
video=data/'videos/chunk-000/observation.images.exo_camera'/(stem+'.mp4')
cap=cv2.VideoCapture(str(video)); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
files=list((work/'object_masks/1').glob('*.png')); assert len(files)==total
export=data/'cari4d_runs/track1_resume'/stem/stem/'export'/stem
meta=json.loads((export/'wild_export.json').read_text()); k=np.asarray(meta['intrinsics'])
panels=[]; records=[]
for frame in np.linspace(0,total-1,9).round().astype(int):
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame)); ok,rgb=cap.read(); assert ok
    mask=cv2.imread(str(work/'object_masks/1'/f'{frame:06d}.png'),0)>0
    yy,xx=np.nonzero(mask)
    overlay=rgb.copy(); overlay[mask]=(overlay[mask]*0.45+np.array([0,255,0])*0.55).astype(np.uint8)
    cv2.putText(overlay,f'{stem} frame {frame}: {mask.sum()} pixels',(20,35),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    panels.append(cv2.resize(overlay,(512,384)))
    records.append({'frame':int(frame),'pixels':int(mask.sum()),'bbox':None if not len(xx) else [int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())]})
sheet=np.concatenate([np.concatenate(panels[j:j+3],axis=1) for j in range(0,9,3)],axis=0)
cv2.imwrite(str(root/'incident_reports/track1_improvement'/(stem+'_corrected_masks.jpg')),sheet)
(work/'mask_review.json').write_text(json.dumps(records,indent=2))
reference={6:205,18:0,28:0}[i]
cap.set(cv2.CAP_PROP_POS_FRAMES,reference); ok,rgb=cap.read(); cap.release(); assert ok
mesh_inputs=work/'mesh_inputs'; mesh_inputs.mkdir(exist_ok=True)
cv2.imwrite(str(mesh_inputs/'rgb.png'),rgb)
mask=cv2.imread(str(work/'object_masks/1'/f'{reference:06d}.png'),0)
cv2.imwrite(str(mesh_inputs/'mask.png'),mask)
depth=read_depth_m(export,0,f'{reference:06d}',depth_root=export.parents[1]/'depth/moge2/aligned.h5')
yy,xx=np.meshgrid(np.arange(depth.shape[0]),np.arange(depth.shape[1]),indexing='ij')
points=np.stack([(xx-k[0,2])*depth/k[0,0],(yy-k[1,2])*depth/k[1,1],depth],axis=-1).astype(np.float32)
np.save(mesh_inputs/'pointmap.npy',points)
(mesh_inputs/'intrinsics.json').write_text(json.dumps({'fx':float(k[0,0]),'fy':float(k[1,1]),'cx':float(k[0,2]),'cy':float(k[1,2]),'width':int(meta['width']),'height':int(meta['height'])}))
print('Masks checked; mesh inputs prepared',stem,flush=True)
