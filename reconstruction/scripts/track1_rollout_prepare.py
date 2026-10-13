"""Prepare isolated inputs after explicit multi-frame visual mask review."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import cv2
import h5py
import numpy as np
from track1_rollout_common import episode_metadata, paths

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--episodes',type=int,nargs='+',required=True)
p.add_argument('--mesh-episode','--pilot-mesh-episode',dest='mesh_episode',type=int,required=True)
p.add_argument('--mesh-source',type=Path)
p.add_argument('--review-note',required=True)
p.add_argument('--corrected-masks',action='store_true')
a=p.parse_args()
metadata=episode_metadata(a.root)
for i in a.episodes:
    q=paths(a.root,i); work=q['work']; work.mkdir(parents=True,exist_ok=True)
    assert metadata[i]['object']==metadata[a.mesh_episode]['object'], 'Mesh sharing is restricted to the same physical object group'
    assert all(q[k].is_file() for k in ('video','depth','init','intrinsics'))
    target=work/'cari4d_masks.h5'
    source=q['data']/q['stem']/'cari4d_masks.h5'
    if target.exists(): raise FileExistsError(target)
    if a.corrected_masks:
        with h5py.File(source,'r') as original, h5py.File(target,'w') as new:
            for key in original.attrs: new.attrs[key]=original.attrs[key]
            for key in original: original.copy(key,new)
            group=new[next(iter(new.keys()))]
            mask_files=sorted((work/'object_masks/1').glob('*.png'))
            cap=cv2.VideoCapture(str(q['video'])); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
            assert len(mask_files)==count
            for frame in range(count):
                mask=cv2.imread(str(work/'object_masks/1'/f'{frame:06d}.png'),cv2.IMREAD_GRAYSCALE)
                assert mask is not None
                key=f'{frame:06d}-k0.obj_rend_mask.png'
                assert group[key].shape==mask.shape
                foreground=max(1,int(np.max(group[key][()])))
                group[key][...]=(mask>0).astype(group[key].dtype)*foreground
    else:
        shutil.copy2(source,target)
    with h5py.File(target,'r+') as f:
        groups=list(f.keys())
        assert len(groups)==1, 'Expected one packed-mask sequence'
        if groups[0]!=q['stem']: f.move(groups[0],q['stem'])
    with h5py.File(target,'r') as f:
        group=f[next(iter(f.keys()))]
        keys=sorted(k for k in group if k.endswith('-k0.obj_rend_mask.png'))
        cap=cv2.VideoCapture(str(q['video'])); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        assert keys==[f'{frame:06d}-k0.obj_rend_mask.png' for frame in range(count)]
    mesh_source=a.mesh_source or q['data']/'improvement_pilot_v1'/f'episode_{a.mesh_episode:06d}'/'mesh/object_metric.glb'
    mesh=work/'mesh'; mesh.mkdir(exist_ok=True)
    mesh_target=mesh/'object_metric.glb'
    if mesh_source.resolve()!=mesh_target.resolve():
        if mesh_target.exists():
            assert hashlib.sha256(mesh_target.read_bytes()).hexdigest()==hashlib.sha256(mesh_source.read_bytes()).hexdigest(), 'Preserve differing existing mesh'
        else: shutil.copy2(mesh_source,mesh_target)
    review={'verdict':'PASS','episode':i,'object':metadata[i]['object'],'note':a.review_note,'mask_source':'corrected_sam2' if a.corrected_masks else 'visually_reviewed_baseline','frames':count,'mask_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'mesh_source':str(mesh_source),'mesh_sha256':hashlib.sha256(mesh_source.read_bytes()).hexdigest(),'ground_truth_used':False}
    (work/'review_approval.json').write_text(json.dumps(review,indent=2)+'\n')
    print('ROLLOUT_INPUTS_PREPARED',i,count,flush=True)
