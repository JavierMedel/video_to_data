"""Validate actual submitted object masks on every scored frame."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
import trimesh

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/repo'));p.add_argument('--episodes',type=int,nargs='+',required=True);a=p.parse_args()
sys.path.insert(0,'/workspace/v2d_foundation_pose/lib/FoundationPose')
sys.path.insert(0,str(a.root/'modules/v2d_cari4d/lib/cari4d'))
from prep.run_foundationpose_mhr_export import _render_candidate_masks, _scaled_image_grid
from prep.mhr_export_utils import read_mask, read_depth_m, camera_calibration, load_edex
from prep.foundationpose_pose_selection import measure_visible_silhouettes
from track1_rollout_common import paths
import Utils
data=a.root/'data/v2d_challenge/track_1';report=a.root/'incident_reports/track1_improvement/cdo_v7'
scored=json.loads((a.root/'incident_reports/track1_scored_frames.json').read_text());ctx=dr.RasterizeCudaContext()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with torch.inference_mode():
    for episode in a.episodes:
        stem=f'episode_{episode:06d}';control=data/'cdo_v6_experiments/candidate_episodes'
        candidate=data/'cdo_v7_experiments/geometry_candidates'/stem
        rescue=data/'cdo_v7_experiments/hand_contact_candidates_v2'/stem
        if (rescue/'review.json').exists():candidate=rescue
        out=candidate/'all_scored_image_review.json'
        fingerprints={f'{label}_{kind}_sha256':sha(folder/(stem+suffix)) for label,folder in [('control',control),('candidate',candidate)] for kind,suffix in [('npz','.npz'),('mesh','_object.glb')]}
        if out.exists():
            existing=json.loads(out.read_text());assert all(existing[k]==v for k,v in fingerprints.items())
            print('FULL_IMAGE_REVIEW_REUSED',episode,flush=True);continue
        work=data/('improvement_pilot_v1' if episode in (6,18,28) else 'improvement_rollout_v1')/stem
        export=work/'run/export'/stem;k,w2c=camera_calibration(load_edex(export),0);assert np.allclose(w2c,np.eye(4))
        frames=np.array(scored[str(episode)]);versions={};depth_rows=[]
        depth_root=paths(a.root,episode)['depth']
        review=json.loads((candidate/'review.json').read_text())

        def depth_render(renderer,pose,k,shape):
            tensors=renderer.mesh_tensors
            matrix=torch.as_tensor(pose,device='cuda',dtype=torch.float32)
            camera=tensors['pos']@matrix[:3,:3].T+matrix[:3,3]
            projection=torch.as_tensor(Utils.projection_matrix_from_intrinsics(k,height=shape[0],width=shape[1],znear=.001,zfar=100.),device='cuda',dtype=torch.float32)
            cv_to_gl=torch.as_tensor(Utils.glcam_in_cvcam,device='cuda',dtype=torch.float32)
            clip=Utils.to_homo_torch(camera)@(projection@cv_to_gl).T
            raster,_=dr.rasterize(ctx,clip[None],tensors['faces'],resolution=shape)
            z,_=dr.interpolate(camera[None,:,2:3].contiguous(),raster,tensors['faces'])
            return torch.flip(z[0,:,:,0],[0]).cpu().numpy(),torch.flip(raster[0,:,:,3]>0,[0]).cpu().numpy()
        for label,folder in [('control',control),('candidate',candidate)]:
            bundle=dict(np.load(folder/(stem+'.npz'),allow_pickle=False));mesh=trimesh.load(folder/(stem+'_object.glb'),force='mesh',process=False)
            renderer=SimpleNamespace(mesh_tensors={'pos':torch.as_tensor(mesh.vertices*float(bundle['object_scale']),device='cuda',dtype=torch.float32),'faces':torch.as_tensor(mesh.faces,device='cuda',dtype=torch.int32)},glctx=ctx)
            versions[label]=(bundle,renderer,[])
        for i,frame in enumerate(frames):
            obj=read_mask(export,'object',0,f'{frame:06d}');human=read_mask(export,'human',0,f'{frame:06d}')
            small_k,shape=_scaled_image_grid(k,obj.shape,512)
            resize=lambda x:cv2.resize(x.astype(np.uint8),(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)>0
            obj,human=resize(obj),resize(human);renders={}
            for label,(bundle,renderer,ious) in versions.items():
                pose=np.eye(4);pose[:3,:3]=bundle['object_rotation'][frame];pose[:3,3]=bundle['object_translation'][frame]
                z,mask=depth_render(renderer,pose,small_k,shape);renders[label]=(z,mask)
                ious.append(float(measure_visible_silhouettes(mask[None],human,obj).iou[0]))
            observed=cv2.resize(read_depth_m(export,0,f'{frame:06d}',depth_root=depth_root),(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)
            bz,bm=renders['control'];nz,nm=renders['candidate']
            support=bm & nm & obj & ~human & np.isfinite(observed) & (observed>.05) & (bz>.05) & (nz>.05)
            if support.sum()>=8:
                depth_rows.append({'frame':int(frame),'before_cm':float(np.median(abs(bz[support]-observed[support]))*100),'after_cm':float(np.median(abs(nz[support]-observed[support]))*100)})
            if i%80==0:print('FULL_IMAGE_REVIEW_PROGRESS',episode,i,len(frames),flush=True)
        old,new=[np.array(versions[label][2]) for label in ['control','candidate']]
        passed=new.mean()>=max(.3,old.mean()-.005) and float(np.mean(new<old-.05))<=.1
        held_depth=[row for row in depth_rows if row['frame'] in review['held_out_frames']]
        depth_summary={'supported_frames':len(depth_rows),'before_cm':float(np.median([r['before_cm'] for r in depth_rows])) if depth_rows else None,'after_cm':float(np.median([r['after_cm'] for r in depth_rows])) if depth_rows else None}
        held_summary={'supported_frames':len(held_depth),'before_cm':float(np.median([r['before_cm'] for r in held_depth])) if held_depth else None,'after_cm':float(np.median([r['after_cm'] for r in held_depth])) if held_depth else None}
        depth_driven=review['geometry_review']['held_out']['depth_driven']
        depth_pass=(len(depth_rows)<4 or depth_summary['after_cm']<=depth_summary['before_cm']+.5) and (not depth_driven or (len(held_depth)>=4 and held_summary['after_cm']<=held_summary['before_cm']-.5))
        result={'episode':episode,'all_scored_frames':True,'frames':frames.tolist(),'image_gate_pass':bool(passed),
                'depth_gate_pass':bool(depth_pass),'common_depth':depth_summary,'held_out_common_depth':held_summary,
                'summary':{'control':{'visible_iou_mean':float(old.mean())},'candidate':{'visible_iou_mean':float(new.mean())}},
                'fraction_frames_iou_drop_over_005':float(np.mean(new<old-.05)),'p10_iou_delta':float(np.quantile(new-old,.1)),
                'control_ious':old.tolist(),'candidate_ious':new.tolist(),**fingerprints,
                'scope':'Every scored frame; identical actual-mesh silhouette metric for control and candidate. Image alignment is a proxy, not official CD-O.'}
        out.write_text(json.dumps(result,indent=2)+'\n');(report/(stem+'_all_scored_image_review.json')).write_text(json.dumps(result,indent=2)+'\n')
        print('FULL_IMAGE_REVIEW_COMPLETE',episode,'PASS' if passed else 'REJECT',float(old.mean()),float(new.mean()),flush=True)
