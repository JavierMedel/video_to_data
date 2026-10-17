"""Isolated v8-parent orientation repair, actual-mesh masks and contact constraints.

Only object_rotation changes. No mesh/depth/translation/human edits or uploads.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import numpy as np
from track1_v9_temporal import sha,write_json,PARENT_SHA
from track1_rotation_selection import orientation_proposals,rotation_path
from track1_contact_smoothing import scored_motion_diagnostics,summarize_selected_contacts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/repo'));p.add_argument('--episodes',type=int,nargs='+',default=[7,18,20]);p.add_argument('--campaign',default='v10_contact_rotation')
    a=p.parse_args();root=a.root.resolve();assert Path(a.campaign).name==a.campaign and a.campaign not in ('.','..')
    data=root/'data/v2d_challenge/track_1';report=root/'incident_reports/track1_improvement'/a.campaign;report.mkdir(parents=True,exist_ok=True)
    lock=(report/'runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    parent=data/'contact_smoothing_v8/submission_episodes';assert sha(data/'submission_packages/track1_contact_smoothing_v8.parquet')==PARENT_SHA
    parent_hashes=json.loads((root/'incident_reports/track1_improvement/contact_smoothing_v8/package_validation.json').read_text())['artifact_sha256']
    assert all(sha(parent/n)==v for n,v in parent_hashes.items())
    os.environ.setdefault('LOGNAME','uid1000');os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','1')
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(data/a.campaign/'runtime_cache/inductor'));os.environ.setdefault('TRITON_CACHE_DIR',str(data/a.campaign/'runtime_cache/triton'))
    sys.path[:0]=['/workspace/v2d_sam3d_body/lib','/workspace/v2d_foundation_pose/lib/FoundationPose',str(root/'modules/v2d_cari4d/lib/cari4d'),str(root/'v2d_submission_kit/v2d_submission_kit/tools/track1')]
    import torch
    import trimesh
    import cv2
    import nvdiffrast.torch as dr
    from scipy.spatial import cKDTree
    from prep.mhr_export_utils import read_mask,camera_calibration,load_edex
    from track1_mask_render import _render_candidate_masks,_scaled_image_grid
    from prep.foundationpose_pose_selection import measure_visible_silhouettes
    from prep.trajectory_diagnostics import rigid_motion_diagnostics
    from mesh_to_mhr_params import MHR
    from lib_mhr.hand_surface_contact import load_mhr_hand_surface_spec
    from lib_mhr.fused_mesh_penetration_review import FusedMeshPenetrationReview
    model_path=root/'data/weights/track1_submission_mhr/mhr_model.pt';spec_path=root/'modules/v2d_cari4d/lib/cari4d/lib_mhr/assets/mhr_hand_surface_spec.npz'
    spec=load_mhr_hand_surface_spec(spec_path);hand_ids=spec.vertex_indices.reshape(-1);fps_ids=(spec.sample_local_indices+np.arange(2)[:,None]*spec.vertex_indices.shape[1]).reshape(-1)
    scored=json.loads((root/'incident_reports/track1_scored_frames.json').read_text());ctx=dr.RasterizeCudaContext()
    for episode in a.episodes:
        assert 0<=episode<30;stem=f'episode_{episode:06d}';out=data/a.campaign/stem;out.mkdir(parents=True,exist_ok=True);rep=report/stem;rep.mkdir(exist_ok=True)
        work=data/('improvement_pilot_v1' if episode in (6,18,28) else 'improvement_rollout_v1')/stem;export=work/'run/export'/stem
        source_files=[parent/(stem+'.npz'),parent/(stem+'_object.glb'),model_path,spec_path,export/'edex',Path(__file__),Path(__file__).with_name('track1_rotation_selection.py'),Path(__file__).with_name('track1_mask_render.py'),Path(__file__).with_name('track1_contact_smoothing.py'),root/'modules/v2d_cari4d/lib/cari4d/lib_mhr/mesh_penetration_review.py',root/'modules/v2d_cari4d/lib/cari4d/lib_mhr/fused_mesh_penetration_review.py',root/'v2d_submission_kit/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py',root/'incident_reports/track1_scored_frames.json']
        inputs={str(x.relative_to(root)):sha(x) for x in source_files};state_path=rep/'status.json'
        if state_path.exists():
            prior=json.loads(state_path.read_text());assert prior['inputs']==inputs,'Preserve old campaign when source/inputs change'
            if prior['state']=='CANDIDATE_READY':
                assert sha(out/(stem+'.npz'))==prior['candidate_npz_sha256'];print('ROTATION_REUSED',episode,flush=True);continue
        state={'episode':episode,'state':'IMAGE_SCREEN','inputs':inputs,'parent':'v8','uploaded':False,'scope':'All-video fitting using predicted masks/actual meshes; same-video diagnostics, not held-out or official scores.'};write_json(state_path,state)
        b=dict(np.load(parent/(stem+'.npz'),allow_pickle=False));count=len(b['pose']);mesh=trimesh.load(parent/(stem+'_object.glb'),force='mesh',process=False);scale=float(b['object_scale']);vertices=np.asarray(mesh.vertices)*scale
        rend=SimpleNamespace(mesh_tensors={'pos':torch.as_tensor(vertices,device='cuda',dtype=torch.float32),'faces':torch.as_tensor(mesh.faces,device='cuda',dtype=torch.int32)},glctx=ctx)
        k,w2c=camera_calibration(load_edex(export),0);assert np.allclose(w2c,np.eye(4))
        screen=out/'image_choices.npz'
        if screen.exists():
            z=dict(np.load(screen,allow_pickle=False));options,ious,eligible=[z[n] for n in ('rotations','ious','eligible')]
        else:
            hypotheses=orientation_proposals(b['object_rotation']);options=np.repeat(b['object_rotation'][:,None],3,axis=1);ious=np.zeros((count,3));eligible=np.zeros((count,3),bool);eligible[:,0]=True
            for frame in range(count):
                name=f'{frame:06d}';obj=read_mask(export,'object',0,name);hm=read_mask(export,'human',0,name);sk,shape=_scaled_image_grid(k,obj.shape,384)
                resize=lambda x:cv2.resize(x.astype(np.uint8),(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)>0
                obj,hm=resize(obj),resize(hm)
                poses=np.tile(np.eye(4),(hypotheses.shape[1],1,1));poses[:,:3,:3]=hypotheses[frame];poses[:,:3,3]=b['object_translation'][frame]
                masks=_render_candidate_masks(rend,poses,sk,shape);metrics=measure_visible_silhouettes(masks,hm,obj);scores=metrics.iou
                ranking=np.argsort(-scores[1:],kind='stable')[:2]+1
                ious[frame,0]=scores[0]
                for choice,index in enumerate(ranking,1):
                    ious[frame,choice]=scores[index];options[frame,choice]=hypotheses[frame,index]
                    eligible[frame,choice]=metrics.observed_visible_pixels>=8 and scores[index]>=scores[0]+.005
                if frame%80==0:print('ROTATION_IMAGE',episode,frame,count,flush=True)
            eligible[:2,1:]=False;eligible[-2:,1:]=False
            tmp=out/'image_choices.tmp.npz';np.savez_compressed(tmp,rotations=options,ious=ious,eligible=eligible);tmp.replace(screen)
        state.update(state='CONTACT_OPTIONS',eligible_nonparent_options=int(eligible[:,1:].sum()),image_choices_sha256=sha(screen));write_json(state_path,state)
        cache=out/'contact_costs.npz';full=np.zeros((count,3));fps=full.copy();gap=full.copy();done=np.zeros((count,3),bool)
        if cache.exists():
            z=dict(np.load(cache,allow_pickle=False));full,fps,gap,done=[z[n] for n in ('full','fps','gap','done')]
        if not done.all():
            model=MHR(str(model_path),'cuda',4,'float32')
            human,_=model.run(torch.as_tensor(b['pose'],device='cuda',dtype=torch.float64),torch.as_tensor(np.r_[b['scales'],b['shape']][None],device='cuda',dtype=torch.float64));hands=(human[:,hand_ids]/1000).double();del human,model
            check=FusedMeshPenetrationReview(mesh.vertices,mesh.faces);tree=cKDTree(mesh.vertices)
            with torch.inference_mode():
                for frame in range(count):
                    for choice in range(3):
                        if done[frame,choice]:continue
                        if choice and not eligible[frame,choice]:
                            full[frame,choice]=full[frame,0];fps[frame,choice]=fps[frame,0];gap[frame,choice]=gap[frame,0];done[frame,choice]=True;continue
                        local=(hands[frame]-torch.as_tensor(b['object_translation'][frame],device='cuda',dtype=torch.float64))@torch.as_tensor(options[frame,choice],device='cuda',dtype=torch.float64)/scale
                        depths=check.depths(local).cpu().numpy()*scale*100
                        full[frame,choice]=depths.mean();fps[frame,choice]=depths[fps_ids].mean();gap[frame,choice]=tree.query(local[fps_ids].cpu().numpy(),workers=4)[0].min()*scale*100;done[frame,choice]=True
                    if frame%8==0 or frame+1==count:
                        tmp=out/'contact_costs.tmp.npz';np.savez_compressed(tmp,full=full,fps=fps,gap=gap,done=done);tmp.replace(cache)
                        state['completed_frame_options']=int(done.sum());write_json(state_path,state)
                    if frame%40==0:print('ROTATION_CONTACT',episode,frame,count,flush=True)
            del hands,check;torch.cuda.empty_cache()
        allowed=eligible & (full<=full[:,:1]+1e-10) & (fps<=fps[:,:1]+1e-10) & ((gap[:,:1]>1.5)|(gap<=gap[:,:1]+.5));allowed[:,0]=True
        path=rotation_path(options,ious,allowed);candidate=dict(b);candidate['object_rotation']=options[np.arange(count),path].copy()
        assert all(np.array_equal(candidate[key],b[key]) for key in b if key!='object_rotation')
        destination=out/(stem+'.npz')
        if destination.exists():
            saved=dict(np.load(destination,allow_pickle=False));assert all(np.array_equal(saved[key],candidate[key]) for key in candidate)
        else:np.savez_compressed(destination,**candidate)
        target=out/(stem+'_object.glb')
        if not target.exists():shutil.copy2(parent/target.name,target)
        assert sha(target)==parent_hashes[target.name]
        np.savez_compressed(out/'selection.npz',path=path,allowed=allowed)
        motions={};scored_motion={}
        for label,bundle in [('submitted_v8',b),('final_candidate',candidate)]:
            pose=np.tile(np.eye(4),(count,1,1));pose[:,:3,:3]=bundle['object_rotation'];pose[:,:3,3]=bundle['object_translation'];motions[label]=rigid_motion_diagnostics(pose,vertices)
            scored_motion[label]=scored_motion_diagnostics(bundle['object_rotation'],bundle['object_translation'],vertices,np.array(scored[str(episode)]))
        write_json(out/'review.json',{'control_label':'submitted_v8','held_out_frames':[],'geometry_review':{'held_out':{'depth_driven':False}},'summary':motions,'scope':state['scope']})
        contacts=summarize_selected_contacts(full,fps,gap,done,path,np.array(scored[str(episode)]));contacts.update(candidate_npz_sha256=sha(destination),control_npz_sha256=parent_hashes[stem+'.npz'],candidate_mesh_sha256=sha(target),control_mesh_sha256=sha(target),all_scored_frames=True,frames=scored[str(episode)],scope='Reused exact finite-option full/FPS mesh checks; not an independent hand rerender or official PEN.')
        write_json(out/'fit_hand_review.json',contacts)
        state.update(state='CANDIDATE_READY',selected_nonparent_frames=int((path!=0).sum()),changed_scored_frames=int((path[np.array(scored[str(episode)])]!=0).sum()),feasible_nonparent_options=int(allowed[:,1:].sum()),candidate_npz_sha256=sha(destination),candidate_mesh_sha256=sha(target),contact_costs_sha256=sha(cache),summary=motions,scored_motion=scored_motion,framewise_contact_pass=True)
        write_json(state_path,state);print('ROTATION_READY',episode,state['selected_nonparent_frames'],flush=True)
    assert all(sha(parent/n)==v for n,v in parent_hashes.items())


if __name__=='__main__':main()
