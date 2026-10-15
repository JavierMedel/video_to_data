"""Audit v6 and fit bounded object-only refinements for all Track 1 episodes.

Selection uses predicted masks and cached monocular depth, never hidden GT.
Every fitted proposal still needs independent hand and full-frame review.
"""
import argparse
import hashlib
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
from scipy.signal import savgol_filter


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/repo'))
    p.add_argument('--episodes', type=int, nargs='+', default=list(range(30)))
    p.add_argument('--audit-only', action='store_true')
    a = p.parse_args()
    sys.path.insert(0, '/workspace/v2d_foundation_pose/lib/FoundationPose')
    sys.path.insert(0, str(a.root/'modules/v2d_cari4d/lib/cari4d'))
    from track1_rollout_common import paths, episode_metadata
    from prep.mhr_export_utils import read_depth_m, read_mask, camera_calibration, load_edex
    from prep.run_foundationpose_mhr_export import _scaled_image_grid
    from prep.foundationpose_pose_selection import measure_visible_silhouettes
    from prep.trajectory_diagnostics import rigid_motion_diagnostics
    import Utils

    data = a.root/'data/v2d_challenge/track_1'
    report = a.root/'incident_reports/track1_improvement/cdo_v7'
    control_dir = data/'cdo_v6_experiments/candidate_episodes'
    experiment = data/'cdo_v7_experiments'
    report.mkdir(parents=True, exist_ok=True)
    scored = json.loads((a.root/'incident_reports/track1_scored_frames.json').read_text())
    metadata = episode_metadata(a.root)
    ctx = dr.RasterizeCudaContext()
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

    def renderer(vertices, faces):
        return SimpleNamespace(pos=torch.as_tensor(vertices, device='cuda', dtype=torch.float32), faces=torch.as_tensor(faces, device='cuda', dtype=torch.int32))

    def render(rend, pose, k, shape):
        pose = torch.as_tensor(pose, device='cuda', dtype=torch.float32)
        camera = rend.pos@pose[:3, :3].T+pose[:3, 3]
        projection = torch.as_tensor(Utils.projection_matrix_from_intrinsics(k, height=shape[0], width=shape[1], znear=.001, zfar=100.), device='cuda', dtype=torch.float32)
        cv_to_gl = torch.as_tensor(Utils.glcam_in_cvcam, device='cuda', dtype=torch.float32)
        clip = Utils.to_homo_torch(camera)@(projection@cv_to_gl).T
        raster, _ = dr.rasterize(ctx, clip[None], rend.faces, resolution=shape)
        depth, _ = dr.interpolate(camera[None, :, 2:3].contiguous(), raster, rend.faces)
        return torch.flip(depth[0, :, :, 0], [0]).cpu().numpy(), torch.flip(raster[0, :, :, 3]>0, [0]).cpu().numpy()

    def evaluate(rend, poses, obs, frames):
        rows, renders = [], {}
        for frame in frames:
            item = obs[int(frame)]
            z, mask = render(rend, poses[frame], item['k'], item['shape'])
            iou = float(measure_visible_silhouettes(mask[None], item['human'], item['obj']).iou[0])
            support = mask & item['obj'] & ~item['human'] & np.isfinite(item['depth']) & (item['depth']>.05) & (z>.05)
            residual = z[support]-item['depth'][support]
            rows.append({'frame': int(frame), 'iou': iou, 'support_pixels': int(support.sum()),
                         'depth_bias_cm': float(np.median(residual)*100) if len(residual)>=8 else None,
                         'depth_error_cm': float(np.median(abs(residual))*100) if len(residual)>=8 else None})
            renders[int(frame)] = (z, mask)
        return rows, renders

    def common_depth(before, after, obs, frames):
        old, new = [], []
        for frame in frames:
            item=obs[int(frame)]; bz,bm=before[int(frame)]; nz,nm=after[int(frame)]
            support=bm & nm & item['obj'] & ~item['human'] & np.isfinite(item['depth']) & (item['depth']>.05) & (bz>.05) & (nz>.05)
            if support.sum()>=8:
                old.append(float(np.median(abs(bz[support]-item['depth'][support]))*100))
                new.append(float(np.median(abs(nz[support]-item['depth'][support]))*100))
        return {'supported_frames': len(old), 'before_cm': float(np.median(old)) if old else None, 'after_cm': float(np.median(new)) if new else None}

    def motion(poses, vertices):
        return {'translation_acceleration_cm_proxy': float(np.linalg.norm(np.diff(poses[:, :3, 3], n=2, axis=0), axis=1).mean()*100), **rigid_motion_diagnostics(poses, vertices)}

    with torch.inference_mode():
        box=trimesh.creation.box(extents=[1.,1.,1.]);pose=np.eye(4);pose[2,3]=3
        z,mask=render(renderer(box.vertices,box.faces),pose,np.array([[80.,0,48],[0,80.,32],[0,0,1.]]),(64,96))
        assert mask[32,48] and abs(z[32,48]-2.5)<1e-4
        for episode in a.episodes:
            stem=f'episode_{episode:06d}'; audit_path=report/(stem+'_audit.json')
            out=experiment/'geometry_candidates'/stem
            if (out/'probe.json').exists() and not a.audit_only:
                print('ALL_EPISODE_PROBE_REUSED',episode,flush=True);continue
            if audit_path.exists() and a.audit_only:
                old=json.loads(audit_path.read_text())
                assert old['control_npz_sha256']==digest(control_dir/(stem+'.npz'))
                assert old['control_mesh_sha256']==digest(control_dir/(stem+'_object.glb'))
                print('ALL_EPISODE_AUDIT_REUSED',episode,flush=True);continue
            q=paths(a.root,episode)
            work=data/('improvement_pilot_v1' if episode in (6,18,28) else 'improvement_rollout_v1')/stem
            export=work/'run/export'/stem
            k,w2c=camera_calibration(load_edex(export),0);assert np.allclose(w2c,np.eye(4))
            control=dict(np.load(control_dir/(stem+'.npz'),allow_pickle=False))
            mesh=trimesh.load(control_dir/(stem+'_object.glb'),force='mesh',process=False)
            vertices=mesh.vertices*float(control['object_scale'])
            poses=np.tile(np.eye(4),(len(control['pose']),1,1));poses[:,:3,:3]=control['object_rotation'];poses[:,:3,3]=control['object_translation']
            frames=np.array(scored[str(episode)]);frames=frames[np.linspace(0,len(frames)-1,min(33,len(frames))).round().astype(int)]
            obs={}
            for frame in frames:
                name=f'{frame:06d}';obj=read_mask(export,'object',0,name);hm=read_mask(export,'human',0,name)
                small_k,shape=_scaled_image_grid(k,obj.shape,512)
                resize=lambda x:cv2.resize(x,(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)
                obs[int(frame)]={'obj':resize(obj.astype(np.uint8))>0,'human':resize(hm.astype(np.uint8))>0,
                                 'depth':resize(read_depth_m(export,0,name,depth_root=q['depth'])),'k':small_k,'shape':shape}
            rend=renderer(vertices,mesh.faces);old_rows,old_renders=evaluate(rend,poses,obs,frames)
            supported=[r for r in old_rows if r['depth_bias_cm'] is not None]
            audit={'episode':episode,'metadata':metadata[episode],'frames':len(poses),'sample_frames':frames.tolist(),
                   'visible_iou_mean':float(np.mean([r['iou'] for r in old_rows])),
                   'depth_supported_frames':len(supported),'median_depth_bias_cm':float(np.median([r['depth_bias_cm'] for r in supported])) if supported else None,
                   'median_depth_error_cm':float(np.median([r['depth_error_cm'] for r in supported])) if supported else None,
                   'center_depth_median_m':float(np.median(poses[:,2,3])),'mesh_extents_m':np.ptp(vertices,axis=0).tolist(),
                   'motion':motion(poses,vertices),'control_npz_sha256':digest(control_dir/(stem+'.npz')),
                   'control_mesh_sha256':digest(control_dir/(stem+'_object.glb')),'samples':old_rows,
                   'scope':'Predicted-mask image alignment and cached monocular depth diagnostics; not official per-episode scores.'}
            if not audit_path.exists():audit_path.write_text(json.dumps(audit,indent=2)+'\n')
            print('ALL_EPISODE_AUDIT',episode,'iou',round(audit['visible_iou_mean'],4),'depth_bias_cm',audit['median_depth_bias_cm'],'jumps',audit['motion']['rotation_jumps_over_45_degrees'],flush=True)
            if a.audit_only:continue
            out.mkdir(parents=True,exist_ok=True)
            fit=frames[np.arange(len(frames))%3!=2];held=frames[np.arange(len(frames))%3==2]
            old_iou={r['frame']:r['iou'] for r in old_rows}
            fit_old=float(np.mean([old_iou[int(f)] for f in fit]));held_old=float(np.mean([old_iou[int(f)] for f in held]))
            base=poses.copy();base[:, :3,3]=savgol_filter(poses[:, :3,3],7,2,axis=0)
            smooth_rows,_=evaluate(rend,base,obs,fit)
            smoothing=float(np.mean([r['iou'] for r in smooth_rows]))>=fit_old-.003
            if not smoothing:base=poses.copy()
            grid=[];best=None
            for scale in (.90,.95,1.,1.05,1.10):
                scaled=renderer(vertices*scale,mesh.faces)
                for offset in (-.10,-.05,0.,.05,.10):
                    trial=base.copy();z=trial[:,2,3].copy()
                    if (z+offset<=.1).any():continue
                    trial[:,:3,3]*=((z+offset)/z)[:,None]
                    rows,renders=evaluate(scaled,trial,obs,fit)
                    iou=float(np.mean([r['iou'] for r in rows]));depth=common_depth(old_renders,renders,obs,fit)
                    gain=depth['before_cm']-depth['after_cm'] if depth['supported_frames']>=8 else None
                    motion_new=motion(trial,vertices*scale)
                    stable=all(motion_new[key]<=max(.1,audit['motion'][key]*1.05) for key in ['translation_acceleration_cm_proxy','sampled_vertex_acceleration_cm_proxy'])
                    eligible=stable and iou>=max(.3,fit_old-.005) and (gain is None or gain>=-.5)
                    motion_gain=1-motion_new['sampled_vertex_acceleration_cm_proxy']/max(1e-6,audit['motion']['sampled_vertex_acceleration_cm_proxy'])
                    objective=iou-fit_old+(.08*np.clip(gain/10.,-1.,1.) if gain is not None else 0.)+.005*max(0.,motion_gain)
                    row={'scale_factor':scale,'depth_offset_m':offset,'fit_iou':iou,'fit_common_depth':depth,'fit_eligible':bool(eligible),'fit_objective':float(objective)}
                    grid.append(row)
                    if eligible and (best is None or objective>best[0]+1e-8):best=(objective,scale,offset,trial,motion_new,row)
                print('ALL_EPISODE_GRID_PROGRESS',episode,'scale',scale,flush=True)
            accepted=False;held_summary=None;chosen=None
            if best is not None and best[0]>.005:
                _,scale,offset,trial,motion_new,chosen=best
                rows,renders=evaluate(renderer(vertices*scale,mesh.faces),trial,obs,held)
                held_iou=float(np.mean([r['iou'] for r in rows]));depth=common_depth(old_renders,renders,obs,held)
                fit_depth=chosen['fit_common_depth'];depth_driven=fit_depth['supported_frames']>=8 and fit_depth['before_cm']-fit_depth['after_cm']>=1.
                held_gain=depth['before_cm']-depth['after_cm'] if depth['supported_frames']>=4 else None
                held_summary={'control_iou':held_old,'candidate_iou':held_iou,'common_depth':depth,'depth_driven':bool(depth_driven)}
                accepted=held_iou>=max(.3,held_old-.005) and (held_gain is None or held_gain>=-.5) and (not depth_driven or (held_gain is not None and held_gain>=.5))
                meaningful=(held_iou-held_old>=.005 or (held_gain is not None and held_gain>=1.) or motion_new['sampled_vertex_acceleration_cm_proxy']<audit['motion']['sampled_vertex_acceleration_cm_proxy']*.95)
                accepted=bool(accepted and meaningful)
                if accepted:
                    result=dict(control);result.update(object_translation=trial[:,:3,3].astype(np.float32),object_scale=np.array(float(control['object_scale'])*scale,dtype=np.float32))
                    for key in ('pose','scales','shape','object_rotation'):assert np.array_equal(result[key],control[key])
                    np.savez_compressed(out/(stem+'.npz'),**result);shutil.copy2(control_dir/(stem+'_object.glb'),out/(stem+'_object.glb'))
                    all_iou=(chosen['fit_iou']*len(fit)+held_iou*len(held))/len(frames)
                    review={'episode':episode,'control_version':6,'ground_truth_used':False,'fit_frames':fit.tolist(),'held_out_frames':held.tolist(),
                            'summary':{'submitted_v6':{'visible_iou_mean':audit['visible_iou_mean'],**audit['motion']},'final_candidate':{'visible_iou_mean':float(all_iou),**motion_new}},
                            'geometry_review':{'accepted':True,'scale_factor':scale,'depth_offset_m':offset,'translation_smoothing':bool(smoothing),'held_out':held_summary},
                            'control_npz_sha256':audit['control_npz_sha256'],'control_mesh_sha256':audit['control_mesh_sha256']}
                    (out/'review.json').write_text(json.dumps(review,indent=2)+'\n')
            probe={'episode':episode,'accepted':accepted,'fit_frames':fit.tolist(),'held_out_frames':held.tolist(),'translation_smoothing':bool(smoothing),
                   'selected_fit':chosen,'held_out':held_summary,'grid':grid,'control_npz_sha256':audit['control_npz_sha256'],'control_mesh_sha256':audit['control_mesh_sha256'],
                   'scope':'Bounded depth/scale fit with disjoint frame checks; full hand and every-scored-frame validation are still required.'}
            (out/'probe.json').write_text(json.dumps(probe,indent=2)+'\n')
            (report/(stem+'_geometry_probe.json')).write_text(json.dumps(probe,indent=2)+'\n')
            print('ALL_EPISODE_PROBE_COMPLETE',episode,'ACCEPT' if accepted else 'RETAIN_CONTROL',flush=True)
    audits=[json.loads(p.read_text()) for p in sorted(report.glob('episode_*_audit.json'))]
    audits.sort(key=lambda d:(1-d['visible_iou_mean'])+.1*min(1.,(d['median_depth_error_cm'] or 0.)/30.)+.1*min(1.,d['motion']['rotation_jumps_over_45_degrees']/max(1,d['frames'])*10.),reverse=True)
    (report/'episode_ranking.json').write_text(json.dumps(audits,indent=2)+'\n')


if __name__=='__main__':
    main()
