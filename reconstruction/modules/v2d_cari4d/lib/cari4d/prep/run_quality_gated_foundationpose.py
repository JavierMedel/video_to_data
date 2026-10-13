"""Ground-truth-free FoundationPose tracking with observed-mask/depth gates.

This opt-in pilot entry point preserves the original export and tracker. Failed
frames never update the trusted tracking state. Registration is retried after
failure; missing poses are carried from actual accepted predictions.
"""
import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
import cv2
import numpy as np
import trimesh
from prep.mhr_export_utils import camera_calibration, frame_names, load_edex, read_depth_m, read_mask, read_rgb
from prep.foundationpose_pose_selection import measure_visible_silhouettes
from prep.quality_gate_utils import assess_candidates
from prep.run_foundationpose_mhr_export import _pose_estimator, _render_candidate_masks, _scaled_image_grid, _set_estimator_pose_last, _tensor_to_numpy
from prep.tracker_checkpoint import load_checkpoint, save_checkpoint

def run(export_seq,depth_file,mesh_file,output,min_iou=0.15,iteration=5,resume=False):
    output=Path(output)
    if output.exists() and not resume: raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    names=frame_names(export_seq,0)
    checkpoint=output.with_suffix('.progress.pkl')
    signature={'min_iou':min_iou,'iteration':iteration,'inputs':[], 'implementation':{}}
    for path in (Path(export_seq)/'wild_export.json',Path(depth_file),Path(mesh_file)):
        stat=path.stat()
        signature['inputs'].append({'path':str(path.resolve()),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns})
    for path in (Path(__file__),Path(__file__).with_name('tracker_checkpoint.py'),Path(__file__).with_name('quality_gate_utils.py')):
        signature['implementation'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    previous=None; poses=[]; records=[]
    if resume and checkpoint.exists():
        poses,records,previous=load_checkpoint(checkpoint,signature,names)
        print('QUALITY_TRACKER_RESUMED',len(records),len(names),flush=True)
    if output.exists():
        if len(records)!=len(names): raise ValueError('Existing tracker output lacks a complete matching checkpoint')
        with output.open('rb') as f: existing=pickle.load(f)
        accepted=[j for j,pose in enumerate(poses) if pose is not None]
        if not accepted: raise ValueError('Existing tracker output has no accepted checkpoint pose')
        expected=[poses[accepted[0]] if pose is None else pose for pose in poses]
        if existing['frames']!=names or not np.array_equal(existing['obj_pose_world'],np.asarray(expected,dtype=np.float32)) or not np.array_equal(existing['fp_reliable'],np.array([r['reliable'] for r in records])):
            raise ValueError('Existing tracker output differs from its complete checkpoint')
        if not output.with_suffix('.json').exists():
            output.with_suffix('.json').write_text(json.dumps({'metadata':existing['metadata'],'frames':existing['quality_diagnostics']},indent=2))
        print('QUALITY_TRACKER_COMPLETE_REUSED',len(records),flush=True)
        return
    mesh=trimesh.load(mesh_file,force='mesh',process=False)
    estimator=_pose_estimator(mesh,output.parent/'quality_tracker_debug')
    diameter=float(estimator.diameter)
    k,w2c=camera_calibration(load_edex(export_seq),0); c2w=np.linalg.inv(w2c)
    transform=_tensor_to_numpy(estimator.get_tf_to_centered_mesh())
    for frame in range(len(records),len(names)):
        name=names[frame]
        rgb=read_rgb(export_seq,0,name); human=read_mask(export_seq,'human',0,name); obj=read_mask(export_seq,'object',0,name)
        depth=read_depth_m(export_seq,0,name,depth_root=depth_file)
        visible=obj&~human&np.isfinite(depth)&(depth>0.05)
        expected=float(np.median(depth[visible])) if visible.sum()>=32 else float('nan')
        small_k,shape=_scaled_image_grid(k,obj.shape,384)
        small_h=cv2.resize(human.astype(np.uint8),(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)>0
        small_o=cv2.resize(obj.astype(np.uint8),(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)>0
        chosen=None; source='carry'; best_iou=0.; rejection='insufficient_visible_depth'
        if np.isfinite(expected):
            if previous is not None:
                _set_estimator_pose_last(estimator,previous)
                estimator.track_one(rgb=rgb,depth=depth,K=k,iteration=iteration)
                tracked=_tensor_to_numpy(estimator.pose_last).reshape(1,4,4)
                masks=_render_candidate_masks(estimator,tracked,small_k,shape)
                metrics=measure_visible_silhouettes(masks,small_h,small_o)
                accepted,scores=assess_candidates(tracked,metrics,expected,diameter,previous,min_iou)
                best_iou=float(metrics.iou[0]); rejection='track_failed_mask_depth_or_motion_gate'
                if accepted[0]: chosen=tracked[0]; source='track'; rejection=''
            if chosen is None:
                # Rank by independent visible silhouette agreement, not the
                # first model-ranked hypothesis. Reacquisition can move farther
                # than regular tracking, but must satisfy observed depth/mask.
                estimator.register(K=k,rgb=rgb,depth=depth,ob_mask=obj,iteration=iteration,seed=0)
                candidates=_tensor_to_numpy(estimator.poses)[:16]
                masks=_render_candidate_masks(estimator,candidates,small_k,shape)
                metrics=measure_visible_silhouettes(masks,small_h,small_o)
                accepted,scores=assess_candidates(candidates,metrics,expected,diameter,None,min_iou)
                best_iou=float(np.max(metrics.iou))
                if accepted.any():
                    index=int(np.argmax(scores)); chosen=candidates[index]; source='register'; rejection=''; best_iou=float(metrics.iou[index])
                else: rejection='registration_failed_mask_or_depth_gate'
        reliable=chosen is not None
        if reliable: previous=chosen.copy()
        if previous is not None:
            _set_estimator_pose_last(estimator,previous)
            poses.append(c2w@(previous@transform))
        else: poses.append(None)
        records.append({'frame':name,'reliable':reliable,'source':source,'iou':best_iou,'expected_depth_m':expected if np.isfinite(expected) else None,'rejection':rejection})
        if resume: save_checkpoint(checkpoint,signature,names,poses,records,previous)
        if frame%25==0: print('QUALITY_TRACKER_PROGRESS',frame,len(names),json.dumps(records[-1]),flush=True)
    valid=[j for j,pose in enumerate(poses) if pose is not None]
    if not valid: raise RuntimeError('No pose passed observed-mask/depth gates; refusing a placeholder trajectory')
    for frame in range(valid[0]): poses[frame]=poses[valid[0]].copy(); records[frame]['source']='backfill_first_accepted'
    payload={'frames':names,'obj_pose_world':np.asarray(poses,dtype=np.float32),'fp_reliable':np.array([r['reliable'] for r in records]),'metadata':{'schema':'v2d.track1.quality_gated_foundationpose.v1','ground_truth_used':False,'mesh':str(mesh_file),'min_iou':min_iou,'diameter_m':diameter,'accepted_frames':sum(r['reliable'] for r in records),'total_frames':len(names),'fallback':'carry_last_accepted_or_backfill_first_accepted'},'quality_diagnostics':records}
    output.with_suffix('.json').write_text(json.dumps({'metadata':payload['metadata'],'frames':records},indent=2))
    temporary=output.with_suffix('.tmp')
    with temporary.open('wb') as f: pickle.dump(payload,f)
    os.replace(temporary,output)
    print('QUALITY_TRACKER_COMPLETE',json.dumps(payload['metadata']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('export-seq','depth-file','mesh-file','output'): p.add_argument('--'+name,required=True)
    p.add_argument('--min-iou',type=float,default=0.15); p.add_argument('--iteration',type=int,default=5)
    p.add_argument('--resume',action='store_true',help='Save every completed frame and resume only matching inputs')
    a=p.parse_args(); run(a.export_seq,a.depth_file,a.mesh_file,a.output,a.min_iou,a.iteration,a.resume)
