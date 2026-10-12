"""Read-only diagnostic against observed video masks, not held-out labels."""
import json
import sys
from pathlib import Path
import h5py
import numpy as np
import torch
import trimesh

ROOT=Path(sys.argv[1]); DATA=ROOT/'data/v2d_challenge/track_1'
scored_frames=json.loads((ROOT/'incident_reports/track1_scored_frames.json').read_text())
rows=[]

def bbox(points):
    return np.r_[points.min(axis=0),points.max(axis=0)]

def box_iou(a,b):
    intersection=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    areas=np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()
    return float(intersection/max(areas-intersection,1e-8))

for i in range(30):
    stem=f'episode_{i:06d}'
    run=(DATA/'cari4d_runs/track1_submission'/stem/stem if i<6 else DATA/'cari4d_runs/track1_resume'/stem/stem)
    if i==1: run=DATA/'cari4d_runs/episode_000001_stability'/stem
    if i==15: run=DATA/'cari4d_runs/episode_000015_test/2026-09-10_16-14-06_foam_grass_block_rotate_02'
    if i==16:
        bundle_path=DATA/stem/'cari4d_refined.pth'
        exports=list((DATA/stem/'cari4d_export').glob('*/wild_export.json'))
    else:
        bundle_path=run/'inference/refined.pth'
        exports=list((run/'export').glob('*/wild_export.json'))
    d=dict(np.load(DATA/'submission_episodes'/(stem+'.npz'),allow_pickle=False))
    b=torch.load(bundle_path,map_location='cpu',weights_only=False)
    export=json.loads(exports[0].read_text())
    k=np.asarray(export['intrinsics'])
    mesh=trimesh.load(DATA/'submission_episodes'/(stem+'_object.glb'),force='mesh')
    verts=np.asarray(mesh.vertices)[::max(1,len(mesh.vertices)//3000)]
    scored=scored_frames[str(i)]
    selected=np.array(scored)[np.linspace(0,len(scored)-1,min(9,len(scored))).round().astype(int)]
    init=np.asarray(b['in']['pose_abs'])
    human=np.asarray(b['pr']['mhr_trans'])
    predictions=np.asarray(b['pr'].get('pose_abs_postopt',b['pr']['pose_abs']))
    entry={'episode':i,'object':json.loads((DATA/'meta/episodes_metadata.jsonl').read_text().splitlines()[i])['object'],
           'mesh_extents_m':mesh.extents.tolist(),'root_object_distance_m_median':float(np.median(np.linalg.norm(d['object_translation']-human,axis=1))),
           'object_z_range_m':[float(d['object_translation'][:,2].min()),float(d['object_translation'][:,2].max())],
           'human_z_range_m':[float(human[:,2].min()),float(human[:,2].max())],
           'object_acceleration_cm_proxy':float(np.linalg.norm(np.diff(d['object_translation'],n=2,axis=0),axis=1).mean()*100),
           'init_acceleration_cm_proxy':float(np.linalg.norm(np.diff(init[:,:3,3],n=2,axis=0),axis=1).mean()*100),
           'prediction_vs_initial_translation_m_median':float(np.median(np.linalg.norm(d['object_translation']-init[:,:3,3],axis=1))),
           'pack_translation_matches_bundle':bool(np.allclose(d['object_translation'],predictions[:,:3,3])),
           'body_conversion_report':str(d.get('report','')),'samples':[]}
    with h5py.File(DATA/stem/'cari4d_masks.h5','r') as masks:
        group=masks[next(iter(masks.keys()))]
        for frame in selected:
            mask=group[f'{frame:06d}-k0.obj_rend_mask.png'][()]
            yy,xx=np.nonzero(mask)
            if not len(xx): continue
            observed=np.array([xx.min(),yy.min(),xx.max(),yy.max()],float)
            metrics={'frame':int(frame)}
            for label,rotation,translation in [('submitted',d['object_rotation'][frame],d['object_translation'][frame]),('initial',init[frame,:3,:3],init[frame,:3,3])]:
                posed=verts@rotation.T+translation
                z=posed[:,2]
                projected=posed@k.T
                uv=projected[:,:2]/projected[:,2:]
                projected_box=bbox(uv)
                metrics[label+'_bbox_iou']=box_iou(projected_box,observed)
                metrics[label+'_center_error_px']=float(np.linalg.norm((projected_box[:2]+projected_box[2:]-observed[:2]-observed[2:])/2))
                metrics[label+'_extent_ratio']=np.divide(projected_box[2:]-projected_box[:2],np.maximum(1,observed[2:]-observed[:2])).tolist()
                metrics[label+'_behind_camera_fraction']=float((z<=0).mean())
            entry['samples'].append(metrics)
    for label in ('submitted','initial'):
        entry[label+'_bbox_iou_median']=float(np.median([s[label+'_bbox_iou'] for s in entry['samples']]))
        entry[label+'_center_error_px_median']=float(np.median([s[label+'_center_error_px'] for s in entry['samples']]))
    rows.append(entry)
    print(i,entry['object'], 'IoU',round(entry['submitted_bbox_iou_median'],3),'initial',round(entry['initial_bbox_iou_median'],3),'root distance',round(entry['root_object_distance_m_median'],2),flush=True)
out=ROOT/'incident_reports/track1_score_audit.json'
out.write_text(json.dumps(rows,indent=2))
print('Saved',out,flush=True)
