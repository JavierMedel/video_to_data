"""Observed-mask human alignment and unsigned hand/object proximity diagnostics."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.spatial import cKDTree
from track1_rollout_common import paths

p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--episode',type=int,required=True); a=p.parse_args()
sys.path.insert(0,'/workspace/v2d_foundation_pose/lib/FoundationPose')
sys.path.insert(0,'/workspace/v2d_sam3d_body/lib')
sys.path.insert(0,str(a.root/'modules/v2d_cari4d/lib/cari4d'))
from lib_mhr import MHRLayer, MHR_PARAM_DIMS
from lib_mhr.hand_surface_contact import load_mhr_hand_surface_spec
from prep.run_foundationpose_mhr_export import _render_candidate_masks, _scaled_image_grid
from prep.mhr_export_utils import read_mask, read_rgb

q=paths(a.root,a.episode); work=q['work']; stem=q['stem']; export=work/'run/export'/stem
old_path=q['baseline_run']/'inference/refined.pth' if a.episode!=16 else q['baseline_run']/'cari4d_refined.pth'
bundles={'baseline':torch.load(old_path,map_location='cpu',weights_only=False),'candidate':torch.load(work/'run/inference/refined.pth',map_location='cpu',weights_only=False)}
scored=json.loads((a.root/'incident_reports/track1_scored_frames.json').read_text())[str(a.episode)]
frames=np.array(scored)[np.linspace(0,len(scored)-1,min(33,len(scored))).round().astype(int)]
panel_indices={0,len(frames)//2,len(frames)-1}
k=np.asarray(json.loads((export/'wild_export.json').read_text())['intrinsics'])
layer=MHRLayer.from_mhr_assets(mhr_assets_root=str(a.root/'data/weights/sam3d_body'),device='cuda')
faces=layer.mesh_faces(device='cuda').detach().cpu().numpy().astype(np.int32)
spec=load_mhr_hand_surface_spec(a.root/'modules/v2d_cari4d/lib/cari4d/lib_mhr/assets/mhr_hand_surface_spec.npz',faces=faces)
hand_ids=spec.vertex_indices.reshape(-1)
renderer=SimpleNamespace(mesh_tensors={'faces':torch.tensor(faces,device='cuda',dtype=torch.int32)},glctx=dr.RasterizeCudaContext())
summary={}; samples={}; panels=[]
for label,bundle in bundles.items():
    vertices=[]
    with torch.no_grad():
        for start in range(0,len(frames),16):
            ids=frames[start:start+16]
            params={key:torch.as_tensor(np.asarray(bundle['pr'][key])[ids],device='cuda',dtype=torch.float32) for key in MHR_PARAM_DIMS}
            vertices.extend(layer.mhr_forward_vertices(params).cpu().numpy())
    pair_root=q['baseline_pairs'] if label=='baseline' else work/'submission_episodes'
    with np.load(pair_root/(stem+'.npz'),allow_pickle=False) as packed:
        rotations=packed['object_rotation']; translations=packed['object_translation']; scale=float(packed['object_scale'].reshape(()))
    mesh=trimesh.load(pair_root/(stem+'_object.glb'),force='mesh',process=False)
    tree=cKDTree(np.asarray(mesh.vertices)*scale)
    rows=[]
    for j,frame in enumerate(frames):
        human=read_mask(export,'human',0,f'{frame:06d}'); obj=read_mask(export,'object',0,f'{frame:06d}')
        small_k,shape=_scaled_image_grid(k,human.shape,768)
        target=cv2.resize(human.astype(np.uint8),(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)>0
        occlusion=cv2.resize(obj.astype(np.uint8),(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)>0
        renderer.mesh_tensors['pos']=torch.tensor(vertices[j],device='cuda',dtype=torch.float32)
        rendered=_render_candidate_masks(renderer,np.eye(4,dtype=np.float32)[None],small_k,shape)[0]>0
        visible=rendered&~occlusion
        union=(visible|target).sum(); iou=float((visible&target).sum()/max(1,union))
        hand_local=(vertices[j][hand_ids]-translations[frame])@rotations[frame]
        distances,_=tree.query(hand_local,k=1)
        rows.append({'frame':int(frame),'human_visible_iou':iou,'nearest_hand_object_vertex_distance_cm_proxy':float(distances.min()*100)})
        if j in panel_indices:
            rgb=cv2.resize(read_rgb(export,0,f'{frame:06d}'),(shape[1],shape[0]))
            rgb[target]=(rgb[target]*.65+np.array([0,180,255])*.35).astype(np.uint8)
            contours,_=cv2.findContours(visible.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb,contours,-1,(255,0,255),1)
            cv2.putText(rgb,f'{label} f{frame} human IoU {iou:.3f}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
            panels.append(cv2.resize(rgb,(512,384)))
    roots=np.asarray(bundle['pr']['mhr_trans'])
    summary[label]={'human_visible_iou_mean':float(np.mean([r['human_visible_iou'] for r in rows])),'human_translation_acceleration_cm_proxy':float(np.linalg.norm(np.diff(roots,n=2,axis=0),axis=1).mean()*100),'hand_object_nearest_vertex_distance_cm_median_proxy':float(np.median([r['nearest_hand_object_vertex_distance_cm_proxy'] for r in rows]))}
    samples[label]=rows
result={'episode':a.episode,'sample_count':len(frames),'ground_truth_used':False,'summary':summary,'samples':samples,'contact_limit':'Unsigned nearest-vertex distance is an upper-bound proximity proxy; it does not measure penetration or prove grasp/contact. Zero proximity alone is not evidence of correctness.'}
(work/'human_contact_review.json').write_text(json.dumps(result,indent=2)+'\n')
sheet=np.concatenate([np.concatenate(panels[j:j+3],axis=1) for j in (0,3)],axis=0)
cv2.imwrite(str(q['report_dir']/(stem+'_human_comparison.jpg')),cv2.cvtColor(sheet,cv2.COLOR_RGB2BGR))
print('ROLLOUT_HUMAN_CONTACT_REVIEW',json.dumps(summary),flush=True)
