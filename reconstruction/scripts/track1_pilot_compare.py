"""Compare baseline and pilot trajectories against the SAME corrected masks."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
p=argparse.ArgumentParser(); p.add_argument('root',type=Path); p.add_argument('episode',type=int)
p.add_argument('--work-root',type=Path); p.add_argument('--baseline-pairs',type=Path); p.add_argument('--report-dir',type=Path)
a=p.parse_args(); root=a.root; i=a.episode; stem=f'episode_{i:06d}'
sys.path.insert(0,'/workspace/v2d_foundation_pose/lib/FoundationPose')
sys.path.insert(0,str(root/'modules/v2d_cari4d/lib/cari4d'))
from prep.run_foundationpose_mhr_export import _render_candidate_masks,_scaled_image_grid
from prep.foundationpose_pose_selection import measure_visible_silhouettes
from prep.mhr_export_utils import read_mask,read_rgb
data=root/'data/v2d_challenge/track_1'; work=data/'improvement_pilot_v1'/stem; export=work/'run/export'/stem
if a.work_root is not None: work=a.work_root/stem; export=work/'run/export'/stem
baseline_pairs=a.baseline_pairs or data/'submission_episodes'
report_dir=a.report_dir or root/'incident_reports/track1_improvement'
report_dir.mkdir(parents=True,exist_ok=True)
k=np.array(json.loads((export/'wild_export.json').read_text())['intrinsics'])
scored=json.loads((root/'incident_reports/track1_scored_frames.json').read_text())[str(i)]
frames=np.array(scored)[np.linspace(0,len(scored)-1,min(33,len(scored))).round().astype(int)]
baseline=dict(np.load(baseline_pairs/(stem+'.npz')))
pilot=dict(np.load(work/'submission_episodes'/(stem+'.npz')))
import pickle
fp=pickle.load(open(work/'run/foundationpose'/(stem+'.pkl'),'rb'))
coco=torch.load(work/'run/inference/coconet.pth',map_location='cpu',weights_only=False)
refined=torch.load(work/'run/inference/refined.pth',map_location='cpu',weights_only=False)
poses_baseline=np.repeat(np.eye(4)[None],len(baseline['pose']),axis=0); poses_baseline[:,:3,:3]=baseline['object_rotation']; poses_baseline[:,:3,3]=baseline['object_translation']
trajectories={'baseline':poses_baseline,'gated_tracker':fp['obj_pose_world'],'coconet':np.asarray(coco['pr']['pose_abs']),'refined':np.asarray(refined['pr'].get('pose_abs_postopt',refined['pr']['pose_abs']))}
mesh_paths={'baseline':baseline_pairs/(stem+'_object.glb'),'pilot':work/'submission_episodes'/(stem+'_object.glb')}
renderers={}
for label,path in mesh_paths.items():
    mesh=trimesh.load(path,force='mesh',process=False)
    renderers[label]=SimpleNamespace(mesh_tensors={'pos':torch.tensor(np.asarray(mesh.vertices),device='cuda',dtype=torch.float32),'faces':torch.tensor(np.asarray(mesh.faces),device='cuda',dtype=torch.int32)},glctx=dr.RasterizeCudaContext())
results={label:[] for label in trajectories}; panels=[]
with torch.no_grad():
    for frame in frames:
        human=read_mask(export,'human',0,f'{frame:06d}'); obj=read_mask(export,'object',0,f'{frame:06d}')
        small_k,shape=_scaled_image_grid(k,obj.shape,768)
        human=cv2.resize(human.astype(np.uint8),(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)>0
        obj=cv2.resize(obj.astype(np.uint8),(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)>0
        rgb=cv2.resize(read_rgb(export,0,f'{frame:06d}'),(shape[1],shape[0]))
        for label,poses in trajectories.items():
            mask=_render_candidate_masks(renderers['baseline' if label=='baseline' else 'pilot'],poses[frame:frame+1],small_k,shape)
            m=measure_visible_silhouettes(mask,human,obj)
            results[label].append({'frame':int(frame),'visible_iou':float(m.iou[0]),'precision':float(m.precision[0]),'recall':float(m.recall[0]),'depth_m':float(poses[frame,2,3])})
            if len(panels)<12 and label in ('baseline','refined'):
                overlay=rgb.copy(); overlay[obj]=(overlay[obj]*.5+np.array([0,255,0])*.5).astype(np.uint8)
                contours,_=cv2.findContours(mask[0].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay,contours,-1,(255,60,60),1)
                cv2.putText(overlay,f'{label} frame {frame} visible IoU {m.iou[0]:.3f}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
                panels.append(cv2.resize(overlay,(512,384)))
summary={}
for label,poses in trajectories.items():
    values=np.array([r['visible_iou'] for r in results[label]])
    summary[label]={'visible_iou_mean':float(values.mean()),'visible_iou_median':float(np.median(values)),'negative_center_depth_frames':int((poses[:,2,3]<=0).sum()),'depth_range_m':[float(poses[:,2,3].min()),float(poses[:,2,3].max())],'translation_acceleration_cm_proxy':float(np.linalg.norm(np.diff(poses[:,:3,3],n=2,axis=0),axis=1).mean()*100)}
report={'episode':i,'comparison':'all methods against corrected observed masks on identical scored-frame samples','held_out_ground_truth_used':False,'sample_count':len(frames),'tracker_metadata':fp['metadata'],'summary':summary,'samples':results}
(work/'comparison.json').write_text(json.dumps(report,indent=2))
sheet=np.concatenate([np.concatenate(panels[j:j+4],axis=1) for j in range(0,12,4)],axis=0)
cv2.imwrite(str(report_dir/(stem+'_candidate_comparison.jpg')),cv2.cvtColor(sheet,cv2.COLOR_RGB2BGR))
with (report_dir/'progress.md').open('a') as f: f.write('\nCandidate comparison '+stem+'\n```json\n'+json.dumps(summary,indent=2)+'\n```\n')
print('PILOT_COMPARISON',json.dumps(report['summary']),flush=True)
