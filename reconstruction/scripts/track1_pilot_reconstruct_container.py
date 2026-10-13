"""Run downstream pilot stages with corrected masks, mesh and gated tracking."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
ROOT=Path('/repo')
SOURCE=ROOT/'modules/v2d_cari4d/lib/cari4d'
sys.path.insert(0,str(SOURCE))
sys.path.insert(0,'/workspace/v2d_sam3d_body/lib')
sys.path.insert(0,'/workspace/v2d_foundation_pose/lib/FoundationPose')
from v2d.cari4d.lib.pack_masks import masks_pack_cari4d_h5
from v2d.cari4d.lib.run_inference import _stage_run, _atomic_json

p=argparse.ArgumentParser(); p.add_argument('--episode',type=int,required=True); p.add_argument('--refinement-steps',type=int,default=100)
p.add_argument('--work-root',type=Path)
p.add_argument('--allow-episode-16',action='store_true')
a=p.parse_args()
assert 0 <= a.episode < 30
if a.work_root is None: assert a.episode in (6,18,28), 'Only approved pilot episodes in the pilot directory'
if a.episode == 16 and not a.allow_episode_16: raise SystemExit('Episode 16 requires explicit isolated-rerun approval')
stem=f'episode_{a.episode:06d}'; data=ROOT/'data/v2d_challenge/track_1'; work=data/'improvement_pilot_v1'/stem
baseline=data/'cari4d_runs/track1_resume'/stem/stem
if a.work_root is not None:
    from track1_rollout_common import paths
    q=paths(ROOT,a.episode)
    work=a.work_root.resolve()/stem
    assert work.is_relative_to(data/'improvement_rollout_v1'), 'Rollout must use its isolated output root'
    baseline=q['baseline_run']
    review=json.loads((work/'review_approval.json').read_text())
    assert review['verdict']=='PASS' and review['episode']==a.episode
    assert hashlib.sha256((work/'cari4d_masks.h5').read_bytes()).hexdigest()==review['mask_sha256'], 'Reviewed masks changed'
    assert hashlib.sha256((work/'mesh/object_metric.glb').read_bytes()).hexdigest()==review['mesh_sha256'], 'Reviewed mesh changed'
run=work/'run'; run.mkdir(exist_ok=True); markers=run/'.stages'
video=work/(stem+'.0.color.mp4')
if not video.exists(): video.hardlink_to(data/'videos/chunk-000/observation.images.exo_camera'/(stem+'.mp4'))
masks=work/'cari4d_masks.h5'
if not masks.exists():
    if a.work_root is not None:
        raise SystemExit('Rollout requires prepared and reviewed packed masks')
    masks_pack_cari4d_h5(video,data/stem/'sam2_human_masks/0',work/'object_masks/1',masks)
weights=ROOT/'data/weights'
env=os.environ.copy()
env.update(PYTHONPATH=os.pathsep.join([str(SOURCE),'/workspace/v2d_sam3d_body/lib','/workspace/v2d_foundation_pose/lib/FoundationPose']),MHR_ASSETS_ROOT=str(weights/'sam3d_body'),FOUNDATIONPOSE_WEIGHTS_DIR=str(weights/'foundationpose/nvlabs_pytorch'),TORCH_HOME=str(weights/'sam3d_body/torch_home'),HF_HOME=str(weights/'hf_home'),OPENCV_IO_ENABLE_OPENEXR='1',PYOPENGL_PLATFORM='egl',HDF5_USE_FILE_LOCKING='FALSE',PYTHONUNBUFFERED='1')
stages=[]
def stage(name,cmd,inputs,outputs):
    stages.append(_stage_run(name,cmd,inputs,outputs,markers,env,False))
python=sys.executable; export=run/'export'/stem; mesh=export/'object_mesh/output_aligned.glb'; export_marker=export/'wild_export.json'
input_mesh=work/'mesh/object_metric.glb'; intrinsics=baseline/'depth/moge2/intrinsics.pkl'; depth=baseline/'depth/moge2/aligned.h5'; init=baseline/'mhr_init'/(stem+'.pkl')
if a.work_root is not None: intrinsics,depth,init=q['intrinsics'],q['depth'],q['init']
stage('02_export',[python,str(SOURCE/'prep/prepare_mhr_wild_export.py'),'--video',str(video),'--mask-h5',str(masks),'--object-mesh',str(input_mesh),'--intrinsics-file',str(intrinsics),'--output-root',str(run/'export')],[video,masks,input_mesh,intrinsics],[export_marker,mesh])
fp=run/'foundationpose'/f'{stem}.pkl'; fp.parent.mkdir(exist_ok=True)
tracker_command=[python,str(SOURCE/'prep/run_quality_gated_foundationpose.py'),'--export-seq',str(export),'--depth-file',str(depth),'--mesh-file',str(mesh),'--output',str(fp)]
tracker_inputs=[export_marker,depth,mesh]
if a.work_root is not None:
    tracker_command.append('--resume')
    tracker_inputs += [SOURCE/'prep/run_quality_gated_foundationpose.py',SOURCE/'prep/tracker_checkpoint.py',SOURCE/'prep/quality_gate_utils.py']
stage('05_object_pose',tracker_command,tracker_inputs,[fp,fp.with_suffix('.json')])
inference=run/'inference'; inference.mkdir(exist_ok=True)
coco=inference/'coconet.pth'; refined=inference/'refined.pth'; cache=inference/'materialized_inputs.h5'; checkpoint=inference/'refinement_checkpoint.pth'
config=weights/'cari4d/2026-08-25-09-35-57/resolved_config.yaml'; model=weights/'cari4d/2026-08-25-09-35-57/step200000.pth'
stage('06_coconet',[python,str(SOURCE/'tools/run_mhr_wild_inference.py'),str(export),'--depth-h5',str(depth),'--mhr-init',str(init),'--foundationpose-file',str(fp),'--config',str(config),'--checkpoint',str(model),'--output',str(coco),'--stride','96','--render-batch-size','32','--crop-workers','8','--crop-buffer-count','2','--input-cache',str(cache),'--device','cuda','--offline-supervision-contract'],[export_marker,depth,init,fp,config,model],[coco,cache])
stage('07_refinement',[python,'-m','learning.training.mhr_opt_refineout','--bundle',str(coco),'--object-mesh',str(mesh),'--out',str(refined),'--mode','smplh_parity','--num-steps',str(a.refinement_steps),'--batch-size','64','--device','cuda','--postopt-checkpoint',str(checkpoint),'--mhr-assets-root',str(weights/'sam3d_body'),'--w-temporal','100','--w-human-pose-prior','200','--contact-activation-distance-m','0.05','--report-every','10','--diagnostics-every','500'],[coco,mesh,weights/'sam3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt'],[refined,checkpoint])
_atomic_json(run/'pipeline_report.json',{'schema':'v2d.track1.improvement_pilot.v1','verdict':'READY_FOR_CONVERSION','episode':stem,'stages':stages,'reused_depth':str(depth),'reused_human_initialization':str(init),'render_deferred':True,'ground_truth_used':False})
print('PILOT_RECONSTRUCTION_COMPLETE',stem,flush=True)
