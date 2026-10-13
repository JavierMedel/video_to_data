"""Generate one depth-grounded metric mesh per remaining physical object group."""
import argparse
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import trimesh
from track1_rollout_common import paths

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(); p.add_argument('--episodes',type=int,nargs='+',required=True)
p.add_argument('--wait-for-masks',type=int,nargs='*',default=[]); a=p.parse_args()
for i in a.wait_for_masks:
    q=paths(ROOT,i)
    sheet=q['report_dir']/(q['stem']+'_corrected_masks.jpg')
    while not sheet.exists(): time.sleep(30)
time.sleep(2)
for i in a.episodes:
    q=paths(ROOT,i); work=q['work']; mesh=work/'mesh'; mesh.mkdir(parents=True,exist_ok=True)
    if (mesh/'object_metric.glb').exists():
        assert (mesh/'mesh_validation.json').exists(), 'Existing mesh requires its validation record'
        continue
    command=['docker','run','--rm','--user','1000:1000','--entrypoint','python','-v',f'{ROOT}:/repo','v2d_cari4d:blackwell','/repo/scripts/track1_rollout_mesh_inputs.py','--root','/repo','--episode',str(i)]
    with (work/'mesh_inputs.log').open('a') as log: subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    prefix=f'/repo/data/v2d_challenge/track_1/improvement_rollout_v1/{q["stem"]}'
    command=['docker','run','--rm','--gpus','all','--user','1000:1000','--entrypoint','python','-e','PYTHONUNBUFFERED=1','-e','XDG_CACHE_HOME=/tmp/v2d-rollout-cache','-e','TORCH_HOME=/repo/data/weights/sam3d/torch_home','-e','HF_HOME=/repo/data/weights/sam3d/hf_home','-v',f'{ROOT}:/repo','-v',f'{ROOT}/modules/v2d_sam3d/lib:/workspace/v2d_sam3d/lib:ro','v2d_sam3d:blackwell','-m','v2d.sam3d.lib.image_to_mesh','--image_path',prefix+'/mesh_inputs/rgb.png','--mask_path',prefix+'/mesh_inputs/mask.png','--mesh_path',prefix+'/mesh/object.glb','--transform_path',prefix+'/mesh/transform.json','--intrinsics_path',prefix+'/mesh/intrinsics.json','--weights_dir','/repo/data/weights/sam3d','--pointmap_path',prefix+'/mesh_inputs/pointmap.npy','--pointmap_intrinsics_path',prefix+'/mesh_inputs/intrinsics.json','--seed','42']
    if not (mesh/'object.glb').exists():
        with (work/'sam3d.log').open('a') as log: subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    transform=json.loads((mesh/'transform.json').read_text()); scales=np.asarray(transform['scale'],float)
    assert scales.shape==(3,) and np.isfinite(scales).all() and (scales>0).all()
    assert np.allclose(scales,scales[0],rtol=1e-4), 'Nonuniform scale requires separate contract review'
    scene=trimesh.load(mesh/'object.glb',force='scene'); scene.apply_scale(float(scales[0])); scene.export(mesh/'object_metric.glb')
    loaded=trimesh.load(mesh/'object_metric.glb',force='mesh'); assert len(loaded.faces) and np.isfinite(loaded.vertices).all() and loaded.area>0
    assert 0.01 < np.linalg.norm(loaded.extents) < 5, 'Metric mesh extent is implausible for these target objects'
    result={'episode':i,'status':'STRUCTURAL_AND_DEPTH_SCALE_PASS','applied_uniform_scale':float(scales[0]),'extents_m':loaded.extents.tolist(),'faces':len(loaded.faces),'calibration':json.loads((work/'mesh_inputs/review.json').read_text()),'ground_truth_used':False,'image_agreement_review':'PENDING_TRACKING_COMPARISON'}
    (mesh/'mesh_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('ROLLOUT_METRIC_MESH_READY',i,json.dumps(result),flush=True)
