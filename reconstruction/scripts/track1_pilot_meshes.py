"""Regenerate depth-grounded SAM3D meshes for reviewed pilot masks."""
import json
import subprocess
from pathlib import Path
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT/'data/v2d_challenge/track_1/improvement_pilot_v1'
REPORT=ROOT/'incident_reports/track1_improvement/progress.md'
for i in (6,28,18):
    stem=f'episode_{i:06d}'; work=PILOT/stem; mesh=work/'mesh'; mesh.mkdir(exist_ok=True)
    if (mesh/'object_metric.glb').exists(): continue
    assert (work/'mask_review.json').exists(), 'Review required before mesh generation'
    prefix=f'/repo/data/v2d_challenge/track_1/improvement_pilot_v1/{stem}'
    cmd=['docker','run','--rm','--gpus','all','--user','1000:1000','--entrypoint','python','-e','PYTHONUNBUFFERED=1','-e','TORCH_HOME=/repo/data/weights/sam3d/torch_home','-e','HF_HOME=/repo/data/weights/sam3d/hf_home','-v',f'{ROOT}:/repo','-v',f'{ROOT}/modules/v2d_sam3d/lib:/workspace/v2d_sam3d/lib:ro','v2d_sam3d:blackwell','-m','v2d.sam3d.lib.image_to_mesh','--image_path',prefix+'/mesh_inputs/rgb.png','--mask_path',prefix+'/mesh_inputs/mask.png','--mesh_path',prefix+'/mesh/object.glb','--transform_path',prefix+'/mesh/transform.json','--intrinsics_path',prefix+'/mesh/intrinsics.json','--weights_dir','/repo/data/weights/sam3d','--pointmap_path',prefix+'/mesh_inputs/pointmap.npy','--pointmap_intrinsics_path',prefix+'/mesh_inputs/intrinsics.json','--seed','42']
    cmd[cmd.index('-e'):cmd.index('-e')] = ['-e','XDG_CACHE_HOME=/tmp/v2d-pilot-cache']
    with REPORT.open('a') as f: f.write('\nSTART '+stem+' depth-grounded SAM3D mesh\n`'+' '.join(cmd)+'`\n')
    with (work/'sam3d.log').open('a') as log: result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        excerpt='\n'.join((work/'sam3d.log').read_text(errors='replace').splitlines()[-20:])
        with REPORT.open('a') as f: f.write(f'\nFAILED {stem} SAM3D exit {result.returncode}\n```\n{excerpt}\n```\n')
        raise SystemExit(result.returncode)
    transform=json.loads((mesh/'transform.json').read_text()); scales=np.asarray(transform['scale'],float)
    assert scales.shape==(3,) and np.isfinite(scales).all() and (scales>0).all()
    assert np.allclose(scales,scales[0],rtol=1e-4), 'Nonuniform scale requires explicit contract handling'
    scene=trimesh.load(mesh/'object.glb',force='scene'); scene.apply_scale(float(scales[0])); scene.export(mesh/'object_metric.glb')
    loaded=trimesh.load(mesh/'object_metric.glb',force='mesh'); assert len(loaded.faces) and np.isfinite(loaded.vertices).all() and loaded.area>0
    with REPORT.open('a') as f: f.write(f'\nCOMPLETE {stem} mesh; applied saved scale {scales.tolist()}, extents metres {loaded.extents.tolist()}\n')
    print('MESH_COMPLETE',stem,loaded.extents.tolist(),flush=True)
