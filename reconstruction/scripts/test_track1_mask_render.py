"""GPU parity with exact source-extracted upstream helper, without tracker imports."""

def main():
    import ast
    import __future__
    from pathlib import Path
    import sys
    from types import SimpleNamespace
    import numpy as np
    import torch
    import nvdiffrast.torch as dr
    root=Path('/repo');sys.path[:0]=['/workspace/v2d_foundation_pose/lib/FoundationPose',str(root/'modules/v2d_cari4d/lib/cari4d')]
    import Utils
    from prep.foundationpose_pose_selection import validate_rigid_transforms
    from track1_mask_render import _render_candidate_masks,_scaled_image_grid
    import trimesh
    source=(root/'modules/v2d_cari4d/lib/cari4d/prep/run_foundationpose_mhr_export.py').read_text();tree=ast.parse(source)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_scaled_image_grid','_render_candidate_masks','_raster_alpha_to_opencv_mask')]
    scope={'np':np,'torch':torch,'dr':dr,'Utils':Utils,'validate_rigid_transforms':validate_rigid_transforms,'_tensor_to_numpy':lambda x:x.detach().cpu().numpy()}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<upstream helpers>', 'exec',flags=__future__.annotations.compiler_flag),scope)
    k=np.array([[80.,0,48],[0,80.,32],[0,0,1.]])
    mesh=trimesh.creation.box();renderer=SimpleNamespace(mesh_tensors={'pos':torch.as_tensor(mesh.vertices,device='cuda',dtype=torch.float32),'faces':torch.as_tensor(mesh.faces,device='cuda',dtype=torch.int32)},glctx=dr.RasterizeCudaContext())
    poses=np.tile(np.eye(4),(3,1,1));poses[:,2,3]=[2.,3.,4.]
    for side in [48,96]:
     sk,shape=_scaled_image_grid(k,(64,96),side);rk,rs=scope['_scaled_image_grid'](k,(64,96),side)
     np.testing.assert_array_equal(sk,rk);assert shape==rs
     np.testing.assert_array_equal(_render_candidate_masks(renderer,poses,sk,shape,chunk_size=2),scope['_render_candidate_masks'](renderer,poses,sk,shape,chunk_size=2))
    print('PASS: grid and batched masks match source-extracted upstream helper at both sizes')


if __name__ == "__main__":
    main()
