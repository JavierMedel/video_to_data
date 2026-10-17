"""Standalone mask-only raster helpers, without loading the tracking pipeline.

Same grid/projection/raster equations as prep.run_foundationpose_mhr_export;
keep tracking/model initialization imports out of artifact-only reviews.
"""
import numpy as np
import torch
import nvdiffrast.torch as dr
import Utils
from prep.foundationpose_pose_selection import validate_rigid_transforms


def _scaled_image_grid(K,image_shape,max_side):
    height,width=image_shape;scale=min(1.,float(max_side)/max(height,width))
    shape=(max(1,int(round(height*scale))),max(1,int(round(width*scale))))
    k=np.asarray(K,dtype=np.float32).copy();k[0]*=shape[1]/width;k[1]*=shape[0]/height
    return k,shape


@torch.no_grad()
def _render_candidate_masks(estimator,candidate_poses_centered,K,image_shape,*,chunk_size=8):
    candidates=validate_rigid_transforms(candidate_poses_centered,'rendered mask candidates')
    height,width=map(int,image_shape)
    if height<=0 or width<=0 or chunk_size<=0:raise ValueError('Invalid raster shape/chunk')
    device=estimator.mesh_tensors['pos'].device
    projection=torch.as_tensor(Utils.projection_matrix_from_intrinsics(K,height=height,width=width,znear=.001,zfar=100.),dtype=torch.float32,device=device)
    cv_to_gl=torch.as_tensor(Utils.glcam_in_cvcam,dtype=torch.float32,device=device)
    homogeneous=Utils.to_homo_torch(estimator.mesh_tensors['pos']);masks=[]
    for begin in range(0,len(candidates),chunk_size):
        pose=torch.as_tensor(candidates[begin:begin+chunk_size],dtype=torch.float32,device=device)
        clip=((projection[None]@cv_to_gl[None]@pose)[:,None]@homogeneous[None,...,None])[...,0]
        raster,_=dr.rasterize(estimator.glctx,clip,estimator.mesh_tensors['faces'],resolution=(height,width))
        masks.append(torch.flip(raster[...,3]>0,dims=[1]).cpu().numpy().astype(bool))
    return np.concatenate(masks,axis=0)
