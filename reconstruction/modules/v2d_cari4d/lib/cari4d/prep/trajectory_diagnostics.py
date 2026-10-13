"""Observed trajectory proxies; these are not hidden-reference ACC metrics."""
import numpy as np


def rigid_motion_diagnostics(poses, vertices, max_vertices=1024):
    poses=np.asarray(poses,dtype=np.float64)
    vertices=np.asarray(vertices,dtype=np.float64)
    if poses.ndim!=3 or poses.shape[1:]!=(4,4) or len(poses)<3 or not np.isfinite(poses).all():
        raise ValueError('Expected at least three finite 4x4 poses')
    if vertices.ndim!=2 or vertices.shape[1]!=3 or not len(vertices) or not np.isfinite(vertices).all() or max_vertices<=0:
        raise ValueError('Expected finite nonempty mesh vertices and a positive sample limit')
    rotations=poses[:,:3,:3]
    relative=np.transpose(rotations[:-1],(0,2,1))@rotations[1:]
    cosine=np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1)
    angles=np.rad2deg(np.arccos(cosine))
    sampled=vertices[::max(1,int(np.ceil(len(vertices)/max_vertices)))]
    points=np.einsum('tij,pj->tpi',rotations,sampled)+poses[:,:3,3,None].transpose(0,2,1)
    acceleration=np.linalg.norm(np.diff(points,n=2,axis=0),axis=-1)
    return {'rotation_step_degrees_mean_proxy':float(angles.mean()),
            'rotation_step_degrees_p95_proxy':float(np.quantile(angles,.95)),
            'rotation_step_degrees_max_proxy':float(angles.max()),
            'rotation_jumps_over_45_degrees':int((angles>45).sum()),
            'sampled_vertex_acceleration_cm_proxy':float(acceleration.mean()*100),
            'sampled_vertex_count':len(sampled)}
