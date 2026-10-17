"""Exact second-order finite-option rotation selection with surface acceleration."""
import numpy as np


def surface_objective(rotations,translations,vertices,ious,path,weight=.02):
    selected=rotations[np.arange(len(path)),path]
    points=np.einsum('tij,pj->tpi',selected,vertices)+translations[:,None]
    accel=np.diff(points,n=2,axis=0)
    return float(np.sum(1-ious[np.arange(len(path)),path])+weight*10000*np.square(accel).sum(axis=-1).mean(axis=1).sum())


def select_surface_path(rotations,translations,vertices,ious,allowed,weight=.02):
    r=np.asarray(rotations,dtype=float);t=np.asarray(translations,dtype=float);v=np.asarray(vertices,dtype=float)
    ious=np.asarray(ious,dtype=float);allowed=np.asarray(allowed,dtype=bool)
    if r.ndim!=4 or r.shape[-2:]!=(3,3) or t.shape!=(len(r),3) or ious.shape!=r.shape[:2] or allowed.shape!=ious.shape or len(r)<3 or v.ndim!=2 or v.shape[1]!=3 or not len(v):raise ValueError('Invalid finite surface problem')
    if any(not np.isfinite(x).all() for x in (r,t,v,ious)) or not allowed[:,0].all() or weight<0:raise ValueError('Nonfinite/invalid fallback')
    k=r.shape[1];mu=v.mean(axis=0);moment=v.T@v/len(v);cost=(1-ious).copy();cost[~allowed]=np.inf
    previous=cost[0,:,None]+cost[1,None,:];parents=np.zeros((len(r),k,k),int)
    for frame in range(2,len(r)):
        # Axes are previous-previous, previous, current state.
        delta=r[frame-2,:,None,None]-2*r[frame-1,None,:,None]+r[frame,None,None,:]
        dt=t[frame]-2*t[frame-1]+t[frame-2]
        squared=np.einsum('abcij,jk,abcik->abc',delta,moment,delta)+2*np.einsum('i,abcij,j->abc',dt,delta,mu)+np.dot(dt,dt)
        candidates=previous[:,:,None]+weight*10000*np.maximum(squared,0)
        parents[frame]=candidates.argmin(axis=0)
        previous=np.min(candidates,axis=0)+cost[frame,None,:]
    before,current=np.unravel_index(np.argmin(previous),previous.shape);path=np.zeros(len(r),int);path[-2:]=before,current
    for frame in range(len(r)-1,1,-1):path[frame-2]=parents[frame,path[frame-1],path[frame]]
    assert allowed[np.arange(len(r)),path].all()
    return path
