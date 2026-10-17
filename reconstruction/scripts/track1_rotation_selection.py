"""Small object-local orientation proposals and constrained temporal selection."""
import numpy as np


def orientation_proposals(rotations):
    r=np.asarray(rotations,dtype=float)
    if r.ndim!=3 or r.shape[1:]!=(3,3) or not np.isfinite(r).all(): raise ValueError('Invalid rotation sequence')
    proposals=[r.copy()]
    for degrees in (-20.,-10.,-5.,5.,10.,20.):
        angle=np.deg2rad(degrees);c,s=np.cos(angle),np.sin(angle)
        for axis in range(3):
            m=np.eye(3);i,j=(axis+1)%3,(axis+2)%3;m[i,i]=m[j,j]=c;m[i,j]=-s;m[j,i]=s
            proposals.append(r@m)
    for radius in (2,4):
        means=np.stack([r[max(0,t-radius):t+radius+1].mean(axis=0) for t in range(len(r))])
        u,_,vh=np.linalg.svd(means);sign=np.ones((len(r),3));sign[:,-1]=np.linalg.det(u@vh)
        proposals.append((u*sign[:,None,:])@vh)
    return np.stack(proposals,axis=1).astype(rotations.dtype)


def rotation_path(rotations, ious, allowed, motion_weight=.01):
    r=np.asarray(rotations,dtype=float);ious=np.asarray(ious,dtype=float);allowed=np.asarray(allowed,dtype=bool)
    if r.shape[:2]!=ious.shape or allowed.shape!=ious.shape or not allowed[:,0].all(): raise ValueError('Invalid finite choices')
    if not np.isfinite(r).all() or not np.isfinite(ious).all():raise ValueError('Nonfinite choice')
    costs=1-ious
    cosine=(np.einsum('tij,tkij->tk',r[:,0],r)-1)/2
    costs+=.001*(np.rad2deg(np.arccos(np.clip(cosine,-1,1)))/20.)**2
    costs[~allowed]=1e12
    previous=costs[0].copy();parents=np.zeros(ious.shape,dtype=int)
    for t in range(1,len(r)):
        cosine=(np.einsum('aij,bij->ab',r[t-1],r[t])-1)/2
        penalty=motion_weight*(np.rad2deg(np.arccos(np.clip(cosine,-1,1)))/20.)**2
        transition=previous[:,None]+penalty;parents[t]=transition.argmin(axis=0)
        previous=costs[t]+transition[parents[t],np.arange(r.shape[1])]
    path=np.zeros(len(r),dtype=int);path[-1]=previous.argmin()
    for t in range(len(r)-1,0,-1):path[t-1]=parents[t,path[t]]
    assert allowed[np.arange(len(r)),path].all()
    return path
