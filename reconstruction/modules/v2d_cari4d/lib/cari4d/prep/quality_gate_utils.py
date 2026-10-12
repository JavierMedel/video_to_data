"""Observed image/depth gates independent of FoundationPose model scores."""
import numpy as np

def assess_candidates(poses,metrics,expected_depth,diameter,previous=None,min_iou=0.15):
    poses=np.asarray(poses)
    positive=poses[:,2,3]>max(0.05,diameter*0.15)
    depth_error=np.abs(poses[:,2,3]-expected_depth)
    depth_ok=depth_error<=max(0.25,diameter)
    observed_ok=metrics.observed_visible_pixels>=8
    silhouette_ok=(metrics.iou>=min_iou)&(metrics.centroid_distance<=0.1)&(metrics.rendered_visibility>=0.1)
    motion_ok=np.ones(len(poses),bool) if previous is None else np.linalg.norm(poses[:,:3,3]-previous[:3,3],axis=1)<=max(0.15,diameter*0.3)
    accepted=positive&depth_ok&observed_ok&silhouette_ok&motion_ok
    score=metrics.iou+0.25*metrics.precision+0.25*metrics.recall-0.1*depth_error/max(diameter,0.1)
    return accepted,np.where(accepted,score,-np.inf)
