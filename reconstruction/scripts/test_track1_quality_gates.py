"""Regression checks for observed catastrophic tracker acceptance."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'modules/v2d_cari4d/lib/cari4d'))
from prep.quality_gate_utils import assess_candidates

class QualityGateTests(unittest.TestCase):
    def metrics(self,iou):
        a=np.array(iou,float)
        return SimpleNamespace(observed_visible_pixels=100,iou=a,centroid_distance=np.zeros(len(a)),rendered_visibility=np.ones(len(a)),precision=np.ones(len(a)),recall=np.ones(len(a)))
    def poses(self,depths):
        p=np.repeat(np.eye(4)[None],len(depths),axis=0); p[:,2,3]=depths; return p
    def test_iron_depth_divergence_is_rejected(self):
        accepted,_=assess_candidates(self.poses([29.28,3.0]),self.metrics([0.8,0.8]),3.0,0.3)
        np.testing.assert_array_equal(accepted,[False,True])
    def test_zero_overlap_is_not_reliable(self):
        accepted,_=assess_candidates(self.poses([3.0]),self.metrics([0.0]),3.0,0.3)
        self.assertFalse(accepted[0])
    def test_behind_camera_prediction_is_rejected(self):
        accepted,_=assess_candidates(self.poses([-0.06]),self.metrics([0.9]),3.0,0.3)
        self.assertFalse(accepted[0])
    def test_jump_requires_reacquisition(self):
        previous=self.poses([3.0])[0]; candidate=self.poses([3.0]); candidate[0,0,3]=1.0
        tracked,_=assess_candidates(candidate,self.metrics([0.9]),3.0,0.3,previous)
        reacquired,_=assess_candidates(candidate,self.metrics([0.9]),3.0,0.3,None)
        self.assertFalse(tracked[0]); self.assertTrue(reacquired[0])

if __name__=='__main__': unittest.main()
