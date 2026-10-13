"""Translation stability must not hide rotation flips in rigid motion reviews."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'modules/v2d_cari4d/lib/cari4d'))
from prep.trajectory_diagnostics import rigid_motion_diagnostics


class MotionDiagnosticTests(unittest.TestCase):
    def poses(self,angles):
        poses=np.repeat(np.eye(4)[None],len(angles),axis=0)
        for i,angle in enumerate(np.deg2rad(angles)):
            poses[i,:3,:3]=[[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]
        return poses

    def test_rotation_flip_is_visible_with_constant_translation(self):
        d=rigid_motion_diagnostics(self.poses([0,0,180,180]),[[1,0,0]])
        self.assertEqual(d['rotation_jumps_over_45_degrees'],1)
        self.assertAlmostEqual(d['rotation_step_degrees_max_proxy'],180)
        self.assertGreater(d['sampled_vertex_acceleration_cm_proxy'],0)

    def test_constant_linear_motion_has_zero_surface_acceleration(self):
        poses=self.poses([0]*5); poses[:,0,3]=np.arange(5)*.05
        d=rigid_motion_diagnostics(poses,[[1,0,0],[0,1,0]])
        self.assertEqual(d['rotation_jumps_over_45_degrees'],0)
        self.assertAlmostEqual(d['sampled_vertex_acceleration_cm_proxy'],0)

    def test_smooth_rotation_is_not_classified_as_a_flip(self):
        d=rigid_motion_diagnostics(self.poses([0,10,20,30]),[[1,0,0]])
        self.assertEqual(d['rotation_jumps_over_45_degrees'],0)
        self.assertAlmostEqual(d['rotation_step_degrees_mean_proxy'],10)


if __name__=='__main__': unittest.main()
