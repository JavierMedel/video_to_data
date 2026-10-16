import unittest
import numpy as np
from track1_review_humans import humans_identical, decode_bundle_human, framewise_contact_pass


class HumanReviewTests(unittest.TestCase):
    def test_each_panel_decodes_its_actual_pose_and_identity(self):
        class Model:
            def run(self, pose, identity): return pose+identity.sum(), pose
        a={'pose':np.zeros((4,3)), 'scales':np.ones(2), 'shape':np.ones(1)}
        b={k:v.copy() for k,v in a.items()}; b['pose'][1,0]=7
        self.assertFalse(humans_identical(a,b))
        x,_=decode_bundle_human(Model(),a,[1],np.asarray)
        y,_=decode_bundle_human(Model(),b,[1],np.asarray)
        self.assertEqual(y[0,0]-x[0,0],7.)
        self.assertTrue(humans_identical(a,{k:v.copy() for k,v in a.items()}))

    def test_episode_gain_cannot_hide_one_worse_frame_or_fps_subset(self):
        a={'per_frame_full_hand_depth_cm':[1.,1.], 'per_frame_fps512_depth_cm':[1.,1.]}
        b={**a,'per_frame_full_hand_depth_cm':[0.,1.1]}
        self.assertEqual(framewise_contact_pass(a,b,[1.,2.],[1.,3.]),(False,True))
        b={**a,'per_frame_fps512_depth_cm':[0.,1.1]}
        self.assertEqual(framewise_contact_pass(a,b,[1.,1.],[1.,1.6]),(False,False))
        self.assertEqual(framewise_contact_pass(a,a,[1.,2.],[1.5,20.]),(True,True))


if __name__=='__main__':unittest.main()
