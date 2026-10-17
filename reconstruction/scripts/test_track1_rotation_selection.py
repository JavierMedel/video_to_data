import itertools
import unittest
import numpy as np
from track1_rotation_selection import orientation_proposals,rotation_path


class RotationTests(unittest.TestCase):
    def test_proper_rotations_and_exact_original(self):
        r=np.tile(np.eye(3,dtype='float32'),(7,1,1));p=orientation_proposals(r)
        np.testing.assert_array_equal(p[:,0],r)
        np.testing.assert_allclose(p@p.swapaxes(-1,-2),np.broadcast_to(np.eye(3),p.shape),atol=2e-7)
        np.testing.assert_allclose(np.linalg.det(p),1,atol=2e-7)

    def test_dp_matches_exhaustive_and_cannot_choose_unsafe(self):
        r=np.tile(np.eye(3),(4,1,1));p=orientation_proposals(r)[:,[0,1,2]]
        iou=np.array([[.4,.8,.6],[.5,.7,.8],[.3,.6,.7],[.5,.6,.8]])
        allowed=np.ones((4,3),bool);allowed[2,2]=False
        path=rotation_path(p,iou,allowed)
        def score(seq):
            value=0.
            for t,k in enumerate(seq):
                angle=np.rad2deg(np.arccos(np.clip((np.sum(p[t,0]*p[t,k])-1)/2,-1,1)))
                value+=1-iou[t,k]+.001*(angle/20)**2
                if t:
                    angle=np.rad2deg(np.arccos(np.clip((np.sum(p[t-1,seq[t-1]]*p[t,k])-1)/2,-1,1)))
                    value+=.01*(angle/20)**2
            return value
        feasible=[s for s in itertools.product(range(3),repeat=4) if all(allowed[t,k] for t,k in enumerate(s))]
        self.assertAlmostEqual(score(path),min(map(score,feasible)))
        self.assertFalse(path[2]==2)


if __name__=='__main__':unittest.main()
