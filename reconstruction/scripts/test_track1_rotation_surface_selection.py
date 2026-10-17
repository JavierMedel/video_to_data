import itertools
import unittest
import numpy as np
from track1_rotation_surface_selection import select_surface_path,surface_objective
from track1_rotation_selection import orientation_proposals


class SurfaceTests(unittest.TestCase):
    def test_covariance_dp_matches_explicit_surface_exhaustive_oracle(self):
        rng=np.random.default_rng(4);r=orientation_proposals(np.tile(np.eye(3),(5,1,1)))[:,[0,1]]
        t=rng.normal(size=(5,3))*.01;v=rng.normal(size=(17,3))*.1;ious=rng.uniform(.4,.9,(5,2));allowed=np.ones((5,2),bool);allowed[2,1]=False
        path=select_surface_path(r,t,v,ious,allowed)
        values=[surface_objective(r,t,v,ious,np.array(p)) for p in itertools.product(range(2),repeat=5) if all(allowed[i,k] for i,k in enumerate(p))]
        self.assertAlmostEqual(surface_objective(r,t,v,ious,path),min(values),places=10)
        self.assertLessEqual(surface_objective(r,t,v,ious,path),surface_objective(r,t,v,ious,np.zeros(5,int)))


if __name__=='__main__':unittest.main()
