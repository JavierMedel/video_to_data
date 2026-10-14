import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'modules/v2d_cari4d/lib/cari4d'))
from prep.circular_hoop_fit import fit_circle_rays, frame_from_normal


class CircleFitTests(unittest.TestCase):
    def test_metric_circle_recovered_from_perspective_rays(self):
        center = np.array([.15, -.2, 3.])
        normal = np.array([.6, .1, .7937253933193772]); normal /= np.linalg.norm(normal)
        frame = frame_from_normal(normal)
        angles = np.linspace(0., 2*np.pi, 240, endpoint=False)
        points = center + .384 * (np.cos(angles)[:, None]*frame[:, 1] + np.sin(angles)[:, None]*frame[:, 2])
        rays = points / points[:, 2:3]
        fitted_center, fitted_normal, error = fit_circle_rays(rays, .384, center + [.02, .01, .15], normal + [.05, .02, -.01], focal_pixels=1200.)
        np.testing.assert_allclose(fitted_center, center, atol=1e-5)
        self.assertGreater(abs(normal @ fitted_normal), .999999)
        self.assertLess(error, 1e-4)

    def test_invalid_rays_rejected(self):
        with self.assertRaises(ValueError):
            fit_circle_rays(np.zeros((3, 3)), .4, [0, 0, 3], [0, 0, 1])


if __name__ == '__main__':
    unittest.main()
