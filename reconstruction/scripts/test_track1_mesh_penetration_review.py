"""Compare containment and depth directly with the released PEN calculation."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'modules/v2d_cari4d/lib/cari4d'))
from lib_mhr.mesh_penetration_review import MeshPenetrationReview
spec = importlib.util.spec_from_file_location('released_pen', root/'v2d_submission_kit/v2d_submission_kit/metric_code/track_1/PEN.py')
released = importlib.util.module_from_spec(spec); spec.loader.exec_module(released)


class PenetrationReviewTests(unittest.TestCase):
    def compare(self, mesh, points):
        expected = released._penetration_depth(points, mesh.vertices, mesh.faces)
        actual = MeshPenetrationReview(mesh.vertices, mesh.faces).depths(points).cpu().numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-7)
        return actual

    def test_inside_outside_and_reversed_winding(self):
        mesh = trimesh.creation.box(extents=[2., 2., 2.])
        points = np.array([[0., 0., 0.], [.9, 0., 0.], [2., 0., 0.], [0., 0., 1.1]])
        np.testing.assert_allclose(self.compare(mesh, points), [1., .1, 0., 0.], atol=1e-6)
        mesh.faces = mesh.faces[:, ::-1]
        self.compare(mesh, points)

    def test_open_mesh_uses_generalized_winding_not_watertight_assumption(self):
        mesh = trimesh.creation.box(extents=[2., 2., 2.])
        mesh.update_faces(np.arange(len(mesh.faces)) >= 2)
        self.compare(mesh, np.random.default_rng(42).uniform(-1.2, 1.2, size=(32, 3)))

    def test_scale_and_degenerate_triangles(self):
        mesh = trimesh.creation.box(extents=[.2, .4, .6])
        mesh.faces = np.concatenate([mesh.faces, [[0, 0, 0]]])
        self.compare(mesh, np.array([[.01, .02, .03], [0., .3, 0.]]))

    def test_real_dense_meshes_match_released_metric_near_surfaces(self):
        base = root/'data/v2d_challenge/track_1/cdo_v5_experiments/pose_candidates'
        for episode in (23, 18, 16, 13):
            with self.subTest(episode=episode):
                stem = f'episode_{episode:06d}'
                mesh = trimesh.load(base/stem/(stem+'_object.glb'), force='mesh', process=False)
                rng = np.random.default_rng(episode)
                indices = np.linspace(0, len(mesh.vertices)-1, 16).round().astype(int)
                surface = mesh.vertices[indices] + rng.normal(size=(16, 3))*1e-6
                interior = rng.uniform(mesh.bounds[0], mesh.bounds[1], size=(16, 3))
                self.compare(mesh, np.concatenate([surface, interior]))


if __name__ == '__main__':
    unittest.main()
