"""Exercise temporal selection with asymmetric evidence and an exhaustive oracle."""
import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'modules/v2d_cari4d/lib/cari4d'))
from prep.temporal_pose_selection import circular_hoop_rotations, project_rotations, select_temporal_rotations


class TemporalSelectionTests(unittest.TestCase):
    def setUp(self):
        self.identity = np.eye(3)
        self.flip = np.diag([-1., -1., 1.])

    def test_asymmetric_evidence_overrides_initial_flip(self):
        rotations = np.tile(np.stack([self.identity, self.flip]), (7, 1, 1, 1))
        costs = np.tile([0., .8], (7, 1))
        costs[3] = [.2, 0.]  # One ambiguous/occluded frame should not flip.
        chosen, path, _ = select_temporal_rotations(rotations, costs)
        np.testing.assert_array_equal(path, np.zeros(7))
        np.testing.assert_allclose(chosen, np.tile(self.identity, (7, 1, 1)))

    def test_path_matches_exhaustive_search(self):
        rotations = np.tile(np.stack([self.identity, self.flip]), (4, 1, 1, 1))
        costs = np.array([[.9, 0.], [.2, 0.], [0., .5], [0., .8]])
        weight = .002
        _, path, objective = select_temporal_rotations(rotations, costs, motion_weight=weight)
        brute = []
        for choices in itertools.product(range(2), repeat=4):
            value = sum(costs[t, c] for t, c in enumerate(choices))
            value += sum(weight * 81. for t in range(1, 4) if choices[t] != choices[t-1])
            brute.append((value, choices))
        optimum = min(brute)
        self.assertAlmostEqual(objective, optimum[0])
        self.assertEqual(tuple(path), optimum[1])

    def test_projection_and_invalid_rotation(self):
        projected = project_rotations(np.diag([1., 1., -1.]))
        self.assertAlmostEqual(np.linalg.det(projected), 1.)
        with self.assertRaises(ValueError):
            select_temporal_rotations(np.zeros((2, 1, 3, 3)), np.zeros((2, 1)))

    def test_circular_hoop_spin_and_plane_sign_do_not_jump(self):
        rotations = np.stack([self.identity, np.diag([1., -1., -1.]),
                              np.diag([-1., 1., -1.]), self.identity])
        result = circular_hoop_rotations(rotations)
        np.testing.assert_allclose(result, np.tile(self.identity, (4, 1, 1)), atol=1e-9)

    def test_translation_penalty_prevents_switch_to_distant_hypothesis(self):
        rotations = np.tile(self.identity, (3, 2, 1, 1))
        translations = np.zeros((3, 2, 3)); translations[:, 1, 2] = 1.
        costs = np.array([[0., .2], [.1, 0.], [0., .2]])
        _, path, _ = select_temporal_rotations(rotations, costs, translations=translations)
        np.testing.assert_array_equal(path, [0, 0, 0])


if __name__ == '__main__':
    unittest.main()
