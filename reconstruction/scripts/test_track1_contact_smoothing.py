import itertools
import unittest

import numpy as np

from track1_contact_smoothing import feasible_options, objective, select_safe_path, summarize_selected_contacts, scored_motion_diagnostics


class ContactSmoothingTests(unittest.TestCase):
    def test_scored_motion_does_not_bridge_unscored_gaps(self):
        t = np.arange(8, dtype=float)
        translations = np.stack([t*t, np.zeros(8), np.ones(8)], axis=1)
        result = scored_motion_diagnostics(np.tile(np.eye(3), (8, 1, 1)), translations, np.zeros((1, 3)), np.array([0, 1, 2, 5, 6, 7]))
        self.assertEqual(result['consecutive_scored_triples'], 2)
        self.assertAlmostEqual(result['translation_acceleration_cm_proxy'], 200.)
        self.assertAlmostEqual(result['sampled_vertex_acceleration_cm_proxy'], 200.)

    def test_no_consecutive_scored_triple_is_unavailable(self):
        result = scored_motion_diagnostics(np.tile(np.eye(3), (5, 1, 1)), np.ones((5, 3)), np.zeros((1, 3)), np.array([0, 2, 4]))
        self.assertEqual(result['consecutive_scored_triples'], 0)
        self.assertIsNone(result['translation_acceleration_cm_proxy'])

    def test_cached_review_selects_exact_rows_and_preserves_scope(self):
        full = np.array([[.1, .01], [.2, .02], [.3, .03]])
        result = summarize_selected_contacts(full, full/2, np.zeros((3, 2)), np.ones((3, 2)), np.array([0, 1, 1]), np.array([2, 0]))
        self.assertAlmostEqual(result['summary']['candidate']['mean_full_hand_depth_cm_unaligned'], .065)
        self.assertTrue(result['hand_penetration_gate_pass'])
        self.assertIsNone(result['summary']['candidate']['max_hand_vertex_depth_cm'])
        with self.assertRaises(ValueError):
            summarize_selected_contacts(full, full, full, np.zeros((3, 2)), np.zeros(3, dtype=int), np.array([0]))

    def test_matches_exhaustive_feasible_oracle(self):
        rng = np.random.default_rng(9)
        options = rng.normal(size=(6, 2, 3))
        allowed = np.ones((6, 2), dtype=bool); allowed[2, 1] = False
        rotations = np.tile(np.eye(3), (6, 1, 1)); centroid = np.array([.1, .2, -.3])
        path, value = select_safe_path(options, allowed, rotations, centroid)
        oracle = min((objective(options, rotations, centroid, np.array(p)), p)
                     for p in itertools.product(range(2), repeat=6) if all(allowed[t, v] for t, v in enumerate(p)))
        self.assertAlmostEqual(value, oracle[0])
        self.assertEqual(tuple(path), oracle[1])

    def test_each_hand_constraint_and_bound_can_reject(self):
        options = np.ones((10, 2, 3)); options[:, :, 2] = 3.
        full = np.zeros((10, 2)); fps = full.copy(); proximity = full.copy()
        full[2, 1] = .001
        fps[3, 1] = .001
        proximity[4, 1] = .6
        options[5, 1, 0] += .031
        mask = feasible_options(options, full, fps, proximity)
        np.testing.assert_array_equal(mask[:, 0], True)
        np.testing.assert_array_equal(mask[2:6, 1], False)
        self.assertTrue(mask[6, 1])
        self.assertFalse(mask[0, 1]); self.assertFalse(mask[-1, 1])

    def test_surface_centroid_objective_has_constant_rotational_remainder(self):
        rng = np.random.default_rng(11)
        vertices = rng.normal(size=(9, 3))
        rotations = np.stack([np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]) for a in np.arange(5)*.4])
        options = rng.normal(size=(5, 2, 3))
        remainders = []
        for p in [np.zeros(5, dtype=int), np.ones(5, dtype=int)]:
            world = np.einsum('tij,vj->tvi', rotations, vertices) + options[np.arange(5), p, None]
            surface = np.square(np.diff(world, n=2, axis=0)).sum(axis=2).mean(axis=1).sum()
            remainders.append(surface - objective(options, rotations, vertices.mean(axis=0), p, 0.))
        self.assertAlmostEqual(*remainders)

    def test_control_path_bounds_objective_and_unsafe_path_is_excluded(self):
        t = np.arange(9)
        options = np.zeros((9, 2, 3)); options[:, :, 2] = 3.
        options[:, 0, 0] = (-1.)**t * .01
        allowed = np.ones((9, 2), dtype=bool); allowed[4, 1] = False
        rotations = np.tile(np.eye(3), (9, 1, 1))
        path, value = select_safe_path(options, allowed, rotations, np.zeros(3))
        self.assertEqual(path[4], 0)
        self.assertLess(value, objective(options, rotations, np.zeros(3), np.zeros(9, dtype=int)))

    def test_no_feasible_frame_is_rejected(self):
        with self.assertRaises(ValueError):
            select_safe_path(np.zeros((3, 2, 3)), np.zeros((3, 2), dtype=bool), np.tile(np.eye(3), (3, 1, 1)), np.zeros(3))


if __name__ == '__main__':
    unittest.main()
