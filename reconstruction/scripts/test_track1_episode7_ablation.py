"""Check the ablation's isolation and camera-ray semantics independently."""
import unittest

import numpy as np

from track1_episode7_ablation import translations, verify_bundle


class Episode7AblationTests(unittest.TestCase):
    def test_control_exact_and_source_untouched(self):
        source = np.array([[.2, -.3, 2. + i / 10] for i in range(9)], dtype=np.float32)
        original = source.copy()
        result = translations(source, False, 0., np.eye(4))
        np.testing.assert_array_equal(result, source)
        result[0] = 0
        np.testing.assert_array_equal(source, original)

    def test_depth_shift_preserves_pixels_for_nonidentity_camera(self):
        w2c = np.array([[0., -1., 0., .1], [1., 0., 0., -.2], [0., 0., 1., .5], [0., 0., 0., 1.]])
        source = np.array([[.2, .4, 2.], [.3, .7, 3.]])
        output = translations(source, False, .07, w2c)
        before = source @ w2c[:3, :3].T + w2c[:3, 3]
        after = output @ w2c[:3, :3].T + w2c[:3, 3]
        np.testing.assert_allclose(after[:, :2] / after[:, 2, None], before[:, :2] / before[:, 2, None], atol=1e-12)
        np.testing.assert_allclose(after[:, 2] - before[:, 2], .07)

    def test_smoothing_preserves_quadratic_including_boundaries(self):
        t = np.arange(11, dtype=float)
        source = np.stack([.01 * t**2, .1 * t, 2. + .02 * t**2], axis=1)
        np.testing.assert_allclose(translations(source, True, 0., np.eye(4)), source, atol=1e-12)

    def test_invalid_depth_and_short_track_rejected(self):
        with self.assertRaises(ValueError):
            translations(np.array([[0., 0., .12]]), False, -.03, np.eye(4))
        with self.assertRaises(ValueError):
            translations(np.ones((6, 3)), True, 0., np.eye(4))

    def test_rotation_or_human_change_rejected(self):
        control = {'pose': np.zeros((8, 2)), 'object_rotation': np.tile(np.eye(3), (8, 1, 1)), 'object_translation': np.ones((8, 3))}
        for key in ('pose', 'object_rotation'):
            candidate = {k: v.copy() for k, v in control.items()}
            candidate[key].flat[0] += 1
            with self.assertRaises(AssertionError):
                verify_bundle(control, candidate)


if __name__ == '__main__':
    unittest.main()
