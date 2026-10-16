import unittest
import numpy as np
from track1_whittaker import whittaker_smooth, scored_acceleration_magnitude


class WhittakerTests(unittest.TestCase):
    def test_banded_matches_dense_pinned_and_weighted(self):
        rng = np.random.default_rng(7); y = rng.normal(size=(23, 4)); w = rng.uniform(.1, 3, 23)
        d = np.diff(np.eye(23), n=2, axis=0)
        for pin in [None, 0, 1, 12, 22]:
            a = np.diag(w)+30*d.T@d; b = w[:, None]*y
            if pin is not None:
                b -= a[:, pin, None]*y[pin]; a[:, pin] = 0; a[pin] = 0; a[pin, pin] = 1; b[pin] = y[pin]
            actual, _ = whittaker_smooth(y, w, 30, pin)
            np.testing.assert_allclose(actual, np.linalg.solve(a, b), atol=1e-11)

    def test_linear_nullspace_and_exact_pin(self):
        y = np.stack([np.arange(100)*.1, np.arange(100)*-.3], axis=1)
        for robust in [0, 3]:
            x, _ = whittaker_smooth(y, np.ones(100), 1000, 37, robust)
            np.testing.assert_allclose(x, y, atol=1e-9)
            np.testing.assert_array_equal(x[37], y[37])

    def test_robust_suppresses_outlier_and_constant_is_finite(self):
        y = np.zeros((101, 1)); y[50] = 10
        plain, _ = whittaker_smooth(y, np.ones(101), 100)
        robust, _ = whittaker_smooth(y, np.ones(101), 100, robust_iters=3)
        self.assertLess(abs(robust[49, 0]), abs(plain[49, 0])*.1)
        x, _ = whittaker_smooth(np.ones((20, 3)), np.ones(20), 300, robust_iters=3)
        np.testing.assert_allclose(x, 1., atol=1e-10)

    def test_endpoint_weighting_and_gap_semantics(self):
        # Each 4-frame stretch has two accelerations replicated at the ends.
        x = np.zeros((12, 3)); x[:4, 0] = [0, 0, 1, 5]; x[8:, 0] = [0, 0, 2, 10]
        self.assertAlmostEqual(scored_acceleration_magnitude(x, np.array([0,1,2,3,8,9,10,11])), 300.)
        with self.assertRaises(ValueError): scored_acceleration_magnitude(x, np.array([0,1,4,5,6]))

    def test_float32_pin_and_invalid_inputs(self):
        y = np.random.default_rng(1).normal(size=(10, 3)).astype('float32')
        x, _ = whittaker_smooth(y, np.ones(10), 10, 3, 3)
        self.assertEqual(x.dtype, y.dtype); self.assertEqual(x[3].tobytes(), y[3].tobytes())
        with self.assertRaises(ValueError): whittaker_smooth(y, np.zeros(10), 10)

    def test_matches_released_acc_magnitude_with_unequal_stretches(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'v2d_submission_kit/v2d_submission_kit'))
        from v2dlb.mhr_metrics import acceleration_error_per_frame
        rng=np.random.default_rng(42); x=rng.normal(size=(18,22,3))
        runs=[np.arange(1,6),np.arange(9,18)]
        oracle=np.concatenate([acceleration_error_per_frame(x[r],np.zeros_like(x[r])) for r in runs]).mean()*100
        self.assertAlmostEqual(scored_acceleration_magnitude(x,np.concatenate(runs)),oracle)

    def test_slow_motion_preserved_and_fast_noise_attenuated(self):
        t=np.arange(600); slow=np.sin(2*np.pi*t/60); y=(slow+.2*(-1.)**t)[:,None]
        x,_=whittaker_smooth(y,np.ones(len(y)),100)
        interior=slice(60,-60)
        amplitude=np.dot(x[interior,0],slow[interior])/np.dot(slow[interior],slow[interior])
        self.assertGreater(amplitude,.95)
        self.assertLess(np.std(x[interior,0]-slow[interior]),.02)


if __name__ == '__main__': unittest.main()
