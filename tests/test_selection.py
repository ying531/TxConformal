import unittest

import numpy as np

from txconformal.selection import (
    estimate_topk,
    estimate_topk_fp,
    fp_budget_threshold,
    tp_min_threshold,
)


class TopKEstimateTests(unittest.TestCase):
    def test_count_and_fdp_scales_are_explicit(self):
        p_values = np.array([0.04, 0.01, 0.03, 0.02])

        estimate = estimate_topk(p_values, K=2)

        self.assertEqual(estimate.k, 2)
        self.assertEqual(estimate.m, 4)
        self.assertAlmostEqual(estimate.p_boundary, 0.02)
        self.assertAlmostEqual(estimate.fp_count, 0.08)
        self.assertAlmostEqual(estimate.fdp, 0.04)
        self.assertAlmostEqual(estimate_topk_fp(p_values, K=2), 0.04)

    def test_invalid_k_is_rejected_instead_of_silently_truncated(self):
        with self.assertRaises(ValueError):
            estimate_topk(np.array([0.1, 0.2]), K=3)

    def test_deprecated_generic_interval_is_rejected(self):
        with self.assertRaises(NotImplementedError):
            estimate_topk_fp(np.array([0.1, 0.2]), K=1, CI=True)

    def test_integer_valued_float_k_remains_backward_compatible(self):
        p_values = np.array([0.01, 0.02, 0.50])
        self.assertEqual(estimate_topk(p_values, K=2.0).k, 2)
        self.assertEqual(fp_budget_threshold(p_values, K=1.0)[1].size, 2)
        self.assertGreaterEqual(tp_min_threshold(p_values, K=1.0)[1].size, 1)

    def test_fractional_k_is_rejected(self):
        with self.assertRaises(TypeError):
            estimate_topk(np.array([0.1, 0.2]), K=1.5)


if __name__ == "__main__":
    unittest.main()
