import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LinearRegression

from txconformal import TxConformal
from txconformal.conformal.pvalues import weighted_pvalues_for_selection


class CandidateWeightTests(unittest.TestCase):
    def setUp(self):
        self.prov = SimpleNamespace(
            f_calib=np.array([0.15, 0.35, 0.65, 0.85]),
            f_test=np.array([0.25, 0.55, 0.75]),
            bins_calib=np.zeros((4, 0)),
            E_calib=None,
        )
        self.y_calib = np.array([0.0, 0.0, 1.0, 0.0])
        self.w_cal = np.array([0.5, 1.0, 1.5, 1.0])
        self.calib_x = np.column_stack(
            [np.ones(4), np.array([-1.0, -0.25, 0.25, 1.0])]
        )
        self.test_x = np.column_stack(
            [np.ones(3), np.array([-0.75, 0.0, 0.75])]
        )

    def test_log_linear_candidate_weights_reach_pvalue_core(self):
        txc = TxConformal(score_name="clip")
        eb_meta = {"calib_x": self.calib_x, "test_x": self.test_x}

        with patch.object(txc, "_weights_from_provider", return_value=(self.w_cal, eb_meta)):
            txc.fit(
                self.prov,
                self.y_calib,
                cutoff=0.5,
                candidate_weight_mode="log_linear",
            )

        normalized_w_cal = self.w_cal / np.mean(self.w_cal)
        model = LinearRegression().fit(self.calib_x, np.log(normalized_w_cal))
        expected_w_test = np.exp(model.predict(self.test_x))
        v_cal, _, v_test, _ = txc._compute_scores(
            self.prov.f_calib,
            self.y_calib,
            self.prov.f_test,
            cutoff=0.5,
        )
        expected_p = weighted_pvalues_for_selection(
            v_cal, normalized_w_cal, v_test, w_test=expected_w_test
        )

        np.testing.assert_allclose(txc._weights_test, expected_w_test)
        np.testing.assert_allclose(txc._p_sel, expected_p)

        result = txc.select(method="top_k", K=2)
        expected_boundary = np.sort(expected_p)[1]
        self.assertAlmostEqual(result.fp_count_est, expected_p.size * expected_boundary)
        self.assertAlmostEqual(result.fdp_est, result.fp_count_est / 2)
        self.assertIsNone(result.fdp_est_CI)
        np.testing.assert_allclose(result.weights_test, expected_w_test)

    def test_explicit_candidate_weights_take_precedence(self):
        txc = TxConformal(score_name="clip")
        explicit = np.array([0.25, 1.0, 4.0])

        with patch.object(
            txc,
            "_weights_from_provider",
            return_value=(self.w_cal, {"calib_x": self.calib_x, "test_x": self.test_x}),
        ):
            txc.fit(
                self.prov,
                self.y_calib,
                cutoff=0.5,
                candidate_weight_mode="uniform",
                w_test=explicit,
            )

        np.testing.assert_array_equal(txc._weights_test, explicit)
        self.assertEqual(txc._meta["candidate_weight_meta"]["mode"], "provided")


if __name__ == "__main__":
    unittest.main()
