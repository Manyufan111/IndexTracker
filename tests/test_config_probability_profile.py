"""Tests for probability model profile defaults in runtime config."""

from __future__ import annotations

import unittest

from indextrack.infra.config import load_runtime_config


class ProbabilityProfileConfigTests(unittest.TestCase):
    def test_baseline_profile_restores_quantile_baseline_defaults(self) -> None:
        config = load_runtime_config(
            {
                "INDEXTRACK_PROB_PROFILE": "baseline",
            }
        )
        prob = config.probability_model
        self.assertEqual(prob.profile, "baseline")
        self.assertAlmostEqual(prob.k_60, 0.65)
        self.assertAlmostEqual(prob.lambda_60, 0.70)
        self.assertAlmostEqual(prob.long_high_vol_strong_trend_scale, 1.00)
        self.assertAlmostEqual(prob.display_prob_cap, 0.90)
        self.assertEqual(prob.calibrator_mode, "softmax")
        self.assertEqual(prob.calibrator_mode_5, "softmax")
        self.assertEqual(prob.calibrator_mode_20, "softmax")
        self.assertEqual(prob.calibrator_mode_60, "softmax")
        self.assertAlmostEqual(prob.calibration_blend_60, 1.0)
        self.assertAlmostEqual(prob.calibration_max_shift_60, 1.0)

    def test_optimized_profile_is_default(self) -> None:
        config = load_runtime_config({})
        prob = config.probability_model
        self.assertEqual(prob.profile, "optimized_v1")
        self.assertAlmostEqual(prob.k_60, 0.60)
        self.assertAlmostEqual(prob.long_high_vol_strong_trend_scale, 1.00)
        self.assertAlmostEqual(prob.lambda_20, 1.0)
        self.assertAlmostEqual(prob.lambda_60, 1.0)
        self.assertAlmostEqual(prob.display_prob_cap, 0.90)
        self.assertEqual(prob.calibrator_mode, "conservative")
        self.assertEqual(prob.calibrator_mode_5, "conservative")
        self.assertEqual(prob.calibrator_mode_20, "none")
        self.assertEqual(prob.calibrator_mode_60, "none")
        self.assertAlmostEqual(prob.calibration_blend_20, 0.0)
        self.assertAlmostEqual(prob.calibration_blend_60, 0.0)
        self.assertAlmostEqual(prob.calibration_max_shift_20, 0.0)
        self.assertAlmostEqual(prob.calibration_max_shift_60, 0.0)

    def test_horizon_calibration_mode_env_override(self) -> None:
        config = load_runtime_config(
            {
                "INDEXTRACK_PROB_PROFILE": "optimized_v1",
                "INDEXTRACK_PROB_CALIB_MODE_20": "conservative",
            }
        )
        prob = config.probability_model
        self.assertEqual(prob.calibrator_mode_20, "conservative")
        self.assertEqual(prob.calibrator_mode_60, "none")


if __name__ == "__main__":
    unittest.main()
