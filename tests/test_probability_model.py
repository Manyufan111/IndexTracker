"""Tests for the calibrated quantile probability model pipeline."""

from __future__ import annotations

import math
import unittest
from datetime import date, datetime, timedelta

import numpy as np

from indextrack.app.models import Candle
from indextrack.domain.probability.calibrator import CalibratorConfig
from indextrack.domain.probability.feature_engineering import ProbabilityFeatureEngine
from indextrack.domain.probability.model import MarketProbabilityModel, MarketProbabilityModelConfig
from indextrack.domain.probability.probability_mapper import ProbabilityMapperConfig
from indextrack.domain.probability.probability_mapper import quantiles_to_raw_probabilities


def _build_synthetic_candles(days: int = 560, seed: int = 7) -> list[Candle]:
    rng = np.random.default_rng(seed)
    rows: list[Candle] = []
    start = date(2024, 1, 2)
    prev_close = 4500.0
    for idx in range(days):
        cyc = 0.0015 * math.sin(idx / 26.0)
        shock = float(rng.normal(0.0, 0.0075))
        daily_ret = 0.00035 + cyc + shock
        close = max(prev_close * (1.0 + daily_ret), 10.0)
        open_price = max(prev_close * (1.0 + float(rng.normal(0.0, 0.003))), 10.0)
        spread = abs(float(rng.normal(0.0, 0.006)))
        high = max(close, open_price) * (1.0 + spread)
        low = min(close, open_price) * (1.0 - spread)
        volume = max(1_000_000.0 * (1.0 + float(rng.normal(0.0, 0.10))), 10_000.0)
        rows.append(
            Candle(
                symbol="SP500",
                trade_date=start + timedelta(days=idx),
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
                source="synthetic",
                fetched_at=datetime(2026, 4, 1, 12, 0, 0),
            )
        )
        prev_close = close
    return rows


class ProbabilityModelTests(unittest.TestCase):
    def test_feature_engine_has_no_leakage(self) -> None:
        candles = _build_synthetic_candles()
        engine = ProbabilityFeatureEngine()
        baseline = engine.transform(candles)

        mutated = list(candles)
        target_idx = 430
        base = mutated[target_idx]
        mutated[target_idx] = Candle(
            symbol=base.symbol,
            trade_date=base.trade_date,
            open=base.open * 1.9,
            high=base.high * 1.9,
            low=base.low * 1.9,
            close=base.close * 1.9,
            volume=base.volume,
            source=base.source,
            fetched_at=base.fetched_at,
        )
        rerun = engine.transform(mutated)

        compare_idx = 220
        np.testing.assert_allclose(
            baseline.values[compare_idx],
            rerun.values[compare_idx],
            rtol=1e-12,
            atol=1e-12,
        )

    def test_predict_probabilities_sum_to_one(self) -> None:
        candles = _build_synthetic_candles()
        model = MarketProbabilityModel().fit(candles)
        outputs = model.predict_proba(candles)
        for horizon in ("short", "mid", "long"):
            item = outputs[horizon]
            total = item.prob_up + item.prob_flat + item.prob_down
            self.assertAlmostEqual(total, 1.0, places=6)
            display_total = item.display_prob_up + item.display_prob_uncertain + item.display_prob_down
            self.assertAlmostEqual(display_total, 1.0, places=6)
            self.assertGreaterEqual(item.prob_up, 0.0)
            self.assertGreaterEqual(item.prob_flat, 0.0)
            self.assertGreaterEqual(item.prob_down, 0.0)
            self.assertGreaterEqual(item.display_prob_up, 0.0)
            self.assertGreaterEqual(item.display_prob_uncertain, 0.0)
            self.assertGreaterEqual(item.display_prob_down, 0.0)
            self.assertLessEqual(item.prob_up, 1.0)
            self.assertLessEqual(item.prob_flat, 1.0)
            self.assertLessEqual(item.prob_down, 1.0)
            self.assertLessEqual(item.display_prob_up, 1.0)
            self.assertLessEqual(item.display_prob_uncertain, 1.0)
            self.assertLessEqual(item.display_prob_down, 1.0)
            if item.regime != "high_vol_extreme":
                self.assertLessEqual(max(item.prob_up, item.prob_flat, item.prob_down), 0.720001)
            if item.display_prob_up + item.display_prob_down > 1e-9:
                cond_up = item.display_prob_up / (item.display_prob_up + item.display_prob_down)
                self.assertLessEqual(cond_up, 0.900001)
                self.assertGreaterEqual(cond_up, 0.099999)

    def test_quantile_crossing_is_handled_in_mapper(self) -> None:
        q10 = np.asarray([0.020], dtype=float)
        q50 = np.asarray([0.010], dtype=float)
        q90 = np.asarray([0.000], dtype=float)
        threshold_up = np.asarray([0.012], dtype=float)
        threshold_down = np.asarray([-0.012], dtype=float)
        down, flat, up, mu, sigma = quantiles_to_raw_probabilities(
            q10=q10,
            q50=q50,
            q90=q90,
            threshold_up=threshold_up,
            threshold_down=threshold_down,
            sigma_floor=0.004,
        )
        self.assertGreaterEqual(float(sigma[0]), 0.004)
        self.assertAlmostEqual(float(down[0] + flat[0] + up[0]), 1.0, places=6)
        self.assertGreaterEqual(float(down[0]), 0.0)
        self.assertGreaterEqual(float(flat[0]), 0.0)
        self.assertGreaterEqual(float(up[0]), 0.0)
        self.assertTrue(np.isfinite(mu[0]))

    def test_backtest_contains_calibration_report(self) -> None:
        candles = _build_synthetic_candles()
        model = MarketProbabilityModel().fit(candles)
        report = model.backtest(candles).to_dict()
        for horizon in ("short", "mid", "long"):
            metrics = report[horizon]
            self.assertIn("log_loss", metrics)
            self.assertIn("brier_score", metrics)
            self.assertIn("accuracy", metrics)
            self.assertIn("confusion_matrix", metrics)
            self.assertIn("calibration_down", metrics)
            self.assertIn("calibration_up", metrics)
            self.assertIn("diagnostics", metrics)
            diagnostics = metrics["diagnostics"]
            self.assertIn("train_rows", diagnostics)
            self.assertIn("oof_rows", diagnostics)
            self.assertIn("train_label_distribution", diagnostics)
            self.assertIn("oof_label_distribution", diagnostics)
            self.assertIn("label_distribution", diagnostics)
            self.assertIn("chain_metrics", diagnostics)
            self.assertIn("chain_shift", diagnostics)
            self.assertIn("sigma_stats", diagnostics)
            self.assertIn("binary_chain_metrics", diagnostics)
            self.assertIn("binary_pipeline", diagnostics)
            self.assertTrue(len(metrics["confusion_matrix"]) == 3)
            self.assertTrue(len(metrics["calibration_down"]) >= 1)
            self.assertTrue(len(metrics["calibration_up"]) >= 1)
            self.assertGreater(int(diagnostics["train_rows"]), 0)
            self.assertGreater(int(diagnostics["oof_rows"]), 0)

    def test_none_calibrator_mode_keeps_raw_and_calibrated_close(self) -> None:
        candles = _build_synthetic_candles()
        config = MarketProbabilityModelConfig(
            calibrator=CalibratorConfig(mode="none"),
            mapper=ProbabilityMapperConfig(
                calibration_blend_5=0.0,
                calibration_blend_20=0.0,
                calibration_blend_60=0.0,
            ),
        )
        model = MarketProbabilityModel(config).fit(candles)
        diagnostics = model.diagnostics(candles)
        long_shift = diagnostics["long"].chain_shift["mean_abs_shift_raw_to_cal"]
        self.assertLessEqual(long_shift, 1e-8)

    def test_horizon_specific_pipeline_can_disable_mid_long_calibration(self) -> None:
        candles = _build_synthetic_candles()
        config = MarketProbabilityModelConfig(
            calibrator=CalibratorConfig(
                mode="conservative",
                mode_5="conservative",
                mode_20="none",
                mode_60="none",
            ),
            mapper=ProbabilityMapperConfig(
                lambda_5=0.80,
                lambda_20=1.00,
                lambda_60=1.00,
                calibration_blend_5=1.00,
                calibration_blend_20=0.00,
                calibration_blend_60=0.00,
                max_calibration_shift_5=0.55,
                max_calibration_shift_20=0.00,
                max_calibration_shift_60=0.00,
            ),
        )
        model = MarketProbabilityModel(config).fit(candles)
        diagnostics = model.diagnostics(candles)
        self.assertTrue(diagnostics["short"].calibration_mode.startswith("conservative"))
        self.assertTrue(diagnostics["mid"].calibration_mode.startswith("none"))
        self.assertTrue(diagnostics["long"].calibration_mode.startswith("none"))
        self.assertLessEqual(diagnostics["mid"].chain_shift["mean_abs_shift_raw_to_cal"], 1e-8)
        self.assertLessEqual(diagnostics["long"].chain_shift["mean_abs_shift_raw_to_cal"], 1e-8)
        self.assertAlmostEqual(
            diagnostics["mid"].chain_metrics.raw.log_loss,
            diagnostics["mid"].chain_metrics.final.log_loss,
            places=8,
        )
        self.assertAlmostEqual(
            diagnostics["long"].chain_metrics.raw.log_loss,
            diagnostics["long"].chain_metrics.final.log_loss,
            places=8,
        )


if __name__ == "__main__":
    unittest.main()
