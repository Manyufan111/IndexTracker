"""Tests for UI view-model mapping."""

from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta

from indextrack.app.models import Candle, DataStatus, ScenarioProbs
from indextrack.domain.probability import HorizonProbabilityOutput
from indextrack.ui.view_model import build_index_card_view_model


class UIViewModelTests(unittest.TestCase):
    def test_chart_points_match_period_window_and_source_info(self) -> None:
        candles: list[Candle] = []
        start = date(2025, 1, 1)
        price = 100.0
        for idx in range(280):
            day = start + timedelta(days=idx)
            price += 0.2
            candles.append(
                Candle(
                    symbol="SP500",
                    trade_date=day,
                    open=price - 0.3,
                    high=price + 0.8,
                    low=price - 0.7,
                    close=price,
                    volume=1000 + idx,
                    source="test_source",
                    fetched_at=datetime(2026, 4, 1, 10, 0, 0),
                )
            )

        scenarios = [
            ScenarioProbs(horizon="short", uptrend_pct=55.23, sideways_pct=18.0, downtrend_pct=26.77),
            ScenarioProbs(horizon="mid", uptrend_pct=41.14, sideways_pct=24.0, downtrend_pct=34.86),
            ScenarioProbs(horizon="long", uptrend_pct=33.67, sideways_pct=32.0, downtrend_pct=34.33),
        ]
        status = DataStatus(
            source="yahoo_finance_primary",
            last_trade_date=candles[-1].trade_date,
            fetched_at=datetime(2026, 4, 1, 10, 0, 0),
            is_fresh=True,
            note=None,
        )

        card = build_index_card_view_model(
            symbol="SP500",
            period="1M",
            model_mode="quantile",
            candles=candles,
            summary_zh="测试结论",
            scenarios=scenarios,
            data_status=status,
            used_cache_fallback=False,
            horizon_outputs={
                "short": HorizonProbabilityOutput(
                    horizon="short",
                    horizon_days=5,
                    prob_down_raw=0.20,
                    prob_flat_raw=0.35,
                    prob_up_raw=0.45,
                    prob_down_calibrated=0.22,
                    prob_flat_calibrated=0.36,
                    prob_up_calibrated=0.42,
                    prob_down=0.24,
                    prob_flat=0.38,
                    prob_up=0.38,
                    threshold_up=0.018,
                    threshold_down=-0.018,
                    regime="mid_vol",
                    mu=0.004,
                    sigma=0.012,
                    confidence=0.38,
                    display_prob_up_raw=0.40,
                    display_prob_down_raw=0.25,
                    display_prob_up_calibrated=0.38,
                    display_prob_down_calibrated=0.27,
                    display_prob_up=0.36,
                    display_prob_uncertain=0.35,
                    display_prob_down=0.29,
                    display_confidence=0.22,
                    display_state="uncertain",
                    display_warning="",
                ),
                "mid": HorizonProbabilityOutput(
                    horizon="mid",
                    horizon_days=20,
                    prob_down_raw=0.30,
                    prob_flat_raw=0.40,
                    prob_up_raw=0.30,
                    prob_down_calibrated=0.28,
                    prob_flat_calibrated=0.42,
                    prob_up_calibrated=0.30,
                    prob_down=0.29,
                    prob_flat=0.43,
                    prob_up=0.28,
                    threshold_up=0.040,
                    threshold_down=-0.040,
                    regime="mid_vol",
                    mu=0.003,
                    sigma=0.020,
                    confidence=0.43,
                    display_prob_up_raw=0.30,
                    display_prob_down_raw=0.30,
                    display_prob_up_calibrated=0.31,
                    display_prob_down_calibrated=0.29,
                    display_prob_up=0.32,
                    display_prob_uncertain=0.40,
                    display_prob_down=0.28,
                    display_confidence=0.19,
                    display_state="uncertain",
                    display_warning="",
                ),
                "long": HorizonProbabilityOutput(
                    horizon="long",
                    horizon_days=60,
                    prob_down_raw=0.35,
                    prob_flat_raw=0.34,
                    prob_up_raw=0.31,
                    prob_down_calibrated=0.33,
                    prob_flat_calibrated=0.36,
                    prob_up_calibrated=0.31,
                    prob_down=0.32,
                    prob_flat=0.38,
                    prob_up=0.30,
                    threshold_up=0.075,
                    threshold_down=-0.075,
                    regime="high_vol",
                    mu=0.002,
                    sigma=0.030,
                    confidence=0.38,
                    display_prob_up_raw=0.26,
                    display_prob_down_raw=0.30,
                    display_prob_up_calibrated=0.27,
                    display_prob_down_calibrated=0.29,
                    display_prob_up=0.28,
                    display_prob_uncertain=0.42,
                    display_prob_down=0.30,
                    display_confidence=0.16,
                    display_state="uncertain",
                    display_warning="",
                ),
            },
        )

        self.assertEqual(card.title, "S&P 500")
        self.assertEqual(card.period, "1M")
        self.assertEqual(len(card.points), 21)
        self.assertIn("数据源:", card.source_line)
        self.assertIn("当前模型: quantile", card.method_line)
        self.assertEqual(len(card.scenario_lines), 3)
        self.assertIn("上涨", card.scenario_lines[0])
        self.assertIn("不确定", card.scenario_lines[0])
        self.assertIn("下跌", card.scenario_lines[0])
        self.assertEqual(len(card.probability_detail_lines), 3)
        self.assertIn("raw", card.probability_detail_lines[0])
        self.assertIn("calibrated", card.probability_detail_lines[0])
        self.assertIn("final", card.probability_detail_lines[0])

    def test_legacy_mode_method_and_detail_hint(self) -> None:
        candles = [
            Candle(
                symbol="SP500",
                trade_date=date(2026, 3, 31),
                open=100,
                high=101,
                low=99,
                close=100.5,
                volume=1000,
                source="test_source",
                fetched_at=datetime(2026, 4, 1, 10, 0, 0),
            )
        ]
        scenarios = [
            ScenarioProbs(horizon="short", uptrend_pct=14.29, sideways_pct=12.0, downtrend_pct=73.71),
            ScenarioProbs(horizon="mid", uptrend_pct=16.67, sideways_pct=18.0, downtrend_pct=65.33),
            ScenarioProbs(horizon="long", uptrend_pct=23.08, sideways_pct=25.0, downtrend_pct=51.92),
        ]
        status = DataStatus(
            source="yahoo_finance_primary",
            last_trade_date=date(2026, 3, 31),
            fetched_at=datetime(2026, 4, 1, 10, 0, 0),
            is_fresh=True,
            note=None,
        )
        card = build_index_card_view_model(
            symbol="SP500",
            period="1M",
            model_mode="legacy",
            candles=candles,
            summary_zh="测试结论",
            scenarios=scenarios,
            data_status=status,
            used_cache_fallback=False,
            horizon_outputs=None,
        )
        self.assertIn("当前模型: legacy", card.method_line)
        self.assertEqual(len(card.probability_detail_lines), 1)
        self.assertIn("未输出 raw/calibrated", card.probability_detail_lines[0])


if __name__ == "__main__":
    unittest.main()
