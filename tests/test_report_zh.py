"""Readability tests for Chinese report output."""

from __future__ import annotations

import unittest
from datetime import date, datetime

from indextrack.app.models import DataStatus, ScenarioProbs, TrendSignals
from indextrack.domain.report import ReportGenerator


class ReportZhTests(unittest.TestCase):
    def test_summary_contains_required_fields_and_sentence_length(self) -> None:
        report = ReportGenerator().generate(
            symbol="SP500",
            trend_signals=TrendSignals(short="uptrend", mid="uptrend", long="downtrend"),
            scenario_probs=[
                ScenarioProbs(horizon="short", uptrend_pct=72.57, sideways_pct=10.0, downtrend_pct=17.43),
                ScenarioProbs(horizon="mid", uptrend_pct=53.64, sideways_pct=18.0, downtrend_pct=28.36),
                ScenarioProbs(horizon="long", uptrend_pct=30.0, sideways_pct=26.0, downtrend_pct=44.0),
            ],
            data_status=DataStatus(
                source="test_source",
                last_trade_date=date(2026, 3, 31),
                fetched_at=datetime(2026, 4, 1, 8, 0, 0),
                is_fresh=False,
                note="数据滞后 1 天。",
            ),
            used_cache_fallback=True,
        )

        summary = report.summary_zh
        self.assertIn("短期", summary)
        self.assertIn("中期", summary)
        self.assertIn("长期", summary)
        self.assertIn("场景概率：上涨", summary)
        self.assertIn("不确定", summary)
        self.assertIn("下跌", summary)
        self.assertIn("数据来源", summary)
        self.assertIn("仅供参考，不构成投资建议", summary)

        sentence_count = len([item for item in summary.split("。") if item.strip()])
        self.assertGreaterEqual(sentence_count, 3)
        self.assertLessEqual(sentence_count, 8)

        self.assertGreaterEqual(len(report.risk_notes), 2)
        self.assertTrue(any("仅供参考" in item for item in report.risk_notes))

    def test_headline_uses_display_labels_when_provided(self) -> None:
        report = ReportGenerator().generate(
            symbol="SP500",
            trend_signals=TrendSignals(short="uptrend", mid="sideways", long="downtrend"),
            scenario_probs=[
                ScenarioProbs(horizon="short", uptrend_pct=46.8, sideways_pct=8.6, downtrend_pct=44.6),
                ScenarioProbs(horizon="mid", uptrend_pct=40.0, sideways_pct=34.0, downtrend_pct=26.0),
                ScenarioProbs(horizon="long", uptrend_pct=31.0, sideways_pct=18.0, downtrend_pct=51.0),
            ],
            data_status=DataStatus(
                source="test_source",
                last_trade_date=date(2026, 3, 31),
                fetched_at=datetime(2026, 4, 1, 8, 0, 0),
                is_fresh=True,
                note=None,
            ),
            used_cache_fallback=False,
            display_labels={
                "short": "mild_up",
                "mid": "uncertain",
                "long": "mild_down",
            },
        )
        self.assertIn("短期偏上、中期不确定、长期偏下", report.summary_zh)


if __name__ == "__main__":
    unittest.main()
