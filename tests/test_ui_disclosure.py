"""Tests for UI disclosure and page rendering."""

from __future__ import annotations

import unittest

from indextrack.ui.disclosure import render_disclosure
from indextrack.ui.page import render_dashboard_page
from indextrack.ui.view_model import ChartPoint, IndexCardViewModel


class UIDisclosureTests(unittest.TestCase):
    def test_disclosure_is_collapsed_by_default(self) -> None:
        html = render_disclosure(
            source_line="数据源: yahoo，更新时间: 2026-04-01 10:00:00",
            method_line="分析方法: 特征+评分+概率",
        )
        self.assertIn("<details", html)
        self.assertNotIn("<details open", html)
        self.assertIn("数据来源", html)
        self.assertIn("分析方法", html)

    def test_dashboard_contains_tabs_and_card_sections(self) -> None:
        card = IndexCardViewModel(
            symbol="SP500",
            title="S&P 500",
            period="1M",
            points=[
                ChartPoint(date_label="2026-03-28", close=6500),
                ChartPoint(date_label="2026-03-31", close=6528),
            ],
            summary_zh="测试总结",
            scenario_lines=["短期: 区间震荡（45.0%）"],
            probability_detail_lines=[
                "短期 明细: raw(上/下/不确定) 57.1%/32.9%/10.0% | calibrated(上/下/不确定) 57.6%/32.4%/10.0% | final(上/下/不确定) 60.0%/30.0%/10.0%"
            ],
            source_line="数据源: yahoo_finance_primary",
            method_line="分析方法: 特征+评分+概率",
        )
        html = render_dashboard_page(
            cards=[card],
            period="1M",
            generated_at="2026-04-01 10:00:00",
            model_mode="legacy",
            ui_notice="测试提示",
        )
        self.assertIn("IndexTrack 趋势总览", html)
        self.assertIn("class=\"tab active\"", html)
        self.assertIn("查看数据来源与分析方法", html)
        self.assertIn("S&amp;P 500", html)
        self.assertIn("calibrated", html)
        self.assertIn("Quantile 模型", html)
        self.assertIn("Legacy 模型", html)
        self.assertIn("model=legacy", html)
        self.assertIn("测试提示", html)


if __name__ == "__main__":
    unittest.main()
