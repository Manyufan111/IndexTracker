"""Tests for scenario probability legality."""

from __future__ import annotations

import unittest

from indextrack.app.models import TrendSignals
from indextrack.domain.scenario import HorizonScores, ScenarioEngine
from indextrack.domain.trend import TrendScoreResult


class ScenarioProbabilityTests(unittest.TestCase):
    def test_probabilities_are_in_range_and_sum_to_100(self) -> None:
        trend = TrendScoreResult(
            signals=TrendSignals(short="uptrend", mid="sideways", long="downtrend"),
            short_score=0.6,
            mid_score=0.1,
            long_score=-0.5,
        )
        momentum = HorizonScores(short=0.4, mid=0.0, long=-0.3)
        volatility = HorizonScores(short=0.2, mid=0.1, long=-0.1)

        scenarios = ScenarioEngine().generate(
            trend_scores=trend,
            momentum_scores=momentum,
            volatility_scores=volatility,
        )

        self.assertEqual([item.horizon for item in scenarios], ["short", "mid", "long"])
        for item in scenarios:
            self.assertGreaterEqual(item.uptrend_pct, 0)
            self.assertGreaterEqual(item.sideways_pct, 0)
            self.assertGreaterEqual(item.downtrend_pct, 0)
            self.assertLessEqual(item.uptrend_pct, 100)
            self.assertLessEqual(item.sideways_pct, 100)
            self.assertLessEqual(item.downtrend_pct, 100)
            total = item.uptrend_pct + item.sideways_pct + item.downtrend_pct
            self.assertAlmostEqual(total, 100.0, places=6)


if __name__ == "__main__":
    unittest.main()
