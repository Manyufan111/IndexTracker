"""Tests for freshness guard with NYSE holiday calendar support."""

from __future__ import annotations

import unittest
from datetime import date, datetime

from indextrack.app.models import Candle
from indextrack.infra.freshness import FreshnessGuard


def _candle(trade_date: date) -> Candle:
    return Candle(
        symbol="SP500",
        trade_date=trade_date,
        open=100,
        high=101,
        low=99,
        close=100.5,
        volume=1000,
        source="test",
        fetched_at=datetime(2026, 1, 1, 0, 0, 0),
    )


class FreshnessGuardHolidayTests(unittest.TestCase):
    def test_observed_holiday_expected_previous_trading_day(self) -> None:
        # 2026-07-03 is NYSE holiday (observed Independence Day).
        guard = FreshnessGuard(timezone_name="America/New_York")
        check = guard.evaluate([_candle(date(2026, 7, 2))], now=datetime(2026, 7, 3, 12, 0, 0))
        self.assertTrue(check.is_fresh)
        self.assertEqual(check.expected_trade_date, date(2026, 7, 2))

    def test_first_trading_day_after_holiday_before_close(self) -> None:
        # 2026-07-06 Monday, before cutoff should still expect previous complete bar (2026-07-02).
        guard = FreshnessGuard(timezone_name="America/New_York")
        check = guard.evaluate([_candle(date(2026, 7, 2))], now=datetime(2026, 7, 6, 10, 0, 0))
        self.assertTrue(check.is_fresh)
        self.assertEqual(check.expected_trade_date, date(2026, 7, 2))

    def test_first_trading_day_after_holiday_after_close(self) -> None:
        # 2026-07-06 after cutoff should expect 2026-07-06 bar.
        guard = FreshnessGuard(timezone_name="America/New_York")
        check = guard.evaluate([_candle(date(2026, 7, 2))], now=datetime(2026, 7, 6, 19, 0, 0))
        self.assertFalse(check.is_fresh)
        self.assertEqual(check.expected_trade_date, date(2026, 7, 6))
        self.assertEqual(check.lag_days, 1)

    def test_dec_31_observed_new_year_holiday(self) -> None:
        # 2021-12-31 is observed New Year's Day holiday for 2022-01-01 (Saturday).
        guard = FreshnessGuard(timezone_name="America/New_York")
        check = guard.evaluate([_candle(date(2021, 12, 30))], now=datetime(2021, 12, 31, 12, 0, 0))
        self.assertTrue(check.is_fresh)
        self.assertEqual(check.expected_trade_date, date(2021, 12, 30))


if __name__ == "__main__":
    unittest.main()
