"""Freshness checking for daily market data."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from datetime import date, datetime, timedelta
from typing import Sequence
from zoneinfo import ZoneInfo

from indextrack.app.models import Candle, DataStatus


@dataclass(frozen=True)
class FreshnessCheck:
    """Rich freshness evaluation result."""

    is_fresh: bool
    expected_trade_date: date
    actual_trade_date: date | None
    lag_days: int | None
    note: str | None = None


class FreshnessGuard:
    """Validate whether candles are updated to the latest expected trade date."""

    def __init__(
        self,
        *,
        timezone_name: str = "America/New_York",
        market_close_hour: int = 16,
        publish_grace_hours: int = 2,
    ) -> None:
        self._timezone = ZoneInfo(timezone_name)
        self._market_close_hour = market_close_hour
        self._publish_grace_hours = publish_grace_hours

    def evaluate(
        self,
        candles: Sequence[Candle],
        *,
        now: datetime | None = None,
    ) -> FreshnessCheck:
        reference_now = self._coerce_now(now)
        expected_trade_date = self._expected_latest_trade_date(reference_now)
        actual_trade_date = candles[-1].trade_date if candles else None

        if actual_trade_date is None:
            return FreshnessCheck(
                is_fresh=False,
                expected_trade_date=expected_trade_date,
                actual_trade_date=None,
                lag_days=None,
                note="未获取到任何行情数据。",
            )

        lag_days = self._trading_day_gap(actual_trade_date, expected_trade_date)
        is_fresh = lag_days <= 0
        note = None
        if not is_fresh:
            note = (
                f"数据滞后 {lag_days} 个交易日（预期最新交易日: {expected_trade_date.isoformat()}，"
                f"实际最新交易日: {actual_trade_date.isoformat()}）。"
            )

        return FreshnessCheck(
            is_fresh=is_fresh,
            expected_trade_date=expected_trade_date,
            actual_trade_date=actual_trade_date,
            lag_days=lag_days,
            note=note,
        )

    def build_data_status(
        self,
        *,
        source: str,
        candles: Sequence[Candle],
        fetched_at: datetime | None = None,
        now: datetime | None = None,
    ) -> DataStatus:
        check = self.evaluate(candles, now=now)
        status_time = fetched_at or (candles[-1].fetched_at if candles else datetime.now(self._timezone))
        last_trade_date = (
            check.actual_trade_date
            if check.actual_trade_date is not None
            else check.expected_trade_date
        )
        return DataStatus(
            source=source,
            last_trade_date=last_trade_date,
            fetched_at=status_time,
            is_fresh=check.is_fresh,
            note=check.note,
        )

    def _expected_latest_trade_date(self, now_local: datetime) -> date:
        current_date = now_local.date()
        if not self._is_trading_day(current_date):
            return self._previous_trading_day(current_date)

        # During trading day before close+grace, yesterday is the expected latest complete bar.
        cutoff_hour = self._market_close_hour + self._publish_grace_hours
        if now_local.hour < cutoff_hour:
            return self._previous_trading_day(current_date)
        return current_date

    def _is_trading_day(self, day: date) -> bool:
        return not _is_non_trading_day(day)

    def _previous_trading_day(self, day: date) -> date:
        candidate = day - timedelta(days=1)
        while not self._is_trading_day(candidate):
            candidate -= timedelta(days=1)
        return candidate

    def _trading_day_gap(self, older: date, newer: date) -> int:
        if older == newer:
            return 0
        if older < newer:
            gap = 0
            current = older
            while current < newer:
                current += timedelta(days=1)
                if self._is_trading_day(current):
                    gap += 1
            return gap
        # Future-dated latest candle relative to expected trade date.
        gap = 0
        current = newer
        while current < older:
            current += timedelta(days=1)
            if self._is_trading_day(current):
                gap += 1
        return -gap

    def _coerce_now(self, now: datetime | None) -> datetime:
        if now is None:
            return datetime.now(self._timezone)
        if now.tzinfo is None:
            return now.replace(tzinfo=self._timezone)
        return now.astimezone(self._timezone)


def _is_non_trading_day(day: date) -> bool:
    return day.weekday() >= 5 or day in _nyse_holidays(day.year)


@lru_cache(maxsize=32)
def _nyse_holidays(year: int) -> frozenset[date]:
    holidays: set[date] = set()

    # Fixed-date holidays (observed)
    holidays.add(_observed_fixed_holiday(date(year, 1, 1)))   # New Year's Day
    holidays.add(_observed_fixed_holiday(date(year, 7, 4)))   # Independence Day
    holidays.add(_observed_fixed_holiday(date(year, 12, 25)))  # Christmas Day

    # Juneteenth is a NYSE holiday starting in 2022.
    if year >= 2022:
        holidays.add(_observed_fixed_holiday(date(year, 6, 19)))

    # Moving holidays
    holidays.add(_nth_weekday_of_month(year, 1, weekday=0, n=3))   # MLK Day
    holidays.add(_nth_weekday_of_month(year, 2, weekday=0, n=3))   # Presidents Day
    holidays.add(_last_weekday_of_month(year, 5, weekday=0))       # Memorial Day
    holidays.add(_nth_weekday_of_month(year, 9, weekday=0, n=1))   # Labor Day
    holidays.add(_nth_weekday_of_month(year, 11, weekday=3, n=4))  # Thanksgiving
    holidays.add(_easter_sunday(year) - timedelta(days=2))         # Good Friday

    # If next-year New Year's is observed on Dec 31 of current year, include it.
    next_new_year_observed = _observed_fixed_holiday(date(year + 1, 1, 1))
    if next_new_year_observed.year == year:
        holidays.add(next_new_year_observed)

    return frozenset(item for item in holidays if item.year == year)


def _observed_fixed_holiday(day: date) -> date:
    if day.weekday() == 5:  # Saturday -> observed Friday
        return day - timedelta(days=1)
    if day.weekday() == 6:  # Sunday -> observed Monday
        return day + timedelta(days=1)
    return day


def _nth_weekday_of_month(year: int, month: int, *, weekday: int, n: int) -> date:
    first_day = date(year, month, 1)
    offset = (weekday - first_day.weekday()) % 7
    return first_day + timedelta(days=offset + (n - 1) * 7)


def _last_weekday_of_month(year: int, month: int, *, weekday: int) -> date:
    if month == 12:
        first_next_month = date(year + 1, 1, 1)
    else:
        first_next_month = date(year, month + 1, 1)
    last_day = first_next_month - timedelta(days=1)
    offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=offset)


def _easter_sunday(year: int) -> date:
    # Anonymous Gregorian algorithm.
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)
