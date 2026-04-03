"""Feature engineering for trend, momentum, and volatility signals."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean
from typing import Sequence

from indextrack.app.models import Candle

TRADING_DAYS_PER_YEAR = 252


class FeatureError(ValueError):
    """Raised when features cannot be computed from input candles."""


@dataclass(frozen=True)
class FeatureSet:
    """Computed features from daily candles."""

    last_close: float
    sma_20: float | None
    sma_50: float | None
    sma_200: float | None
    price_vs_sma_20_pct: float | None
    price_vs_sma_50_pct: float | None
    price_vs_sma_200_pct: float | None
    momentum_5d_pct: float | None
    momentum_20d_pct: float | None
    momentum_60d_pct: float | None
    volatility_20d_ann: float | None
    volatility_60d_ann: float | None
    atr_14_pct: float | None


class FeatureEngine:
    """Compute reusable technical features from normalized candles."""

    def compute(self, candles: Sequence[Candle]) -> FeatureSet:
        if not candles:
            raise FeatureError("candles 不能为空")
        ordered = sorted(candles, key=lambda item: item.trade_date)
        closes = [row.close for row in ordered]
        highs = [row.high for row in ordered]
        lows = [row.low for row in ordered]

        last_close = closes[-1]
        sma_20 = _sma(closes, 20)
        sma_50 = _sma(closes, 50)
        sma_200 = _sma(closes, 200)

        return FeatureSet(
            last_close=last_close,
            sma_20=sma_20,
            sma_50=sma_50,
            sma_200=sma_200,
            price_vs_sma_20_pct=_pct_distance(last_close, sma_20),
            price_vs_sma_50_pct=_pct_distance(last_close, sma_50),
            price_vs_sma_200_pct=_pct_distance(last_close, sma_200),
            momentum_5d_pct=_momentum_pct(closes, 5),
            momentum_20d_pct=_momentum_pct(closes, 20),
            momentum_60d_pct=_momentum_pct(closes, 60),
            volatility_20d_ann=_annualized_volatility(closes, 20),
            volatility_60d_ann=_annualized_volatility(closes, 60),
            atr_14_pct=_atr_pct(highs, lows, closes, 14),
        )


def _sma(values: Sequence[float], window: int) -> float | None:
    if len(values) < window:
        return None
    return mean(values[-window:])


def _pct_distance(price: float, baseline: float | None) -> float | None:
    if baseline is None or baseline == 0:
        return None
    return (price - baseline) / baseline * 100


def _momentum_pct(closes: Sequence[float], window: int) -> float | None:
    if len(closes) <= window:
        return None
    past = closes[-(window + 1)]
    current = closes[-1]
    if past == 0:
        return None
    return (current - past) / past * 100


def _annualized_volatility(closes: Sequence[float], window: int) -> float | None:
    if len(closes) <= window:
        return None
    sub = closes[-(window + 1) :]
    returns: list[float] = []
    for prev, curr in zip(sub[:-1], sub[1:]):
        if prev == 0:
            continue
        returns.append((curr - prev) / prev)
    if len(returns) < 2:
        return None
    avg = mean(returns)
    variance = sum((item - avg) ** 2 for item in returns) / (len(returns) - 1)
    daily_std = sqrt(variance)
    return daily_std * sqrt(TRADING_DAYS_PER_YEAR)


def _atr_pct(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], window: int) -> float | None:
    if len(closes) <= window:
        return None
    start_index = len(closes) - window
    true_ranges: list[float] = []
    for idx in range(start_index, len(closes)):
        current_high = highs[idx]
        current_low = lows[idx]
        prev_close = closes[idx - 1] if idx > 0 else closes[idx]
        tr = max(
            current_high - current_low,
            abs(current_high - prev_close),
            abs(current_low - prev_close),
        )
        true_ranges.append(tr)
    atr = mean(true_ranges)
    latest_close = closes[-1]
    if latest_close == 0:
        return None
    return atr / latest_close * 100
