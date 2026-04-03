"""Feature engineering pipeline for probability modeling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import logging
from math import sqrt
from typing import Sequence

import numpy as np

from indextrack.app.models import Candle

LOGGER = logging.getLogger("indextrack.probability.feature_engineering")


class FeatureEngineeringError(ValueError):
    """Raised when feature engineering cannot be completed."""


@dataclass(frozen=True)
class FeatureMatrix:
    """Feature matrix and aligned market series."""

    dates: list[date]
    feature_names: list[str]
    values: np.ndarray
    close: np.ndarray
    daily_return: np.ndarray
    hist_vol_20: np.ndarray

    @property
    def row_count(self) -> int:
        return int(self.values.shape[0])

    @property
    def col_count(self) -> int:
        return int(self.values.shape[1])


class ProbabilityFeatureEngine:
    """Build no-leakage features from OHLCV time series."""

    def transform(self, candles: Sequence[Candle]) -> FeatureMatrix:
        if not candles:
            raise FeatureEngineeringError("candles 不能为空")

        ordered = sorted(candles, key=lambda item: item.trade_date)
        dates = [item.trade_date for item in ordered]
        close = np.asarray([item.close for item in ordered], dtype=float)
        high = np.asarray([item.high for item in ordered], dtype=float)
        low = np.asarray([item.low for item in ordered], dtype=float)
        volume = np.asarray([item.volume if item.volume is not None else np.nan for item in ordered], dtype=float)
        n = close.shape[0]
        if n < 120:
            raise FeatureEngineeringError("样本不足，至少需要 120 条日线用于稳定建模")

        ret_1 = _return_n(close, 1)
        ret_3 = _return_n(close, 3)
        ret_5 = _return_n(close, 5)
        ret_10 = _return_n(close, 10)
        ret_20 = _return_n(close, 20)
        ret_60 = _return_n(close, 60)

        ma5 = _rolling_mean(close, 5)
        ma10 = _rolling_mean(close, 10)
        ma20 = _rolling_mean(close, 20)
        ma60 = _rolling_mean(close, 60)

        ma_distance_5 = _safe_div(close, ma5) - 1.0
        ma_distance_10 = _safe_div(close, ma10) - 1.0
        ma_distance_20 = _safe_div(close, ma20) - 1.0
        ma_distance_60 = _safe_div(close, ma60) - 1.0

        ma_spread_5_20 = _safe_div(ma5, ma20) - 1.0
        ma_spread_20_60 = _safe_div(ma20, ma60) - 1.0

        vol5 = _rolling_std(ret_1, 5)
        vol10 = _rolling_std(ret_1, 10)
        vol20 = _rolling_std(ret_1, 20)
        vol60 = _rolling_std(ret_1, 60)

        dd20 = _rolling_max_drawdown(close, 20)
        dd60 = _rolling_max_drawdown(close, 60)

        rsi14 = _rsi(close, 14)
        macd_line, macd_signal, macd_hist = _macd(close)
        bollinger_z = _bollinger_zscore(close, 20)
        volume_z20 = _zscore(volume, 20)

        pos20 = _relative_position(close, 20)
        pos60 = _relative_position(close, 60)

        feature_names = [
            "ret_1",
            "ret_3",
            "ret_5",
            "ret_10",
            "ret_20",
            "ret_60",
            "ma_distance_5",
            "ma_distance_10",
            "ma_distance_20",
            "ma_distance_60",
            "ma_spread_5_20",
            "ma_spread_20_60",
            "vol_5",
            "vol_10",
            "vol_20",
            "vol_60",
            "drawdown_20",
            "drawdown_60",
            "rsi_14",
            "macd_line",
            "macd_signal",
            "macd_hist",
            "bollinger_zscore",
            "volume_zscore_20",
            "relative_pos_20",
            "relative_pos_60",
        ]
        values = np.column_stack(
            [
                ret_1,
                ret_3,
                ret_5,
                ret_10,
                ret_20,
                ret_60,
                ma_distance_5,
                ma_distance_10,
                ma_distance_20,
                ma_distance_60,
                ma_spread_5_20,
                ma_spread_20_60,
                vol5,
                vol10,
                vol20,
                vol60,
                dd20,
                dd60,
                rsi14,
                macd_line,
                macd_signal,
                macd_hist,
                bollinger_z,
                volume_z20,
                pos20,
                pos60,
            ]
        )

        LOGGER.debug(
            "feature_matrix_built rows=%s cols=%s date_start=%s date_end=%s",
            values.shape[0],
            values.shape[1],
            dates[0],
            dates[-1],
        )
        return FeatureMatrix(
            dates=dates,
            feature_names=feature_names,
            values=values,
            close=close,
            daily_return=ret_1,
            hist_vol_20=vol20,
        )


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-12)
    out[valid] = a[valid] / b[valid]
    return out


def _shift(arr: np.ndarray, periods: int) -> np.ndarray:
    if periods <= 0:
        return arr.copy()
    out = np.full(arr.shape[0], np.nan, dtype=float)
    out[periods:] = arr[:-periods]
    return out


def _return_n(close: np.ndarray, periods: int) -> np.ndarray:
    prev = _shift(close, periods)
    return _safe_div(close, prev) - 1.0


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full(arr.shape[0], np.nan, dtype=float)
    if arr.shape[0] < window:
        return out
    for idx in range(window - 1, arr.shape[0]):
        window_values = arr[idx - window + 1 : idx + 1]
        if np.isnan(window_values).all():
            continue
        out[idx] = float(np.nanmean(window_values))
    return out


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full(arr.shape[0], np.nan, dtype=float)
    if arr.shape[0] < window:
        return out
    for idx in range(window - 1, arr.shape[0]):
        window_values = arr[idx - window + 1 : idx + 1]
        finite = window_values[np.isfinite(window_values)]
        if finite.shape[0] < max(3, window // 2):
            continue
        out[idx] = float(np.std(finite, ddof=1))
    return out


def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full(arr.shape[0], np.nan, dtype=float)
    if arr.shape[0] < window:
        return out
    for idx in range(window - 1, arr.shape[0]):
        out[idx] = float(np.nanmax(arr[idx - window + 1 : idx + 1]))
    return out


def _rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full(arr.shape[0], np.nan, dtype=float)
    if arr.shape[0] < window:
        return out
    for idx in range(window - 1, arr.shape[0]):
        out[idx] = float(np.nanmin(arr[idx - window + 1 : idx + 1]))
    return out


def _rolling_max_drawdown(close: np.ndarray, window: int) -> np.ndarray:
    out = np.full(close.shape[0], np.nan, dtype=float)
    if close.shape[0] < window:
        return out
    for idx in range(window - 1, close.shape[0]):
        view = close[idx - window + 1 : idx + 1]
        peaks = np.maximum.accumulate(view)
        dd = _safe_div(view, peaks) - 1.0
        out[idx] = float(np.nanmin(dd))
    return out


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    out = np.full(arr.shape[0], np.nan, dtype=float)
    if arr.shape[0] == 0:
        return out
    alpha = 2.0 / (span + 1.0)
    prev = arr[0]
    out[0] = prev
    for idx in range(1, arr.shape[0]):
        current = arr[idx]
        prev = alpha * current + (1.0 - alpha) * prev
        out[idx] = prev
    return out


def _rsi(close: np.ndarray, window: int) -> np.ndarray:
    delta = close - _shift(close, 1)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = _rolling_mean(gains, window)
    avg_loss = _rolling_mean(losses, window)

    rs = _safe_div(avg_gain, avg_loss)
    out = 100.0 - (100.0 / (1.0 + rs))
    out[(avg_loss == 0) & np.isfinite(avg_gain)] = 100.0
    out[(avg_gain == 0) & (avg_loss == 0)] = 50.0
    return out / 100.0


def _macd(close: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    line = ema12 - ema26
    signal = _ema(line, 9)
    hist = line - signal
    return line, signal, hist


def _bollinger_zscore(close: np.ndarray, window: int) -> np.ndarray:
    ma = _rolling_mean(close, window)
    std = _rolling_std(close, window)
    z = _safe_div(close - ma, std)
    z[~np.isfinite(z)] = np.nan
    return z


def _zscore(arr: np.ndarray, window: int) -> np.ndarray:
    ma = _rolling_mean(arr, window)
    std = _rolling_std(arr, window)
    z = _safe_div(arr - ma, std)
    z[~np.isfinite(z)] = np.nan
    return z


def _relative_position(close: np.ndarray, window: int) -> np.ndarray:
    rolling_high = _rolling_max(close, window)
    rolling_low = _rolling_min(close, window)
    denom = rolling_high - rolling_low
    pos = _safe_div(close - rolling_low, denom)
    pos[(~np.isfinite(pos)) & np.isfinite(rolling_high) & np.isfinite(rolling_low)] = 0.5
    return pos


def annualized_threshold_scale(hist_vol: np.ndarray, horizon_days: int) -> np.ndarray:
    """Convert daily volatility to horizon-scaled return threshold component."""
    return hist_vol * sqrt(float(horizon_days))
