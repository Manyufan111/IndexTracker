"""Application-level data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

Horizon = Literal["short", "mid", "long"]
TrendDirection = Literal["uptrend", "sideways", "downtrend"]
VALID_HORIZONS = {"short", "mid", "long"}
VALID_TREND_DIRECTIONS = {"uptrend", "sideways", "downtrend"}
PROBABILITY_SUM_TOLERANCE = 0.5


class ModelValidationError(ValueError):
    """Raised when a model contains invalid values."""


@dataclass(frozen=True)
class Candle:
    """Standard daily OHLCV market record."""

    symbol: str
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float | None
    source: str
    fetched_at: datetime

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ModelValidationError("symbol 不能为空")
        if not self.source.strip():
            raise ModelValidationError("source 不能为空")
        if self.high < self.low:
            raise ModelValidationError("high 不能小于 low")
        if any(value <= 0 for value in (self.open, self.high, self.low, self.close)):
            raise ModelValidationError("OHLC 必须全部为正数")
        if self.volume is not None and self.volume < 0:
            raise ModelValidationError("volume 不能为负数")


@dataclass(frozen=True)
class DataStatus:
    """Data freshness and provenance for an analysis result."""

    source: str
    last_trade_date: date
    fetched_at: datetime
    is_fresh: bool
    note: str | None = None

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ModelValidationError("source 不能为空")


@dataclass(frozen=True)
class DataSnapshotMeta:
    """Metadata for the latest successful data snapshot."""

    symbol: str
    source: str
    last_trade_date: date
    fetched_at: datetime
    row_count: int

    def __post_init__(self) -> None:
        if self.row_count <= 0:
            raise ModelValidationError("row_count 必须是正整数")


@dataclass(frozen=True)
class TrendSignals:
    """Directional trend label for short/mid/long horizons."""

    short: TrendDirection
    mid: TrendDirection
    long: TrendDirection

    def __post_init__(self) -> None:
        for field_name, direction in (
            ("short", self.short),
            ("mid", self.mid),
            ("long", self.long),
        ):
            if direction not in VALID_TREND_DIRECTIONS:
                raise ModelValidationError(f"{field_name} 趋势值非法: {direction}")


@dataclass(frozen=True)
class ScenarioProbs:
    """Three-scenario probability result for a single horizon."""

    horizon: Horizon
    uptrend_pct: float
    sideways_pct: float
    downtrend_pct: float

    def __post_init__(self) -> None:
        if self.horizon not in VALID_HORIZONS:
            raise ModelValidationError(f"horizon 值非法: {self.horizon}")
        values = (self.uptrend_pct, self.sideways_pct, self.downtrend_pct)
        if any(value < 0 or value > 100 for value in values):
            raise ModelValidationError("场景概率必须位于 0-100 区间")
        total = self.uptrend_pct + self.sideways_pct + self.downtrend_pct
        if abs(total - 100.0) > PROBABILITY_SUM_TOLERANCE:
            raise ModelValidationError(
                f"场景概率之和必须接近 100%，当前为 {total:.2f}%"
            )


@dataclass(frozen=True)
class AnalysisResult:
    """User-facing analysis payload for a single index."""

    symbol: str
    as_of: date
    trend_signals: TrendSignals
    scenario_probs: list[ScenarioProbs] = field(default_factory=list)
    confidence: float = 0.0
    data_status: DataStatus | None = None
    summary_zh: str = ""
    risk_notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ModelValidationError("symbol 不能为空")
        if self.confidence < 0 or self.confidence > 1:
            raise ModelValidationError("confidence 必须位于 0-1 区间")
        if not self.scenario_probs:
            raise ModelValidationError("scenario_probs 不能为空")
        horizons = {item.horizon for item in self.scenario_probs}
        missing = VALID_HORIZONS - horizons
        if missing:
            sorted_missing = ", ".join(sorted(missing))
            raise ModelValidationError(f"缺少时间窗口概率: {sorted_missing}")
