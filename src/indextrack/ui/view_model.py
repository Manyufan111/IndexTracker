"""View-model builders for IndexTrack UI pages."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from indextrack.app.models import Candle, DataStatus, ScenarioProbs

_PERIOD_TO_POINTS = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
}

_SYMBOL_LABELS = {
    "SP500": "S&P 500",
    "NASDAQ": "Nasdaq Composite",
}

@dataclass(frozen=True)
class ChartPoint:
    """One data point rendered in trend chart."""

    date_label: str
    close: float


@dataclass(frozen=True)
class IndexCardViewModel:
    """UI-ready model for one index card."""

    symbol: str
    title: str
    period: str
    points: list[ChartPoint] = field(default_factory=list)
    summary_zh: str = ""
    scenario_lines: list[str] = field(default_factory=list)
    probability_detail_lines: list[str] = field(default_factory=list)
    source_line: str = ""
    method_line: str = ""


def build_index_card_view_model(
    *,
    symbol: str,
    period: str,
    model_mode: str,
    candles: list[Candle],
    summary_zh: str,
    scenarios: list[ScenarioProbs],
    data_status: DataStatus,
    used_cache_fallback: bool,
    horizon_outputs: dict[str, Any] | None = None,
) -> IndexCardViewModel:
    """Build a UI card model from analysis outputs."""
    normalized_period = period.strip().upper()
    max_points = _PERIOD_TO_POINTS.get(normalized_period, _PERIOD_TO_POINTS["1Y"])
    selected_candles = candles[-max_points:] if len(candles) > max_points else candles

    points = [
        ChartPoint(date_label=item.trade_date.isoformat(), close=item.close)
        for item in selected_candles
    ]
    title = _SYMBOL_LABELS.get(symbol.upper(), symbol.upper())
    scenario_lines = _format_scenario_lines(scenarios)
    detail_lines = _format_probability_detail_lines(horizon_outputs or {})
    if not detail_lines:
        detail_lines = [
            "当前模型未输出 raw/calibrated 概率明细（请切换 quantile 模型查看）。"
        ]

    source_chunks = [
        f"数据源: {data_status.source}",
        f"最新交易日: {data_status.last_trade_date.isoformat()}",
        f"更新时间: {_format_datetime(data_status.fetched_at)}",
        f"新鲜度: {'最新可用' if data_status.is_fresh else '可能滞后'}",
    ]
    if used_cache_fallback:
        source_chunks.append("本次使用缓存回退")
    if data_status.note:
        source_chunks.append(f"提示: {data_status.note}")
    source_line = "；".join(source_chunks)

    normalized_mode = model_mode.strip().lower()
    if normalized_mode == "quantile":
        method_line = (
            "当前模型: quantile（分位数回归 q10/q50/q90 + OOF 多分类校准 + 三分类主链路 + 二分类展示层校准与约束）。"
        )
    else:
        method_line = "当前模型: legacy（趋势/动量/波动规则引擎加权评分）。"

    return IndexCardViewModel(
        symbol=symbol,
        title=title,
        period=normalized_period,
        points=points,
        summary_zh=summary_zh,
        scenario_lines=scenario_lines,
        probability_detail_lines=detail_lines,
        source_line=source_line,
        method_line=method_line,
    )


def _format_scenario_lines(scenarios: list[ScenarioProbs]) -> list[str]:
    lines: list[str] = []
    for item in scenarios:
        horizon = {"short": "短期", "mid": "中期", "long": "长期"}[item.horizon]
        lines.append(
            f"{horizon}: 上涨 {item.uptrend_pct:.1f}% / "
            f"下跌 {item.downtrend_pct:.1f}% / "
            f"不确定 {item.sideways_pct:.1f}%"
        )
    return lines


def _format_probability_detail_lines(
    horizon_outputs: dict[str, Any],
) -> list[str]:
    lines: list[str] = []
    for horizon in ("short", "mid", "long"):
        item = horizon_outputs.get(horizon)
        if item is None:
            continue
        label = {"short": "短期", "mid": "中期", "long": "长期"}[horizon]
        lines.append(
            f"{label} 明细: raw(上/下/不确定) "
            f"{item.display_prob_up_raw * 100:.1f}%/"
            f"{item.display_prob_down_raw * 100:.1f}%/"
            f"{item.display_prob_uncertain * 100:.1f}% | "
            f"calibrated(上/下/不确定) "
            f"{item.display_prob_up_calibrated * 100:.1f}%/"
            f"{item.display_prob_down_calibrated * 100:.1f}%/"
            f"{item.display_prob_uncertain * 100:.1f}% | "
            f"final(上/下/不确定) "
            f"{item.display_prob_up * 100:.1f}%/"
            f"{item.display_prob_down * 100:.1f}%/"
            f"{item.display_prob_uncertain * 100:.1f}% | "
            f"internal_state={item.internal_state} | "
            f"label={item.label} | "
            f"headline={item.headline_label} | "
            f"display_label={item.display_label} | "
            f"top1={item.top1_prob * 100:.1f}% | "
            f"top2={item.top2_prob * 100:.1f}% | "
            f"margin={item.margin * 100:.1f}% | "
            f"threshold=[{item.threshold_down * 100:.2f}%, {item.threshold_up * 100:.2f}%] | "
            f"regime={item.regime} | signal_strength={item.signal_strength * 100:.1f}%"
        )
        if item.warning_code:
            lines.append(
                f"{label} 告警: level={item.warning_level} | "
                f"code={item.warning_code} | msg={item.warning_message}"
            )
    return lines


def _format_datetime(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")
