"""Chinese report generation for index analysis results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from indextrack.app.models import DataStatus, ScenarioProbs, TrendSignals

SYMBOL_LABELS = {
    "SP500": "S&P 500 指数",
    "NASDAQ": "纳斯达克综合指数",
    "^GSPC": "S&P 500 指数",
    "^IXIC": "纳斯达克综合指数",
}

TREND_LABELS_ZH = {
    "uptrend": "上行",
    "sideways": "不确定",
    "downtrend": "下行",
}

DISPLAY_LABELS_ZH = {
    "strong_up": "上行",
    "mild_up": "偏上",
    "uncertain": "不确定",
    "mild_down": "偏下",
    "strong_down": "下行",
    "high-confidence up": "上行",
    "high-confidence down": "下行",
}

@dataclass(frozen=True)
class ReportOutput:
    """User-facing Chinese summary and risk notes."""

    summary_zh: str
    risk_notes: list[str]


class ReportGenerator:
    """Generate concise Chinese analysis narratives."""

    def generate(
        self,
        *,
        symbol: str,
        trend_signals: TrendSignals,
        scenario_probs: list[ScenarioProbs],
        data_status: DataStatus,
        used_cache_fallback: bool,
        display_labels: Mapping[str, str] | None = None,
    ) -> ReportOutput:
        scenario_by_horizon = {item.horizon: item for item in scenario_probs}
        short = scenario_by_horizon["short"]
        mid = scenario_by_horizon["mid"]
        long = scenario_by_horizon["long"]

        symbol_label = SYMBOL_LABELS.get(symbol.upper(), symbol.upper())
        short_label = _headline_label("short", short, display_labels)
        mid_label = _headline_label("mid", mid, display_labels)
        long_label = _headline_label("long", long, display_labels)
        trend_sentence = (
            f"{symbol_label}当前趋势判断为：短期{short_label}、"
            f"中期{mid_label}、长期{long_label}。"
        )
        short_sentence = (
            f"短期（1-5个交易日）场景概率：上涨 {short.uptrend_pct:.1f}% / "
            f"下跌 {short.downtrend_pct:.1f}% / "
            f"不确定 {short.sideways_pct:.1f}%。"
        )
        mid_sentence = (
            f"中期（1-4周）场景概率：上涨 {mid.uptrend_pct:.1f}% / "
            f"下跌 {mid.downtrend_pct:.1f}% / "
            f"不确定 {mid.sideways_pct:.1f}%。"
        )
        long_sentence = (
            f"长期（1-3个月）场景概率：上涨 {long.uptrend_pct:.1f}% / "
            f"下跌 {long.downtrend_pct:.1f}% / "
            f"不确定 {long.sideways_pct:.1f}%。"
        )
        data_sentence = (
            f"本次分析数据来源于 {data_status.source}，最新交易日为 "
            f"{data_status.last_trade_date.isoformat()}，数据新鲜度状态为"
            f"{'最新可用' if data_status.is_fresh else '可能滞后'}。"
        )

        summary_lines = [
            trend_sentence,
            short_sentence,
            mid_sentence,
            long_sentence,
            data_sentence,
        ]
        if data_status.note:
            summary_lines.append(f"数据提示：{data_status.note}")
        if used_cache_fallback:
            summary_lines.append("本次结果使用了缓存回退，请重点关注后续数据更新。")
        summary_lines.append("以上为模型估计结果，仅供参考，不构成投资建议。")

        risk_notes = self._build_risk_notes(data_status=data_status, used_cache_fallback=used_cache_fallback)
        return ReportOutput(summary_zh=" ".join(summary_lines), risk_notes=risk_notes)

    @staticmethod
    def _build_risk_notes(*, data_status: DataStatus, used_cache_fallback: bool) -> list[str]:
        notes: list[str] = []
        if used_cache_fallback:
            notes.append("主备数据源异常，本次使用缓存数据，时效性可能下降。")
        if not data_status.is_fresh:
            notes.append("数据未达到最新可用交易日，趋势判断可能偏保守。")
        if data_status.note:
            notes.append(data_status.note)
        notes.append("概率来自统计规则模型，无法覆盖突发事件冲击。")
        notes.append("仅供参考，非投资建议。")
        return notes


def _trend_label_from_probs(item: ScenarioProbs) -> str:
    if item.sideways_pct >= max(item.uptrend_pct, item.downtrend_pct):
        return "不确定"
    return "上行" if item.uptrend_pct >= item.downtrend_pct else "下行"


def _headline_label(
    horizon: str,
    item: ScenarioProbs,
    display_labels: Mapping[str, str] | None,
) -> str:
    if display_labels and horizon in display_labels:
        key = str(display_labels[horizon])
        if key in DISPLAY_LABELS_ZH:
            return DISPLAY_LABELS_ZH[key]
    return _trend_label_from_probs(item)
