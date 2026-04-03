"""Simple SVG line chart renderer for UI pages."""

from __future__ import annotations

from html import escape

from indextrack.ui.view_model import ChartPoint


def render_trend_chart(
    *,
    title: str,
    points: list[ChartPoint],
    line_color: str,
    width: int = 760,
    height: int = 280,
) -> str:
    """Render a lightweight SVG trend chart."""
    if len(points) < 2:
        return (
            '<div class="chart-empty">'
            f"{escape(title)}：暂无足够数据绘制趋势图。"
            "</div>"
        )

    pad_x = 32.0
    pad_y = 24.0
    usable_w = width - 2 * pad_x
    usable_h = height - 2 * pad_y

    min_close = min(item.close for item in points)
    max_close = max(item.close for item in points)
    if max_close == min_close:
        max_close += 1.0

    point_pairs: list[tuple[float, float]] = []
    total = len(points)
    for idx, point in enumerate(points):
        x = pad_x + (idx / (total - 1)) * usable_w
        y = pad_y + (1 - (point.close - min_close) / (max_close - min_close)) * usable_h
        point_pairs.append((x, y))

    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in point_pairs)
    first_date = escape(points[0].date_label)
    last_date = escape(points[-1].date_label)
    latest = points[-1].close

    return f"""
<div class="chart-wrap">
  <svg class="trend-chart" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)} 趋势图">
    <line x1="{pad_x}" y1="{pad_y}" x2="{pad_x}" y2="{height - pad_y}" class="axis-line"></line>
    <line x1="{pad_x}" y1="{height - pad_y}" x2="{width - pad_x}" y2="{height - pad_y}" class="axis-line"></line>
    <polyline points="{polyline}" fill="none" stroke="{line_color}" stroke-width="2.4"></polyline>
  </svg>
  <div class="chart-meta">
    <span>{first_date}</span>
    <span>最新收盘: {latest:.2f}</span>
    <span>{last_date}</span>
  </div>
</div>
""".strip()

