"""Analysis text panel renderer for UI."""

from __future__ import annotations

from html import escape


def render_analysis_panel(
    *,
    summary_zh: str,
    scenario_lines: list[str],
    probability_detail_lines: list[str],
) -> str:
    """Render the analysis section below trend chart."""
    summary = escape(summary_zh)
    scenario_html = "".join(f"<li>{escape(item)}</li>" for item in scenario_lines)
    detail_html = "".join(f"<li>{escape(item)}</li>" for item in probability_detail_lines)
    return f"""
<section class="analysis-panel">
  <h3>走势分析</h3>
  <p>{summary}</p>
  <ul>{scenario_html}</ul>
  <ul class="probability-detail-list">{detail_html}</ul>
</section>
""".strip()
