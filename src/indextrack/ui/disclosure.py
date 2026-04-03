"""Collapsible disclosure block renderer."""

from __future__ import annotations

from html import escape


def render_disclosure(*, source_line: str, method_line: str) -> str:
    """Render collapsed-by-default details for data source and methods."""
    return f"""
<details class="disclosure">
  <summary>查看数据来源与分析方法</summary>
  <div class="disclosure-body">
    <p><strong>数据来源：</strong>{escape(source_line)}</p>
    <p><strong>分析方法：</strong>{escape(method_line)}</p>
  </div>
</details>
""".strip()

