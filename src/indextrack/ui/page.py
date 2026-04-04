"""HTML page and lightweight UI server for IndexTrack."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from html import escape
from typing import Callable
from urllib.parse import parse_qs, urlparse

from indextrack.ui.analysis_panel import render_analysis_panel
from indextrack.ui.charts import render_trend_chart
from indextrack.ui.disclosure import render_disclosure
from indextrack.ui.view_model import IndexCardViewModel


def render_dashboard_page(
    *,
    cards: list[IndexCardViewModel],
    period: str,
    generated_at: str,
    vix_value: float | None = None,
    vix_source: str | None = None,
    model_mode: str = "quantile",
    ui_notice: str | None = None,
) -> str:
    """Render complete HTML dashboard for both indices."""
    normalized_model = _normalize_model_mode(model_mode)
    cards_html = "".join(_render_card(item) for item in cards)
    period_tabs = _render_period_tabs(period=period, model_mode=normalized_model)
    model_tabs = ""
    notice_html = ""
    vix_text = "N/A" if vix_value is None else f"{vix_value:.2f}"
    vix_source_text = vix_source or "unavailable"
    if ui_notice:
        notice_html = f'<div class="notice">{escape(ui_notice)}</div>'
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IndexTrack 趋势面板</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --card: #ffffff;
      --text: #16202a;
      --muted: #5f6b77;
      --line: #d9e1ea;
      --blue: #2f6fed;
      --teal: #0f9d86;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Noto Sans CJK SC", sans-serif;
      background: linear-gradient(180deg, #f3f7ff 0%, var(--bg) 60%);
      color: var(--text);
    }}
    .container {{
      max-width: 980px;
      margin: 0 auto;
      padding: 28px 16px 36px;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 12px;
      margin-bottom: 14px;
    }}
    .header h1 {{
      margin: 0;
      font-size: 26px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
    }}
    .header-right {{
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      gap: 4px;
    }}
    .vix {{
      color: #9a3b00;
      font-size: 13px;
      background: #fff1e6;
      border: 1px solid #ffd7bc;
      border-radius: 999px;
      padding: 3px 10px;
    }}
    .vix-source {{
      color: var(--muted);
      font-size: 12px;
    }}
    .tabs {{
      display: flex;
      gap: 8px;
      margin: 12px 0 20px;
      flex-wrap: wrap;
    }}
    .tab {{
      padding: 6px 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      text-decoration: none;
      color: var(--muted);
      font-size: 13px;
      background: #fff;
    }}
    .tab.active {{
      color: #fff;
      background: var(--blue);
      border-color: var(--blue);
    }}
    .notice {{
      margin: 0 0 14px;
      border: 1px solid #f3d6a4;
      background: #fff8ea;
      color: #7d4f00;
      border-radius: 10px;
      padding: 8px 10px;
      font-size: 13px;
      line-height: 1.4;
    }}
    .card {{
      background: var(--card);
      border: 1px solid #e5ecf4;
      border-radius: 14px;
      padding: 14px 14px 10px;
      margin-bottom: 14px;
      box-shadow: 0 6px 22px rgba(7, 42, 87, 0.05);
    }}
    .card h2 {{
      margin: 2px 0 10px;
      font-size: 19px;
    }}
    .trend-chart {{
      width: 100%;
      height: auto;
      display: block;
      background: linear-gradient(180deg, #fbfdff 0%, #f4f8ff 100%);
      border-radius: 10px;
      border: 1px solid #e5edf7;
    }}
    .axis-line {{
      stroke: #d4deea;
      stroke-width: 1;
    }}
    .chart-meta {{
      margin-top: 6px;
      display: flex;
      justify-content: space-between;
      color: var(--muted);
      font-size: 12px;
    }}
    .analysis-panel {{
      margin-top: 12px;
      background: #fbfdff;
      border: 1px solid #e8eef6;
      border-radius: 10px;
      padding: 10px 12px;
    }}
    .analysis-panel h3 {{
      margin: 0 0 8px;
      font-size: 15px;
    }}
    .analysis-panel p {{
      margin: 0;
      color: #243647;
      line-height: 1.5;
      font-size: 14px;
    }}
    .analysis-panel ul {{
      margin: 8px 0 0;
      padding-left: 18px;
      color: #35506a;
      font-size: 13px;
    }}
    .analysis-panel .probability-detail-list {{
      margin-top: 8px;
      color: #496683;
      font-size: 12px;
    }}
    .disclosure {{
      margin-top: 10px;
      border: 1px solid #e7edf5;
      border-radius: 10px;
      background: #ffffff;
      padding: 8px 10px;
    }}
    .disclosure summary {{
      cursor: pointer;
      color: #1f3d5b;
      font-size: 13px;
      font-weight: 600;
    }}
    .disclosure-body {{
      margin-top: 8px;
      color: #45617d;
      font-size: 13px;
      line-height: 1.5;
    }}
    .disclosure-body p {{
      margin: 6px 0;
    }}
    .chart-empty {{
      margin: 8px 0;
      color: var(--muted);
      font-size: 14px;
      background: #f7f9fc;
      border: 1px dashed #d8e1ec;
      border-radius: 10px;
      padding: 18px;
    }}
    .valuation-meta {{
      margin-top: 8px;
      color: #3a556f;
      font-size: 12px;
      font-weight: 600;
    }}
    .valuation-source {{
      margin-top: 4px;
      color: var(--muted);
      font-size: 11px;
    }}
  </style>
</head>
<body>
  <main class="container">
    <header class="header">
      <h1>IndexTrack 趋势总览</h1>
      <div class="header-right">
        <div class="vix">恐慌指数 VIX：{escape(vix_text)}</div>
        <div class="vix-source">VIX 来源：{escape(vix_source_text)}</div>
        <div class="meta">生成时间：{escape(generated_at)}</div>
      </div>
    </header>
    {period_tabs}
    {model_tabs}
    {notice_html}
    {cards_html}
  </main>
</body>
</html>"""


def run_ui_server(
    *,
    host: str,
    port: int,
    build_dashboard: Callable[[str, str], str],
    default_model: str = "quantile",
) -> None:
    """Run simple threaded HTTP server for UI."""

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/healthz":
                self._write_plain(200, "ok")
                return
            if parsed.path != "/":
                self._write_plain(404, "not found")
                return

            query = parse_qs(parsed.query)
            period = query.get("period", ["1M"])[0].strip().upper()
            model_mode = _normalize_model_mode(query.get("model", [default_model])[0])
            try:
                html = build_dashboard(period, model_mode)
            except Exception as exc:  # noqa: BLE001
                self._write_plain(500, f"ui render failed: {exc}")
                return
            self._write_html(200, html)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            # Keep HTTP handler quiet; app logger handles major events.
            return

        def _write_html(self, status: int, body: str) -> None:
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _write_plain(self, status: int, body: str) -> None:
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer((host, port), _Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _render_card(card: IndexCardViewModel) -> str:
    color = "#2f6fed" if card.symbol.upper() == "SP500" else "#0f9d86"
    chart_html = render_trend_chart(title=card.title, points=card.points, line_color=color)
    analysis_html = render_analysis_panel(
        summary_zh=card.summary_zh,
        scenario_lines=card.scenario_lines,
        probability_detail_lines=card.probability_detail_lines,
    )
    disclosure_html = render_disclosure(source_line=card.source_line, method_line=card.method_line)
    return f"""
<section class="card">
  <h2>{escape(card.title)}（{escape(card.period)}）</h2>
  {chart_html}
  <div class="valuation-meta">{escape(card.pe_line)}</div>
  <div class="valuation-source">{escape(card.pe_source_line)}</div>
  {analysis_html}
  {disclosure_html}
</section>
""".strip()


def _render_period_tabs(*, period: str, model_mode: str) -> str:
    periods = ["1M", "3M", "6M", "1Y"]
    html = ['<nav class="tabs" aria-label="周期选择">']
    for tab_period in periods:
        class_name = "tab active" if tab_period == period else "tab"
        href = f"/?period={tab_period}&model={model_mode}"
        html.append(
            f'<a class="{class_name}" href="{href}">{tab_period}</a>'
        )
    html.append("</nav>")
    return "".join(html)


def _normalize_model_mode(value: str) -> str:
    return "quantile"
