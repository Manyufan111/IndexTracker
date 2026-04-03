"""Primary market data provider based on Yahoo Finance chart API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from indextrack.app.models import Candle
from indextrack.infra.providers.base import (
    MarketDataProvider,
    ProviderError,
    ProviderResponseError,
    UnsupportedSymbolError,
)

YAHOO_PRIMARY_BASE_URL = "https://query2.finance.yahoo.com/v8/finance/chart"
DEFAULT_USER_AGENT = "Mozilla/5.0 (IndexTrack/0.1)"

_SYMBOL_ALIASES = {
    "SP500": "^GSPC",
    "SPX": "^GSPC",
    "GSPC": "^GSPC",
    "^GSPC": "^GSPC",
    "NASDAQ": "^IXIC",
    "NASDAQCOMPOSITE": "^IXIC",
    "IXIC": "^IXIC",
    "^IXIC": "^IXIC",
}


@dataclass(frozen=True)
class YahooFinancePrimaryProvider(MarketDataProvider):
    """Fetch daily candles for SP500/Nasdaq from Yahoo Finance."""

    request_timeout_sec: int = 15
    user_agent: str = DEFAULT_USER_AGENT
    base_url: str = YAHOO_PRIMARY_BASE_URL

    @property
    def name(self) -> str:
        return "yahoo_finance_primary"

    def fetch_daily(self, symbol: str, start: date, end: date) -> list[Candle]:
        if end < start:
            raise ProviderError("end 不能早于 start")

        yahoo_symbol = self._to_yahoo_symbol(symbol)
        payload = self._request_chart(yahoo_symbol, start, end)
        return self._parse_candles(payload, symbol=symbol.upper().strip(), start=start, end=end)

    def _request_chart(self, yahoo_symbol: str, start: date, end: date) -> dict[str, Any]:
        params = {
            "interval": "1d",
            "includePrePost": "false",
            # Yahoo's period2 is exclusive; +1 day keeps end date inclusive.
            "period1": str(_to_epoch_seconds(start)),
            "period2": str(_to_epoch_seconds(end + timedelta(days=1))),
        }
        url = f"{self.base_url}/{quote(yahoo_symbol, safe='')}" + "?" + urlencode(params)
        request = Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urlopen(request, timeout=self.request_timeout_sec) as response:
                content = response.read().decode("utf-8")
        except HTTPError as exc:
            raise ProviderError(f"Yahoo 请求失败: HTTP {exc.code}") from exc
        except URLError as exc:
            raise ProviderError(f"Yahoo 请求失败: {exc.reason}") from exc

        try:
            payload: dict[str, Any] = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError("Yahoo 返回内容不是有效 JSON") from exc
        return payload

    def _parse_candles(
        self,
        payload: dict[str, Any],
        *,
        symbol: str,
        start: date,
        end: date,
    ) -> list[Candle]:
        chart = payload.get("chart")
        if not isinstance(chart, dict):
            raise ProviderResponseError("Yahoo 返回缺少 chart 字段")

        error = chart.get("error")
        if error:
            description = error.get("description", "unknown error")
            raise ProviderResponseError(f"Yahoo 返回业务错误: {description}")

        results = chart.get("result")
        if not isinstance(results, list) or not results:
            raise ProviderResponseError("Yahoo 返回 result 为空")

        result = results[0]
        timestamps = result.get("timestamp")
        indicators = result.get("indicators")
        if not isinstance(timestamps, list) or not isinstance(indicators, dict):
            raise ProviderResponseError("Yahoo 返回缺少 timestamp 或 indicators")

        quote_entries = indicators.get("quote")
        if not isinstance(quote_entries, list) or not quote_entries:
            raise ProviderResponseError("Yahoo 返回缺少 quote 数据")

        quote_data = quote_entries[0]
        opens = _as_float_list(quote_data.get("open"))
        highs = _as_float_list(quote_data.get("high"))
        lows = _as_float_list(quote_data.get("low"))
        closes = _as_float_list(quote_data.get("close"))
        volumes = _as_float_list(quote_data.get("volume"))
        min_len = min(len(timestamps), len(opens), len(highs), len(lows), len(closes), len(volumes))
        fetched_at = datetime.now(tz=timezone.utc)

        candles: list[Candle] = []
        for idx in range(min_len):
            open_v = opens[idx]
            high_v = highs[idx]
            low_v = lows[idx]
            close_v = closes[idx]
            if any(value is None for value in (open_v, high_v, low_v, close_v)):
                continue

            trade_date = datetime.fromtimestamp(int(timestamps[idx]), tz=timezone.utc).date()
            if trade_date < start or trade_date > end:
                continue

            volume = volumes[idx]
            candles.append(
                Candle(
                    symbol=symbol,
                    trade_date=trade_date,
                    open=float(open_v),
                    high=float(high_v),
                    low=float(low_v),
                    close=float(close_v),
                    volume=float(volume) if volume is not None else None,
                    source=self.name,
                    fetched_at=fetched_at,
                )
            )

        if not candles:
            raise ProviderResponseError(f"Yahoo 未返回有效日线数据: {symbol}")
        candles.sort(key=lambda item: item.trade_date)
        return candles

    @staticmethod
    def _to_yahoo_symbol(symbol: str) -> str:
        normalized = (
            symbol.strip()
            .upper()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
            .replace("&", "")
        )
        mapped = _SYMBOL_ALIASES.get(normalized)
        if mapped:
            return mapped
        raise UnsupportedSymbolError(f"不支持的指数标识: {symbol}")


def _to_epoch_seconds(value: date) -> int:
    return int(datetime.combine(value, time.min, tzinfo=timezone.utc).timestamp())


def _as_float_list(values: Any) -> list[float | None]:
    if not isinstance(values, list):
        return []
    result: list[float | None] = []
    for value in values:
        if value is None:
            result.append(None)
            continue
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            result.append(None)
    return result
