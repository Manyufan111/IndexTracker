"""Yahoo quote metadata provider for UI extras (PE + VIX)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from indextrack.infra.providers.base import ProviderError, ProviderResponseError
from indextrack.infra.providers.primary import DEFAULT_USER_AGENT

_QUOTE_PATH = "/v7/finance/quote"
_SUMMARY_PATH = "/v10/finance/quoteSummary"
_ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
_BASE_URLS: tuple[tuple[str, str], ...] = (
    ("https://query2.finance.yahoo.com", "yahoo_finance_primary"),
    ("https://query1.finance.yahoo.com", "yahoo_finance_secondary"),
)
_FRED_VIX_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
_PE_PROXY_BY_SYMBOL = {
    "SP500": "SPY",
    "NASDAQ": "QQQ",
}
_ALPHA_PE_PROXY = {
    "SP500": "SPY",
    "NASDAQ": "QQQ",
}
_SEED_FILE_ENV = "INDEXTRACK_META_SEED_FILE"
_DEFAULT_SEED_FILE = ".data/market_meta_seed.json"
_ALPHA_VANTAGE_KEY_ENV = "INDEXTRACK_ALPHA_VANTAGE_API_KEY"
_DEFAULT_BUNDLED_SEED_FILE = "meta_seed_defaults.json"

_SYMBOL_ALIASES = {
    "SP500": "^GSPC",
    "SPX": "^GSPC",
    "GSPC": "^GSPC",
    "^GSPC": "^GSPC",
    "NASDAQ": "^IXIC",
    "NASDAQCOMPOSITE": "^IXIC",
    "IXIC": "^IXIC",
    "^IXIC": "^IXIC",
    "VIX": "^VIX",
    "^VIX": "^VIX",
    "SPY": "SPY",
    "QQQ": "QQQ",
}


@dataclass(frozen=True)
class DashboardMarketMeta:
    """Extra market metadata used by UI rendering."""

    pe_by_symbol: dict[str, float | None]
    pe_source_by_symbol: dict[str, str]
    vix: float | None
    vix_source: str
    source: str
    fetched_at: datetime
    note: str | None = None


@dataclass(frozen=True)
class YahooFinanceQuoteProvider:
    """Fetch quote metadata with primary/secondary Yahoo fallback."""

    request_timeout_sec: int = 15
    user_agent: str = DEFAULT_USER_AGENT

    def fetch_dashboard_meta(self, symbols: list[str]) -> DashboardMarketMeta:
        normalized_symbols = [self._normalize_symbol(item) for item in symbols]
        yahoo_symbols = [self._to_yahoo_symbol(item) for item in normalized_symbols]
        proxy_symbols = [
            self._to_yahoo_symbol(_PE_PROXY_BY_SYMBOL[symbol])
            for symbol in normalized_symbols
            if symbol in _PE_PROXY_BY_SYMBOL
        ]
        all_quote_symbols = sorted(set([*yahoo_symbols, *proxy_symbols, "^VIX"]))

        quote_payload: dict[str, Any] | None = None
        source_name = "unavailable"
        last_error: Exception | None = None
        used_base_urls: list[tuple[str, str]] = []
        for base_url, source in _BASE_URLS:
            used_base_urls.append((base_url, source))
            try:
                quote_payload = self._request_quote(base_url=base_url, yahoo_symbols=all_quote_symbols)
                source_name = f"{source}_quote"
                break
            except ProviderError as exc:
                last_error = exc
                continue

        if quote_payload is None:
            if last_error is None:
                raise ProviderError("无法拉取 PE/VIX 数据：未知错误")
            raise ProviderError(f"无法拉取 PE/VIX 数据: {last_error}") from last_error

        quote_map = self._parse_quote_map(quote_payload)
        pe_by_symbol: dict[str, float | None] = {}
        pe_source_by_symbol: dict[str, str] = {}
        summary_failed_symbols: list[str] = []
        for symbol, yahoo_symbol in zip(normalized_symbols, yahoo_symbols):
            pe_value = self._extract_numeric(
                quote_map.get(yahoo_symbol),
                ("trailingPE", "forwardPE"),
            )
            pe_source = f"{source_name}:index_quote"
            if pe_value is None:
                pe_value = self._fetch_pe_from_summary(yahoo_symbol=yahoo_symbol, base_urls=used_base_urls)
                if pe_value is not None:
                    pe_source = f"{source_name}:index_summary"
            if pe_value is None:
                proxy = _PE_PROXY_BY_SYMBOL.get(symbol)
                proxy_yahoo_symbol = self._to_yahoo_symbol(proxy) if proxy else None
                if proxy_yahoo_symbol:
                    pe_value = self._extract_numeric(
                        quote_map.get(proxy_yahoo_symbol),
                        ("trailingPE", "forwardPE"),
                    )
                    if pe_value is not None:
                        pe_source = f"{source_name}:proxy_quote({proxy_yahoo_symbol})"
            if pe_value is None:
                proxy = _PE_PROXY_BY_SYMBOL.get(symbol)
                proxy_yahoo_symbol = self._to_yahoo_symbol(proxy) if proxy else None
                if proxy_yahoo_symbol:
                    pe_value = self._fetch_pe_from_summary(
                        yahoo_symbol=proxy_yahoo_symbol,
                        base_urls=used_base_urls,
                    )
                    if pe_value is not None:
                        pe_source = f"{source_name}:proxy_summary({proxy_yahoo_symbol})"
            if pe_value is None:
                pe_source = "unavailable"
                summary_failed_symbols.append(symbol)
            pe_by_symbol[symbol] = pe_value
            pe_source_by_symbol[symbol] = pe_source

        vix_value = self._extract_numeric(
            quote_map.get("^VIX"),
            ("regularMarketPrice", "regularMarketPreviousClose", "previousClose"),
        )
        vix_source = f"{source_name}:quote(^VIX)" if vix_value is not None else "unavailable"
        fetched_at = datetime.now(tz=timezone.utc)
        note = None
        if summary_failed_symbols:
            note = (
                "以下指数未获取到有效 PE: " + ",".join(sorted(summary_failed_symbols))
            )
        return DashboardMarketMeta(
            pe_by_symbol=pe_by_symbol,
            pe_source_by_symbol=pe_source_by_symbol,
            vix=vix_value,
            vix_source=vix_source,
            source=source_name,
            fetched_at=fetched_at,
            note=note,
        )

    def fetch_vix_from_fred(self) -> float | None:
        payload = self._request_text(_FRED_VIX_CSV_URL)
        lines = [line.strip() for line in payload.splitlines() if line.strip()]
        if len(lines) < 2:
            raise ProviderResponseError("FRED VIX 数据为空")
        # CSV 格式: DATE,VIXCLS
        for line in reversed(lines[1:]):
            if "," not in line:
                continue
            _date_text, value_text = line.split(",", 1)
            value_text = value_text.strip()
            if value_text in {"", ".", "nan", "NaN"}:
                continue
            try:
                return float(value_text)
            except ValueError:
                continue
        raise ProviderResponseError("FRED VIX 无可用数值")

    def fetch_pe_from_alpha_vantage(self, symbol: str) -> tuple[float | None, str]:
        api_key = self._resolve_alpha_vantage_key()
        if not api_key:
            return None, "alpha_vantage:disabled(no_api_key)"
        proxy_symbol = _ALPHA_PE_PROXY.get(symbol.upper())
        if proxy_symbol is None:
            return None, "alpha_vantage:unsupported_symbol"
        params = {
            "function": "OVERVIEW",
            "symbol": proxy_symbol,
            "apikey": api_key,
        }
        url = f"{_ALPHA_VANTAGE_URL}?" + urlencode(params)
        try:
            payload = self._request_json(url)
        except ProviderError as exc:
            return None, f"alpha_vantage:error({exc})"
        value = self._extract_numeric(payload, ("PERatio",))
        if value is None:
            return None, "alpha_vantage:missing_peratio"
        return value, f"alpha_vantage:overview({proxy_symbol})"

    def fetch_vix_from_alpha_vantage(self) -> tuple[float | None, str]:
        api_key = self._resolve_alpha_vantage_key()
        if not api_key:
            return None, "alpha_vantage:disabled(no_api_key)"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": "VIX",
            "apikey": api_key,
        }
        url = f"{_ALPHA_VANTAGE_URL}?" + urlencode(params)
        try:
            payload = self._request_json(url)
        except ProviderError as exc:
            return None, f"alpha_vantage:error({exc})"
        value = self._extract_numeric(
            payload,
            ("Global Quote.05. price", "Global Quote.08. previous close"),
        )
        if value is None:
            return None, "alpha_vantage:missing_quote"
        return value, "alpha_vantage:global_quote(VIX)"

    def load_meta_seed(self) -> dict[str, Any]:
        seed_path = Path(os.environ.get(_SEED_FILE_ENV, _DEFAULT_SEED_FILE))
        if not seed_path.exists():
            bundled = Path(__file__).with_name(_DEFAULT_BUNDLED_SEED_FILE)
            if not bundled.exists():
                return {}
            try:
                payload = json.loads(bundled.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}
            if isinstance(payload, dict):
                payload["_seed_origin"] = "bundled_default"
                return payload
            return {}
        try:
            payload = json.loads(seed_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if isinstance(payload, dict):
            payload["_seed_origin"] = str(seed_path)
            return payload
        return {}

    def read_pe_from_seed(self, symbol: str) -> tuple[float | None, str]:
        payload = self.load_meta_seed()
        pe_obj = payload.get("pe")
        if not isinstance(pe_obj, dict):
            return None, "seed:missing(pe)"
        value = pe_obj.get(symbol.upper())
        numeric = self._to_float(value)
        if numeric is None:
            return None, f"seed:missing(pe:{symbol.upper()})"
        origin = payload.get("_seed_origin", "unknown")
        return numeric, f"seed:file(pe:{origin})"

    def read_vix_from_seed(self) -> tuple[float | None, str]:
        payload = self.load_meta_seed()
        numeric = self._to_float(payload.get("vix"))
        if numeric is None:
            return None, "seed:missing(vix)"
        origin = payload.get("_seed_origin", "unknown")
        return numeric, f"seed:file(vix:{origin})"

    def _request_quote(self, *, base_url: str, yahoo_symbols: list[str]) -> dict[str, Any]:
        params = {"symbols": ",".join(yahoo_symbols)}
        url = f"{base_url}{_QUOTE_PATH}?" + urlencode(params)
        return self._request_json(url)

    def _fetch_pe_from_summary(
        self,
        *,
        yahoo_symbol: str,
        base_urls: list[tuple[str, str]],
    ) -> float | None:
        params = {"modules": "summaryDetail,defaultKeyStatistics,financialData"}
        suffix = quote(yahoo_symbol, safe="")
        query = urlencode(params)
        for base_url, _source in base_urls:
            url = f"{base_url}{_SUMMARY_PATH}/{suffix}?{query}"
            try:
                payload = self._request_json(url)
            except ProviderError:
                continue
            result = self._parse_quote_summary(payload)
            value = self._extract_numeric(
                result,
                (
                    "summaryDetail.trailingPE",
                    "summaryDetail.forwardPE",
                    "defaultKeyStatistics.trailingPE",
                    "defaultKeyStatistics.forwardPE",
                ),
            )
            if value is not None:
                return value
        return None

    def _request_json(self, url: str) -> dict[str, Any]:
        request = Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urlopen(request, timeout=self.request_timeout_sec) as response:
                content = response.read().decode("utf-8")
        except HTTPError as exc:
            raise ProviderError(f"Yahoo 元数据请求失败: HTTP {exc.code}") from exc
        except URLError as exc:
            raise ProviderError(f"Yahoo 元数据请求失败: {exc.reason}") from exc

        try:
            payload: dict[str, Any] = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError("Yahoo 元数据响应不是有效 JSON") from exc
        return payload

    @staticmethod
    def _resolve_alpha_vantage_key() -> str | None:
        value = os.environ.get(_ALPHA_VANTAGE_KEY_ENV, "").strip()
        return value or None

    def _request_text(self, url: str) -> str:
        request = Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urlopen(request, timeout=self.request_timeout_sec) as response:
                return response.read().decode("utf-8")
        except HTTPError as exc:
            raise ProviderError(f"文本数据请求失败: HTTP {exc.code}") from exc
        except URLError as exc:
            raise ProviderError(f"文本数据请求失败: {exc.reason}") from exc

    @staticmethod
    def _parse_quote_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
        quote_response = payload.get("quoteResponse")
        if not isinstance(quote_response, dict):
            raise ProviderResponseError("Yahoo quote 响应缺少 quoteResponse")
        results = quote_response.get("result")
        if not isinstance(results, list):
            raise ProviderResponseError("Yahoo quote 响应缺少 result 列表")

        mapping: dict[str, dict[str, Any]] = {}
        for item in results:
            if not isinstance(item, dict):
                continue
            symbol = item.get("symbol")
            if isinstance(symbol, str) and symbol:
                mapping[symbol] = item
        return mapping

    @staticmethod
    def _parse_quote_summary(payload: dict[str, Any]) -> dict[str, Any]:
        quote_summary = payload.get("quoteSummary")
        if not isinstance(quote_summary, dict):
            return {}
        error = quote_summary.get("error")
        if error:
            return {}
        results = quote_summary.get("result")
        if not isinstance(results, list) or not results:
            return {}
        result = results[0]
        return result if isinstance(result, dict) else {}

    @classmethod
    def _extract_numeric(
        cls,
        payload: dict[str, Any] | None,
        keys: tuple[str, ...],
    ) -> float | None:
        if not isinstance(payload, dict):
            return None
        for key in keys:
            value: Any = payload
            for token in key.split("."):
                if not isinstance(value, dict):
                    value = None
                    break
                value = value.get(token)
            numeric = cls._to_float(value)
            if numeric is not None:
                return numeric
        return None

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, dict):
            raw = value.get("raw")
            if raw is None:
                return None
            try:
                return float(raw)
            except (TypeError, ValueError):
                return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        normalized = symbol.strip().upper()
        if not normalized:
            raise ProviderError("symbol 不能为空")
        return normalized

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
        raise ProviderError(f"不支持的指数标识: {symbol}")
