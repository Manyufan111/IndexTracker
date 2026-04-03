"""Provider router with primary/secondary fallback and retry."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from time import sleep
from typing import Iterable

from indextrack.app.models import Candle
from indextrack.infra.providers.base import MarketDataProvider, ProviderError


@dataclass(frozen=True)
class ProviderAttempt:
    """Execution snapshot for one provider attempt."""

    provider: str
    success: bool
    error: str | None = None


@dataclass(frozen=True)
class ProviderFetchResult:
    """Normalized fetch result after routing and fallback."""

    candles: list[Candle]
    source: str
    used_fallback: bool
    attempts: list[ProviderAttempt] = field(default_factory=list)


class ProviderRouter:
    """Try primary first, then fallback to secondary provider."""

    def __init__(
        self,
        primary: MarketDataProvider,
        secondary: MarketDataProvider,
        *,
        retries: int = 1,
        retry_delay_sec: float = 0.0,
    ) -> None:
        if retries < 0:
            raise ValueError("retries 不能为负数")
        if retry_delay_sec < 0:
            raise ValueError("retry_delay_sec 不能为负数")

        self._primary = primary
        self._secondary = secondary
        self._retries = retries
        self._retry_delay_sec = retry_delay_sec

    def fetch_daily(self, symbol: str, start: date, end: date) -> ProviderFetchResult:
        attempts: list[ProviderAttempt] = []

        primary_result = self._fetch_with_retry(
            provider=self._primary,
            symbol=symbol,
            start=start,
            end=end,
            attempts=attempts,
        )
        if primary_result is not None:
            return ProviderFetchResult(
                candles=primary_result,
                source=self._primary.name,
                used_fallback=False,
                attempts=attempts,
            )

        secondary_result = self._fetch_with_retry(
            provider=self._secondary,
            symbol=symbol,
            start=start,
            end=end,
            attempts=attempts,
        )
        if secondary_result is not None:
            return ProviderFetchResult(
                candles=secondary_result,
                source=self._secondary.name,
                used_fallback=True,
                attempts=attempts,
            )

        raise ProviderError(self._build_error_message(symbol=symbol, attempts=attempts))

    def _fetch_with_retry(
        self,
        *,
        provider: MarketDataProvider,
        symbol: str,
        start: date,
        end: date,
        attempts: list[ProviderAttempt],
    ) -> list[Candle] | None:
        for attempt_index in range(self._retries + 1):
            try:
                candles = provider.fetch_daily(symbol=symbol, start=start, end=end)
            except Exception as exc:  # noqa: BLE001
                attempts.append(
                    ProviderAttempt(
                        provider=provider.name,
                        success=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                if attempt_index < self._retries and self._retry_delay_sec > 0:
                    sleep(self._retry_delay_sec)
                continue

            attempts.append(ProviderAttempt(provider=provider.name, success=True))
            return candles
        return None

    @staticmethod
    def _build_error_message(symbol: str, attempts: Iterable[ProviderAttempt]) -> str:
        detail_items: list[str] = []
        seen: set[str] = set()
        for attempt in attempts:
            if attempt.success:
                continue
            item = f"{attempt.provider} -> {attempt.error or 'unknown error'}"
            if item in seen:
                continue
            seen.add(item)
            detail_items.append(item)
        details = "; ".join(detail_items)
        if not details:
            details = "no attempt details"
        return f"主备数据源均拉取失败: {symbol}. 详情: {details}"
