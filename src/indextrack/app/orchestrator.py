"""Application orchestrator for data fetch and fallback."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from indextrack.app.models import Candle, DataStatus
from indextrack.infra.freshness import FreshnessGuard
from indextrack.infra.providers.base import ProviderError
from indextrack.infra.providers.router import ProviderFetchResult, ProviderRouter
from indextrack.infra.repository.sqlite_repo import SQLiteRepository


class DataUnavailableError(RuntimeError):
    """Raised when neither providers nor local cache can provide data."""


@dataclass(frozen=True)
class DataFetchOutcome:
    """Result payload for the data stage before analysis."""

    symbol: str
    candles: list[Candle]
    data_status: DataStatus
    source: str
    used_provider_fallback: bool
    used_cache_fallback: bool
    warning: str | None = None


class AnalysisOrchestrator:
    """Coordinates data fetch, freshness checks, and cache fallback."""

    def __init__(
        self,
        *,
        provider_router: ProviderRouter,
        repository: SQLiteRepository,
        freshness_guard: FreshnessGuard,
    ) -> None:
        self._provider_router = provider_router
        self._repository = repository
        self._freshness_guard = freshness_guard

    def fetch_market_data(
        self,
        *,
        symbol: str,
        start: date,
        end: date,
        lookback_days: int = 365,
    ) -> DataFetchOutcome:
        try:
            provider_result = self._provider_router.fetch_daily(symbol=symbol, start=start, end=end)
        except ProviderError as exc:
            return self._fallback_to_cache(
                symbol=symbol,
                lookback_days=lookback_days,
                provider_error=str(exc),
            )

        return self._use_provider_result(
            symbol=symbol,
            provider_result=provider_result,
            lookback_days=lookback_days,
        )

    def _use_provider_result(
        self,
        *,
        symbol: str,
        provider_result: ProviderFetchResult,
        lookback_days: int,
    ) -> DataFetchOutcome:
        self._repository.save_prices(
            symbol=symbol,
            candles=provider_result.candles,
            source=provider_result.source,
        )
        cached_or_recent = self._repository.load_prices(symbol=symbol, lookback_days=lookback_days)
        candles = cached_or_recent or provider_result.candles

        status = self._freshness_guard.build_data_status(
            source=provider_result.source,
            candles=provider_result.candles,
        )
        warning = (
            "主数据源失败，已自动切换备用数据源。"
            if provider_result.used_fallback
            else None
        )
        return DataFetchOutcome(
            symbol=symbol,
            candles=candles,
            data_status=_with_note(status, warning),
            source=provider_result.source,
            used_provider_fallback=provider_result.used_fallback,
            used_cache_fallback=False,
            warning=warning,
        )

    def _fallback_to_cache(
        self,
        *,
        symbol: str,
        lookback_days: int,
        provider_error: str,
    ) -> DataFetchOutcome:
        cached = self._repository.load_prices(symbol=symbol, lookback_days=lookback_days)
        snapshot = self._repository.last_success(symbol=symbol)
        if not cached or snapshot is None:
            raise DataUnavailableError(
                f"主备数据源均失败，且无可用缓存。symbol={symbol}，错误={provider_error}"
            )

        cache_warning = (
            "主备数据源均拉取失败，已回退到最近缓存数据。"
            f" 最近快照日期: {snapshot.last_trade_date.isoformat()}。"
        )
        freshness_status = self._freshness_guard.build_data_status(
            source=snapshot.source,
            candles=cached,
            fetched_at=snapshot.fetched_at,
        )
        combined_note = _join_notes(cache_warning, freshness_status.note)
        status = _with_note(freshness_status, combined_note)

        return DataFetchOutcome(
            symbol=symbol,
            candles=cached,
            data_status=status,
            source=snapshot.source,
            used_provider_fallback=False,
            used_cache_fallback=True,
            warning=combined_note,
        )


def _with_note(status: DataStatus, note: str | None) -> DataStatus:
    if not note:
        return status
    return DataStatus(
        source=status.source,
        last_trade_date=status.last_trade_date,
        fetched_at=status.fetched_at,
        is_fresh=status.is_fresh,
        note=_join_notes(status.note, note),
    )


def _join_notes(left: str | None, right: str | None) -> str | None:
    parts = [item for item in (left, right) if item]
    if not parts:
        return None
    return " ".join(parts)
