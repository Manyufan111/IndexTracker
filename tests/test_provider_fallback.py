"""Tests for provider fallback and cache fallback behavior."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

from indextrack.app.models import Candle
from indextrack.app.orchestrator import AnalysisOrchestrator
from indextrack.infra.freshness import FreshnessGuard
from indextrack.infra.providers.base import ProviderError
from indextrack.infra.providers.router import ProviderFetchResult, ProviderRouter
from indextrack.infra.repository.sqlite_repo import SQLiteRepository


class _FailProvider:
    name = "fail_provider"

    def fetch_daily(self, symbol: str, start: date, end: date) -> list[Candle]:
        raise ProviderError("simulated provider failure")


class _OkProvider:
    name = "ok_provider"

    def fetch_daily(self, symbol: str, start: date, end: date) -> list[Candle]:
        return [
            Candle(
                symbol=symbol,
                trade_date=end,
                open=100,
                high=101,
                low=99,
                close=100.5,
                volume=1000,
                source=self.name,
                fetched_at=datetime(2026, 3, 31, 20, 0, 0),
            )
        ]


class _RouterSuccess:
    def fetch_daily(self, symbol: str, start: date, end: date) -> ProviderFetchResult:
        candles = _OkProvider().fetch_daily(symbol, start, end)
        return ProviderFetchResult(
            candles=candles,
            source="ok_provider",
            used_fallback=False,
            attempts=[],
        )


class _RouterFail:
    def fetch_daily(self, symbol: str, start: date, end: date) -> ProviderFetchResult:
        raise ProviderError("router failed")


class ProviderFallbackTests(unittest.TestCase):
    def test_router_switches_to_secondary_when_primary_fails(self) -> None:
        router = ProviderRouter(_FailProvider(), _OkProvider(), retries=0)
        result = router.fetch_daily(
            symbol="SP500",
            start=date(2026, 3, 1),
            end=date(2026, 3, 31),
        )
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.source, "ok_provider")
        self.assertEqual(len(result.candles), 1)
        self.assertGreaterEqual(len(result.attempts), 2)

    def test_orchestrator_uses_cache_when_router_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "cache.db"
            repository = SQLiteRepository(db_path=str(db_path))
            guard = FreshnessGuard()

            # Seed one successful snapshot first.
            seed_orchestrator = AnalysisOrchestrator(
                provider_router=_RouterSuccess(),  # type: ignore[arg-type]
                repository=repository,
                freshness_guard=guard,
            )
            seed_orchestrator.fetch_market_data(
                symbol="SP500",
                start=date(2026, 3, 1),
                end=date(2026, 3, 31),
                lookback_days=365,
            )

            # Then fail router and verify cache fallback.
            fail_orchestrator = AnalysisOrchestrator(
                provider_router=_RouterFail(),  # type: ignore[arg-type]
                repository=repository,
                freshness_guard=guard,
            )
            outcome = fail_orchestrator.fetch_market_data(
                symbol="SP500",
                start=date(2026, 3, 1),
                end=date(2026, 3, 31),
                lookback_days=365,
            )
            self.assertTrue(outcome.used_cache_fallback)
            self.assertGreaterEqual(len(outcome.candles), 1)
            self.assertIsNotNone(outcome.warning)


if __name__ == "__main__":
    unittest.main()
