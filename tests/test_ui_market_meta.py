"""Tests for UI market meta resolver (quote/fallback/cache chain)."""

from __future__ import annotations

import logging
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

from indextrack.cli import _fetch_ui_market_meta
from indextrack.infra.providers.quote import DashboardMarketMeta, YahooFinanceQuoteProvider
from indextrack.infra.repository.sqlite_repo import SQLiteRepository


class UIMarketMetaTests(unittest.TestCase):
    def test_resolver_uses_cache_when_quote_and_fred_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = SQLiteRepository(str(Path(temp_dir) / "indextrack.db"))
            now = datetime(2026, 4, 3, 18, 0, 0, tzinfo=timezone.utc)
            repo.save_market_meta(
                key="PE_SP500",
                value=23.1,
                source="yahoo:index_quote",
                fetched_at=now,
            )
            repo.save_market_meta(
                key="VIX",
                value=20.7,
                source="fred_vixcls_csv",
                fetched_at=now,
            )

            provider = Mock()
            provider.fetch_dashboard_meta.return_value = DashboardMarketMeta(
                pe_by_symbol={"SP500": None, "NASDAQ": 31.2},
                pe_source_by_symbol={"SP500": "unavailable", "NASDAQ": "yahoo:index_quote"},
                vix=None,
                vix_source="unavailable",
                source="yahoo_quote",
                fetched_at=now,
                note=None,
            )
            provider.fetch_pe_from_alpha_vantage.return_value = (None, "alpha_vantage:disabled(no_api_key)")
            provider.read_pe_from_seed.return_value = (None, "seed:missing(pe)")
            provider.read_vix_from_seed.return_value = (None, "seed:missing(vix)")
            provider.fetch_vix_from_alpha_vantage.return_value = (None, "alpha_vantage:disabled(no_api_key)")
            provider.fetch_vix_from_fred.side_effect = RuntimeError("fred down")

            meta = _fetch_ui_market_meta(
                quote_provider=provider,
                repository=repo,
                logger=logging.getLogger("test.ui.meta"),
            )

            self.assertEqual(meta.pe_by_symbol["SP500"], 23.1)
            self.assertEqual(meta.pe_source_by_symbol["SP500"], "cache:yahoo:index_quote")
            self.assertEqual(meta.pe_by_symbol["NASDAQ"], 31.2)
            self.assertEqual(meta.vix, 20.7)
            self.assertEqual(meta.vix_source, "cache:fred_vixcls_csv")

    def test_resolver_uses_bundled_seed_when_no_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = SQLiteRepository(str(Path(temp_dir) / "indextrack.db"))
            now = datetime(2026, 4, 3, 18, 0, 0, tzinfo=timezone.utc)

            provider = Mock()
            provider.fetch_dashboard_meta.return_value = DashboardMarketMeta(
                pe_by_symbol={"SP500": None, "NASDAQ": None},
                pe_source_by_symbol={"SP500": "unavailable", "NASDAQ": "unavailable"},
                vix=None,
                vix_source="unavailable",
                source="unavailable",
                fetched_at=now,
                note="quote failed",
            )
            provider.fetch_pe_from_alpha_vantage.return_value = (None, "alpha_vantage:disabled(no_api_key)")
            provider.fetch_vix_from_alpha_vantage.return_value = (None, "alpha_vantage:disabled(no_api_key)")
            provider.fetch_vix_from_fred.side_effect = RuntimeError("fred down")

            real_provider = YahooFinanceQuoteProvider()
            provider.read_pe_from_seed.side_effect = real_provider.read_pe_from_seed
            provider.read_vix_from_seed.side_effect = real_provider.read_vix_from_seed

            with patch.dict(os.environ, {"INDEXTRACK_META_SEED_FILE": "/tmp/nonexistent_seed_file.json"}):
                meta = _fetch_ui_market_meta(
                    quote_provider=provider,
                    repository=repo,
                    logger=logging.getLogger("test.ui.meta.seed"),
                )

            self.assertIsNotNone(meta.pe_by_symbol["SP500"])
            self.assertIsNotNone(meta.pe_by_symbol["NASDAQ"])
            self.assertIsNotNone(meta.vix)
            self.assertIn("bundled_default", meta.pe_source_by_symbol["SP500"])
            self.assertIn("bundled_default", meta.vix_source)


if __name__ == "__main__":
    unittest.main()
