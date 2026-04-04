"""Tests for Yahoo quote metadata provider."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from indextrack.infra.providers.quote import YahooFinanceQuoteProvider


class QuoteProviderTests(unittest.TestCase):
    def test_fetch_dashboard_meta_reads_pe_and_vix_from_quote(self) -> None:
        provider = YahooFinanceQuoteProvider()
        payload = {
            "quoteResponse": {
                "result": [
                    {"symbol": "^GSPC", "trailingPE": 24.11},
                    {"symbol": "^IXIC", "trailingPE": 33.02},
                    {"symbol": "^VIX", "regularMarketPrice": 19.77},
                ]
            }
        }
        with patch.object(YahooFinanceQuoteProvider, "_request_quote", return_value=payload):
            with patch.object(YahooFinanceQuoteProvider, "_fetch_pe_from_summary", return_value=None):
                meta = provider.fetch_dashboard_meta(["SP500", "NASDAQ"])

        self.assertEqual(meta.pe_by_symbol["SP500"], 24.11)
        self.assertEqual(meta.pe_by_symbol["NASDAQ"], 33.02)
        self.assertEqual(meta.vix, 19.77)
        self.assertEqual(meta.source, "yahoo_finance_primary_quote")
        self.assertIn("index_quote", meta.pe_source_by_symbol["SP500"])
        self.assertIn("index_quote", meta.pe_source_by_symbol["NASDAQ"])
        self.assertIn("quote(^VIX)", meta.vix_source)

    def test_fetch_dashboard_meta_uses_summary_fallback_for_missing_pe(self) -> None:
        provider = YahooFinanceQuoteProvider()
        payload = {
            "quoteResponse": {
                "result": [
                    {"symbol": "^GSPC"},
                    {"symbol": "^VIX", "regularMarketPrice": 25.5},
                ]
            }
        }
        with patch.object(YahooFinanceQuoteProvider, "_request_quote", return_value=payload):
            with patch.object(YahooFinanceQuoteProvider, "_fetch_pe_from_summary", return_value=22.6):
                meta = provider.fetch_dashboard_meta(["SP500"])

        self.assertEqual(meta.pe_by_symbol["SP500"], 22.6)
        self.assertEqual(meta.vix, 25.5)
        self.assertIsNone(meta.note)
        self.assertIn("index_summary", meta.pe_source_by_symbol["SP500"])

    def test_fetch_dashboard_meta_uses_proxy_quote_when_index_pe_missing(self) -> None:
        provider = YahooFinanceQuoteProvider()
        payload = {
            "quoteResponse": {
                "result": [
                    {"symbol": "^GSPC"},
                    {"symbol": "SPY", "trailingPE": 26.8},
                    {"symbol": "^VIX", "regularMarketPrice": 18.6},
                ]
            }
        }
        with patch.object(YahooFinanceQuoteProvider, "_request_quote", return_value=payload):
            with patch.object(YahooFinanceQuoteProvider, "_fetch_pe_from_summary", return_value=None):
                meta = provider.fetch_dashboard_meta(["SP500"])

        self.assertEqual(meta.pe_by_symbol["SP500"], 26.8)
        self.assertIn("proxy_quote(SPY)", meta.pe_source_by_symbol["SP500"])

    def test_fetch_vix_from_fred_parses_last_valid_value(self) -> None:
        provider = YahooFinanceQuoteProvider()
        fred_csv = "DATE,VIXCLS\n2026-04-01,20.10\n2026-04-02,.\n2026-04-03,22.34\n"
        with patch.object(YahooFinanceQuoteProvider, "_request_text", return_value=fred_csv):
            value = provider.fetch_vix_from_fred()
        self.assertEqual(value, 22.34)

    def test_alpha_vantage_pe_disabled_without_api_key(self) -> None:
        provider = YahooFinanceQuoteProvider()
        with patch.dict(os.environ, {}, clear=True):
            value, source = provider.fetch_pe_from_alpha_vantage("SP500")
        self.assertIsNone(value)
        self.assertIn("disabled", source)

    def test_read_meta_seed_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            seed = Path(temp_dir) / "seed.json"
            seed.write_text(
                '{"pe":{"SP500":25.6,"NASDAQ":32.4},"vix":19.1}',
                encoding="utf-8",
            )
            provider = YahooFinanceQuoteProvider()
            with patch.dict(os.environ, {"INDEXTRACK_META_SEED_FILE": str(seed)}):
                pe, pe_source = provider.read_pe_from_seed("SP500")
                vix, vix_source = provider.read_vix_from_seed()
        self.assertEqual(pe, 25.6)
        self.assertEqual(vix, 19.1)
        self.assertIn("seed:file(pe:", pe_source)
        self.assertIn("seed:file(vix:", vix_source)

    def test_read_meta_seed_falls_back_to_bundled_defaults(self) -> None:
        provider = YahooFinanceQuoteProvider()
        with patch.dict(os.environ, {"INDEXTRACK_META_SEED_FILE": "/tmp/not_exists_seed.json"}):
            pe, pe_source = provider.read_pe_from_seed("SP500")
            vix, vix_source = provider.read_vix_from_seed()
        self.assertIsNotNone(pe)
        self.assertIsNotNone(vix)
        self.assertIn("bundled_default", pe_source)
        self.assertIn("bundled_default", vix_source)


if __name__ == "__main__":
    unittest.main()
