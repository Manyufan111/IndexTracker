"""Integration tests for end-to-end CLI flow."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from indextrack.app.models import Candle
from indextrack.cli import main


def _build_mock_candles(symbol: str, end: date, days: int = 300) -> list[Candle]:
    rows: list[Candle] = []
    start = end - timedelta(days=days - 1)
    price = 100.0
    for idx in range(days):
        trade_day = start + timedelta(days=idx)
        price += 0.1
        rows.append(
            Candle(
                symbol=symbol,
                trade_date=trade_day,
                open=price - 0.4,
                high=price + 0.7,
                low=price - 0.8,
                close=price,
                volume=1000 + idx,
                source="mock_primary",
                fetched_at=datetime(2026, 3, 31, 20, 0, 0),
            )
        )
    return rows


class CliIntegrationTests(unittest.TestCase):
    @patch("indextrack.cli.YahooFinanceSecondaryProvider")
    @patch("indextrack.cli.YahooFinancePrimaryProvider")
    def test_cli_main_end_to_end_with_mock_provider(
        self,
        mock_primary_cls,
        mock_secondary_cls,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "indextrack.db"
            log_path = Path(temp_dir) / "indextrack.log"

            def fetch_daily(symbol: str, start: date, end: date) -> list[Candle]:
                return _build_mock_candles(symbol=symbol, end=end, days=300)

            mock_primary = SimpleNamespace(name="mock_primary", fetch_daily=fetch_daily)
            mock_secondary = SimpleNamespace(
                name="mock_secondary",
                fetch_daily=lambda symbol, start, end: _build_mock_candles(
                    symbol=symbol, end=end, days=300
                ),
            )
            mock_primary_cls.return_value = mock_primary
            mock_secondary_cls.return_value = mock_secondary

            exit_code = main(
                [
                    "--index",
                    "SP500",
                    "--period",
                    "1M",
                    "--model",
                    "legacy",
                    "--db-path",
                    str(db_path),
                    "--log-path",
                    str(log_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(db_path.exists())
            self.assertTrue(log_path.exists())


if __name__ == "__main__":
    unittest.main()
