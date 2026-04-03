"""Tests for CLI model fallback when quantile dependencies are unavailable."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from indextrack.app.models import Candle
from indextrack.cli import main
from indextrack.domain.features import FeatureError


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


class CliModelFallbackTests(unittest.TestCase):
    @patch("indextrack.cli._build_probability_model_config")
    @patch("indextrack.cli.YahooFinanceSecondaryProvider")
    @patch("indextrack.cli.YahooFinancePrimaryProvider")
    def test_quantile_dependency_missing_auto_falls_back_to_legacy(
        self,
        mock_primary_cls,
        mock_secondary_cls,
        mock_build_probability_config,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "indextrack.db"
            log_path = Path(temp_dir) / "indextrack.log"

            def fetch_daily(symbol: str, start: date, end: date) -> list[Candle]:
                return _build_mock_candles(symbol=symbol, end=end, days=300)

            mock_primary = SimpleNamespace(name="mock_primary", fetch_daily=fetch_daily)
            mock_secondary = SimpleNamespace(name="mock_secondary", fetch_daily=fetch_daily)
            mock_primary_cls.return_value = mock_primary
            mock_secondary_cls.return_value = mock_secondary
            mock_build_probability_config.side_effect = FeatureError(
                "概率模型依赖缺失: No module named 'numpy'"
            )

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(
                    [
                        "--index",
                        "SP500",
                        "--period",
                        "1M",
                        "--model",
                        "quantile",
                        "--db-path",
                        str(db_path),
                        "--log-path",
                        str(log_path),
                    ]
                )

        output = buffer.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("已自动切换到 --model legacy", output)
        self.assertIn("短期:", output)


if __name__ == "__main__":
    unittest.main()
