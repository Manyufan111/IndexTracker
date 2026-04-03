"""Tests for CLI-friendly error handling."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from indextrack.cli import main


class CliErrorHandlingTests(unittest.TestCase):
    def test_invalid_period_returns_friendly_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "indextrack.db"
            log_path = Path(temp_dir) / "indextrack.log"
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(
                    [
                        "--index",
                        "SP500",
                        "--period",
                        "2Q",
                        "--db-path",
                        str(db_path),
                        "--log-path",
                        str(log_path),
                    ]
                )

        output = buffer.getvalue()
        self.assertEqual(exit_code, 1)
        self.assertIn("错误: 不支持的 period", output)
        self.assertIn("建议:", output)
        self.assertIn("1M/3M/6M/1Y/3Y/5Y", output)

    def test_invalid_timezone_returns_friendly_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "indextrack.db"
            log_path = Path(temp_dir) / "indextrack.log"
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(
                    [
                        "--index",
                        "SP500",
                        "--period",
                        "1M",
                        "--timezone",
                        "Mars/Olympus",
                        "--db-path",
                        str(db_path),
                        "--log-path",
                        str(log_path),
                    ]
                )

        output = buffer.getvalue()
        self.assertEqual(exit_code, 1)
        self.assertIn("错误: 不支持的时区", output)
        self.assertIn("建议:", output)
        self.assertIn("America/New_York", output)


if __name__ == "__main__":
    unittest.main()
