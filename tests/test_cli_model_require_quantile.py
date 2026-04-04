"""Tests for quantile-only CLI behavior."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from indextrack.cli import main
from indextrack.domain.features import FeatureError


class CliModelRequireQuantileTests(unittest.TestCase):
    @patch("indextrack.cli._build_probability_model_config")
    def test_quantile_dependency_missing_returns_error(self, mock_build_probability_config) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "indextrack.db"
            log_path = Path(temp_dir) / "indextrack.log"
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
        self.assertEqual(exit_code, 1)
        self.assertIn("错误: 概率模型依赖缺失", output)
        self.assertNotIn("自动切换", output)


if __name__ == "__main__":
    unittest.main()
