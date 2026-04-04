"""Tests for UI lookback sizing logic."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from indextrack.cli import _ui_lookback_calendar_days


class UiLookbackTests(unittest.TestCase):
    def test_quantile_mode_uses_minimum_training_window(self) -> None:
        lookback = _ui_lookback_calendar_days(
            period="1M",
            probability_model_config=None,
        )
        self.assertGreaterEqual(lookback, 476)

    def test_quantile_mode_uses_larger_training_window(self) -> None:
        lookback = _ui_lookback_calendar_days(
            period="1M",
            probability_model_config=SimpleNamespace(split=SimpleNamespace(min_train_size=120, min_valid_size=20)),
        )
        self.assertGreaterEqual(lookback, 476)

    def test_quantile_mode_respects_split_config(self) -> None:
        cfg = SimpleNamespace(split=SimpleNamespace(min_train_size=150, min_valid_size=25))
        lookback = _ui_lookback_calendar_days(
            period="1M",
            probability_model_config=cfg,
        )
        # 150 + 2*25 + 120 = 320 trading days -> 544 calendar days.
        self.assertGreaterEqual(lookback, 544)


if __name__ == "__main__":
    unittest.main()
