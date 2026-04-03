"""Tests for logging setup lifecycle."""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from indextrack.infra.logging import setup_logging


class LoggingSetupTests(unittest.TestCase):
    def tearDown(self) -> None:
        root = logging.getLogger()
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()

    def test_reconfig_closes_previous_file_handler(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first_log = Path(temp_dir) / "first.log"
            second_log = Path(temp_dir) / "second.log"

            setup_logging(log_path=str(first_log))
            old_handlers = list(logging.getLogger().handlers)
            old_file_handler = next(
                handler for handler in old_handlers if isinstance(handler, logging.FileHandler)
            )

            setup_logging(log_path=str(second_log))
            new_handlers = list(logging.getLogger().handlers)
            new_file_handler = next(
                handler for handler in new_handlers if isinstance(handler, logging.FileHandler)
            )

            self.assertIsNone(old_file_handler.stream)
            self.assertEqual(len(new_handlers), 2)
            self.assertEqual(Path(new_file_handler.baseFilename), second_log)


if __name__ == "__main__":
    unittest.main()
