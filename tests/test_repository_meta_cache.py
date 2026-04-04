"""Tests for repository market meta cache helpers."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from indextrack.infra.repository.sqlite_repo import SQLiteRepository


class RepositoryMetaCacheTests(unittest.TestCase):
    def test_save_and_load_market_meta(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "indextrack.db"
            repo = SQLiteRepository(str(db_path))
            now = datetime(2026, 4, 3, 18, 0, 0, tzinfo=timezone.utc)

            repo.save_market_meta(
                key="PE_SP500",
                value=24.33,
                source="yahoo_primary:index_quote",
                fetched_at=now,
                note="ui_live",
            )
            entry = repo.load_market_meta(key="PE_SP500")
            self.assertIsNotNone(entry)
            assert entry is not None
            self.assertEqual(entry.value, 24.33)
            self.assertEqual(entry.source, "yahoo_primary:index_quote")
            self.assertEqual(entry.note, "ui_live")


if __name__ == "__main__":
    unittest.main()
