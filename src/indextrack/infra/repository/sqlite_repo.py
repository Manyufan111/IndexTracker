"""SQLite-backed repository for prices and fetch snapshots."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

from indextrack.app.models import Candle, DataSnapshotMeta

DEFAULT_DB_PATH = ".data/indextrack.db"


@dataclass(frozen=True)
class RepositoryStats:
    """Simple stats for repository writes."""

    rows_written: int
    last_trade_date: date


class SQLiteRepository:
    """Persist normalized candles and fetch snapshots in SQLite."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def save_prices(self, symbol: str, candles: Iterable[Candle], source: str) -> RepositoryStats:
        normalized_symbol = symbol.strip().upper()
        rows = list(candles)
        if not rows:
            raise ValueError("save_prices 需要至少一条 candle 数据")
        if not normalized_symbol:
            raise ValueError("symbol 不能为空")

        sorted_rows = sorted(rows, key=lambda item: item.trade_date)
        last_trade_date = sorted_rows[-1].trade_date
        fetched_at = sorted_rows[-1].fetched_at

        with closing(self._connect()) as conn:
            with conn:
                conn.executemany(
                    """
                    INSERT INTO prices (
                        symbol, trade_date, open, high, low, close, volume, source, fetched_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(symbol, trade_date) DO UPDATE SET
                        open=excluded.open,
                        high=excluded.high,
                        low=excluded.low,
                        close=excluded.close,
                        volume=excluded.volume,
                        source=excluded.source,
                        fetched_at=excluded.fetched_at
                    """,
                    [
                        (
                            normalized_symbol,
                            row.trade_date.isoformat(),
                            row.open,
                            row.high,
                            row.low,
                            row.close,
                            row.volume,
                            source,
                            row.fetched_at.isoformat(),
                        )
                        for row in sorted_rows
                    ],
                )
                conn.execute(
                    """
                    INSERT INTO snapshots (symbol, source, fetched_at, last_trade_date, row_count)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        normalized_symbol,
                        source,
                        fetched_at.isoformat(),
                        last_trade_date.isoformat(),
                        len(sorted_rows),
                    ),
                )

        return RepositoryStats(rows_written=len(sorted_rows), last_trade_date=last_trade_date)

    def load_prices(self, symbol: str, lookback_days: int) -> list[Candle]:
        if lookback_days <= 0:
            raise ValueError("lookback_days 必须为正整数")
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            raise ValueError("symbol 不能为空")

        latest = self.last_success(normalized_symbol)
        if latest is None:
            return []
        start_date = latest.last_trade_date - timedelta(days=lookback_days - 1)

        with closing(self._connect()) as conn:
            cursor = conn.execute(
                """
                SELECT
                    symbol, trade_date, open, high, low, close, volume, source, fetched_at
                FROM prices
                WHERE symbol = ? AND trade_date >= ?
                ORDER BY trade_date ASC
                """,
                (normalized_symbol, start_date.isoformat()),
            )
            rows = cursor.fetchall()

        return [self._row_to_candle(row) for row in rows]

    def last_success(self, symbol: str) -> DataSnapshotMeta | None:
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            raise ValueError("symbol 不能为空")

        with closing(self._connect()) as conn:
            cursor = conn.execute(
                """
                SELECT symbol, source, last_trade_date, fetched_at, row_count
                FROM snapshots
                WHERE symbol = ?
                ORDER BY fetched_at DESC
                LIMIT 1
                """,
                (normalized_symbol,),
            )
            row = cursor.fetchone()

        if row is None:
            return None
        return DataSnapshotMeta(
            symbol=row["symbol"],
            source=row["source"],
            last_trade_date=date.fromisoformat(row["last_trade_date"]),
            fetched_at=datetime.fromisoformat(row["fetched_at"]),
            row_count=int(row["row_count"]),
        )

    def _initialize(self) -> None:
        with closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS prices (
                        symbol TEXT NOT NULL,
                        trade_date TEXT NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume REAL NULL,
                        source TEXT NOT NULL,
                        fetched_at TEXT NOT NULL,
                        PRIMARY KEY(symbol, trade_date)
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        source TEXT NOT NULL,
                        fetched_at TEXT NOT NULL,
                        last_trade_date TEXT NOT NULL,
                        row_count INTEGER NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_snapshots_symbol_fetched_at
                    ON snapshots(symbol, fetched_at DESC)
                    """
                )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _row_to_candle(row: sqlite3.Row) -> Candle:
        volume = row["volume"]
        return Candle(
            symbol=row["symbol"],
            trade_date=date.fromisoformat(row["trade_date"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(volume) if volume is not None else None,
            source=row["source"],
            fetched_at=datetime.fromisoformat(row["fetched_at"]),
        )
