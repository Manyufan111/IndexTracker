#!/usr/bin/env python3
"""Cloud-friendly UI launcher for IndexTrack."""

from __future__ import annotations

import os

from indextrack.cli import main


def _read_env(name: str, default: str) -> str:
    value = os.getenv(name, default)
    cleaned = value.strip()
    return cleaned if cleaned else default


def _build_args() -> list[str]:
    host = _read_env("HOST", "0.0.0.0")
    port = _read_env("PORT", "8000")
    period = _read_env("INDEXTRACK_DEPLOY_PERIOD", "1Y")
    model = _read_env("INDEXTRACK_DEPLOY_MODEL", "quantile")
    db_path = _read_env("INDEXTRACK_DB_PATH", ".data/indextrack.db")
    log_level = _read_env("INDEXTRACK_LOG_LEVEL", "INFO").upper()

    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    os.makedirs(".data", exist_ok=True)

    return [
        "--ui",
        "--ui-host",
        host,
        "--ui-port",
        port,
        "--period",
        period,
        "--model",
        model,
        "--db-path",
        db_path,
        "--log-level",
        log_level,
    ]


if __name__ == "__main__":
    raise SystemExit(main(_build_args()))
