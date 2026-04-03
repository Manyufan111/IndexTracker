"""Logging setup utilities."""

from __future__ import annotations

import logging
from pathlib import Path

DEFAULT_LOG_PATH = ".data/indextrack.log"


def setup_logging(*, level: str = "INFO", log_path: str = DEFAULT_LOG_PATH) -> None:
    """Configure root logger with console + file handlers."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric_level)

    if root.handlers:
        # Close old handlers before replacing them to avoid leaked file descriptors.
        for handler in list(root.handlers):
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # noqa: BLE001
                continue

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(numeric_level)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
