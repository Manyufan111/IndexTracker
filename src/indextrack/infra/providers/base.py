"""Base abstractions for market data providers."""

from __future__ import annotations

from datetime import date
from typing import Protocol

from indextrack.app.models import Candle


class ProviderError(RuntimeError):
    """Base exception for provider failures."""


class UnsupportedSymbolError(ProviderError):
    """Raised when a provider cannot serve the requested symbol."""


class ProviderResponseError(ProviderError):
    """Raised when an upstream provider response cannot be parsed."""


class MarketDataProvider(Protocol):
    """Protocol that every market data provider implementation must satisfy."""

    name: str

    def fetch_daily(self, symbol: str, start: date, end: date) -> list[Candle]:
        """Fetch normalized daily candles for a symbol in [start, end]."""
        ...
