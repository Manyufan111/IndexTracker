"""Secondary market data provider based on Yahoo mirror endpoint."""

from __future__ import annotations

from dataclasses import dataclass

from indextrack.infra.providers.primary import YahooFinancePrimaryProvider

YAHOO_SECONDARY_BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"


@dataclass(frozen=True)
class YahooFinanceSecondaryProvider(YahooFinancePrimaryProvider):
    """Fallback provider using a second Yahoo host, without API key dependency."""

    base_url: str = YAHOO_SECONDARY_BASE_URL

    @property
    def name(self) -> str:
        return "yahoo_finance_secondary"

