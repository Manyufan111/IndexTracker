"""Trend scoring based on engineered features."""

from __future__ import annotations

from dataclasses import dataclass

from indextrack.app.models import TrendSignals
from indextrack.domain.features import FeatureSet


@dataclass(frozen=True)
class TrendScoreResult:
    """Directional trend output with raw horizon scores."""

    signals: TrendSignals
    short_score: float
    mid_score: float
    long_score: float


class TrendScorer:
    """Convert feature values into short/mid/long trend directions."""

    def score(self, features: FeatureSet) -> TrendScoreResult:
        short_score = _avg(
            _signed_bucket(features.price_vs_sma_20_pct, threshold=1.0),
            _signed_bucket(features.momentum_5d_pct, threshold=0.8),
            _signed_bucket(features.momentum_20d_pct, threshold=1.2),
        )
        mid_score = _avg(
            _signed_bucket(features.price_vs_sma_50_pct, threshold=1.2),
            _signed_bucket(features.momentum_20d_pct, threshold=1.0),
            _signed_bucket(features.momentum_60d_pct, threshold=2.0),
        )
        long_score = _avg(
            _signed_bucket(features.price_vs_sma_200_pct, threshold=1.5),
            _signed_bucket(features.momentum_60d_pct, threshold=2.5),
        )

        signals = TrendSignals(
            short=_to_direction(short_score),
            mid=_to_direction(mid_score),
            long=_to_direction(long_score),
        )
        return TrendScoreResult(
            signals=signals,
            short_score=short_score,
            mid_score=mid_score,
            long_score=long_score,
        )


def _to_direction(score: float) -> str:
    if score >= 0.34:
        return "uptrend"
    if score <= -0.34:
        return "downtrend"
    return "sideways"


def _signed_bucket(value: float | None, threshold: float) -> float:
    if value is None:
        return 0.0
    if value >= threshold:
        return 1.0
    if value <= -threshold:
        return -1.0
    return 0.0


def _avg(*values: float) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
