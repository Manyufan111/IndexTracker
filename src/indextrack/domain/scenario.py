"""Scenario-related sub-scorers and utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

from indextrack.app.models import Horizon, ScenarioProbs
from indextrack.domain.trend import TrendScoreResult

from indextrack.domain.features import FeatureSet


@dataclass(frozen=True)
class HorizonScores:
    """Generic short/mid/long score container in [-1, 1]."""

    short: float
    mid: float
    long: float


@dataclass(frozen=True)
class HorizonWeightConfig:
    """Logit weights for one horizon in ScenarioEngine."""

    trend: float
    momentum: float
    volatility: float
    neutral_bias: float
    side_volatility_bias: float


class MomentumScorer:
    """Map momentum features into normalized horizon scores."""

    def score(self, features: FeatureSet) -> HorizonScores:
        return HorizonScores(
            short=_normalize(features.momentum_5d_pct, scale=3.0),
            mid=_normalize(features.momentum_20d_pct, scale=6.0),
            long=_normalize(features.momentum_60d_pct, scale=12.0),
        )


class VolatilityScorer:
    """Assess volatility pressure: higher volatility => lower stability score."""

    def score(self, features: FeatureSet) -> HorizonScores:
        short = _stability_from_volatility(
            annualized_vol=features.volatility_20d_ann,
            atr_pct=features.atr_14_pct,
            target_vol=0.18,
            target_atr=2.2,
        )
        mid = _stability_from_volatility(
            annualized_vol=features.volatility_20d_ann,
            atr_pct=features.atr_14_pct,
            target_vol=0.20,
            target_atr=2.8,
        )
        long = _stability_from_volatility(
            annualized_vol=features.volatility_60d_ann,
            atr_pct=features.atr_14_pct,
            target_vol=0.22,
            target_atr=3.2,
        )
        return HorizonScores(short=short, mid=mid, long=long)


class ScenarioEngine:
    """Convert horizon scores into up/sideways/down probabilities."""

    def __init__(
        self,
        *,
        weights: dict[Horizon, HorizonWeightConfig] | None = None,
    ) -> None:
        default_weights = {
            "short": HorizonWeightConfig(
                trend=1.4,
                momentum=1.1,
                volatility=0.5,
                neutral_bias=1.1,
                side_volatility_bias=0.3,
            ),
            "mid": HorizonWeightConfig(
                trend=1.5,
                momentum=1.0,
                volatility=0.45,
                neutral_bias=1.0,
                side_volatility_bias=0.28,
            ),
            "long": HorizonWeightConfig(
                trend=1.6,
                momentum=0.9,
                volatility=0.4,
                neutral_bias=0.9,
                side_volatility_bias=0.25,
            ),
        }
        self._weights = default_weights
        if weights:
            self._weights.update(weights)

    def generate(
        self,
        *,
        trend_scores: TrendScoreResult,
        momentum_scores: HorizonScores,
        volatility_scores: HorizonScores,
    ) -> list[ScenarioProbs]:
        short = self._to_probs(
            horizon="short",
            trend=trend_scores.short_score,
            momentum=momentum_scores.short,
            volatility=volatility_scores.short,
        )
        mid = self._to_probs(
            horizon="mid",
            trend=trend_scores.mid_score,
            momentum=momentum_scores.mid,
            volatility=volatility_scores.mid,
        )
        long = self._to_probs(
            horizon="long",
            trend=trend_scores.long_score,
            momentum=momentum_scores.long,
            volatility=volatility_scores.long,
        )
        return [short, mid, long]

    def _to_probs(
        self,
        *,
        horizon: Horizon,
        trend: float,
        momentum: float,
        volatility: float,
    ) -> ScenarioProbs:
        cfg = self._weights[horizon]
        up_logit = cfg.trend * trend + cfg.momentum * momentum + cfg.volatility * volatility
        down_logit = -cfg.trend * trend - cfg.momentum * momentum - cfg.volatility * volatility

        neutral_strength = 1 - min(1.0, (abs(trend) + abs(momentum)) / 2)
        side_logit = cfg.neutral_bias * neutral_strength + cfg.side_volatility_bias * max(
            volatility, -0.5
        )

        up_prob, side_prob, down_prob = _softmax_to_pct(up_logit, side_logit, down_logit)
        return ScenarioProbs(
            horizon=horizon,
            uptrend_pct=up_prob,
            sideways_pct=side_prob,
            downtrend_pct=down_prob,
        )


def build_horizon_weights(
    *,
    short: tuple[float, float, float, float, float],
    mid: tuple[float, float, float, float, float],
    long: tuple[float, float, float, float, float],
) -> dict[Horizon, HorizonWeightConfig]:
    """Build ScenarioEngine weights from tuple-based configuration."""
    return {
        "short": HorizonWeightConfig(*short),
        "mid": HorizonWeightConfig(*mid),
        "long": HorizonWeightConfig(*long),
    }


def _stability_from_volatility(
    *,
    annualized_vol: float | None,
    atr_pct: float | None,
    target_vol: float,
    target_atr: float,
) -> float:
    vol_component = _inverse_normalize(annualized_vol, target=target_vol)
    atr_component = _inverse_normalize(atr_pct, target=target_atr)
    return _clamp((vol_component + atr_component) / 2)


def _normalize(value: float | None, *, scale: float) -> float:
    if value is None or scale <= 0:
        return 0.0
    return _clamp(value / scale)


def _inverse_normalize(value: float | None, *, target: float) -> float:
    if value is None or target <= 0:
        return 0.0
    return _clamp((target - value) / target)


def _clamp(value: float, *, low: float = -1.0, high: float = 1.0) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _softmax_to_pct(logit_a: float, logit_b: float, logit_c: float) -> tuple[float, float, float]:
    raw_a = exp(logit_a)
    raw_b = exp(logit_b)
    raw_c = exp(logit_c)
    total = raw_a + raw_b + raw_c
    if total == 0:
        return 33.34, 33.33, 33.33
    up = raw_a / total * 100
    side = raw_b / total * 100
    down = 100 - up - side
    return up, side, down
