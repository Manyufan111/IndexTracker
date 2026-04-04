"""Label construction for multi-horizon probability modeling."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Mapping

import numpy as np

from indextrack.app.models import Horizon
from indextrack.domain.probability.feature_engineering import FeatureMatrix, annualized_threshold_scale

LOGGER = logging.getLogger("indextrack.probability.label_builder")

HORIZON_DAYS: dict[Horizon, int] = {
    "short": 5,
    "mid": 20,
    "long": 60,
}

HORIZON_NAMES = {5: "short", 20: "mid", 60: "long"}

LABEL_DOWN = 0
LABEL_FLAT = 1
LABEL_UP = 2


@dataclass(frozen=True)
class HorizonLabelBundle:
    """Label and threshold arrays for one horizon."""

    horizon: Horizon
    horizon_days: int
    future_return: np.ndarray
    threshold_up: np.ndarray
    threshold_down: np.ndarray
    labels: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class LabelConfig:
    """Threshold multiplier config."""

    k_5: float = 0.35
    k_20: float = 0.50
    k_60: float = 0.60
    regime_adjustment_enabled: bool = True
    vol_high_scale_5: float = 0.92
    vol_high_scale_20: float = 0.90
    vol_high_scale_60: float = 0.88
    vol_low_scale_5: float = 1.03
    vol_low_scale_20: float = 1.02
    vol_low_scale_60: float = 1.01
    trend_strong_scale_5: float = 0.97
    trend_strong_scale_20: float = 0.94
    trend_strong_scale_60: float = 0.92
    drawdown_deep_scale_5: float = 0.97
    drawdown_deep_scale_20: float = 0.94
    drawdown_deep_scale_60: float = 0.92
    long_high_vol_strong_trend_scale: float = 1.00

    def as_map(self) -> dict[int, float]:
        return {5: self.k_5, 20: self.k_20, 60: self.k_60}

    def vol_high_scale(self, horizon_days: int) -> float:
        if horizon_days <= 5:
            value = self.vol_high_scale_5
        elif horizon_days <= 20:
            value = self.vol_high_scale_20
        else:
            value = self.vol_high_scale_60
        return float(np.clip(value, 0.70, 1.10))

    def vol_low_scale(self, horizon_days: int) -> float:
        if horizon_days <= 5:
            value = self.vol_low_scale_5
        elif horizon_days <= 20:
            value = self.vol_low_scale_20
        else:
            value = self.vol_low_scale_60
        return float(np.clip(value, 0.90, 1.20))

    def trend_strong_scale(self, horizon_days: int) -> float:
        if horizon_days <= 5:
            value = self.trend_strong_scale_5
        elif horizon_days <= 20:
            value = self.trend_strong_scale_20
        else:
            value = self.trend_strong_scale_60
        return float(np.clip(value, 0.70, 1.10))

    def drawdown_deep_scale(self, horizon_days: int) -> float:
        if horizon_days <= 5:
            value = self.drawdown_deep_scale_5
        elif horizon_days <= 20:
            value = self.drawdown_deep_scale_20
        else:
            value = self.drawdown_deep_scale_60
        return float(np.clip(value, 0.70, 1.10))


class LabelBuilder:
    """Build up/flat/down labels for each horizon."""

    def __init__(self, config: LabelConfig | None = None) -> None:
        self._config = config or LabelConfig()

    def build(self, feature_matrix: FeatureMatrix) -> dict[Horizon, HorizonLabelBundle]:
        close = feature_matrix.close
        hist_vol_20 = feature_matrix.hist_vol_20
        k_map = self._config.as_map()

        outputs: dict[Horizon, HorizonLabelBundle] = {}
        for horizon_days, horizon_name in HORIZON_NAMES.items():
            k_h = k_map[horizon_days]
            future_return = _future_return(close, horizon_days)
            threshold_component = annualized_threshold_scale(hist_vol_20, horizon_days)
            regime_factor = _regime_threshold_factor(
                feature_matrix=feature_matrix,
                config=self._config,
                horizon_days=horizon_days,
            )
            threshold_up = threshold_component * k_h * regime_factor
            threshold_down = -threshold_up

            labels = np.full(close.shape[0], -1, dtype=int)
            valid_mask = (
                np.isfinite(future_return)
                & np.isfinite(threshold_up)
                & np.isfinite(threshold_down)
            )
            labels[valid_mask & (future_return > threshold_up)] = LABEL_UP
            labels[valid_mask & (future_return < threshold_down)] = LABEL_DOWN
            labels[valid_mask & (labels < 0)] = LABEL_FLAT

            outputs[horizon_name] = HorizonLabelBundle(
                horizon=horizon_name,
                horizon_days=horizon_days,
                future_return=future_return,
                threshold_up=threshold_up,
                threshold_down=threshold_down,
                labels=labels,
                valid_mask=valid_mask,
            )
            LOGGER.debug(
                "labels_built horizon=%s rows=%s valid=%s up=%s flat=%s down=%s",
                horizon_name,
                close.shape[0],
                int(np.sum(valid_mask)),
                int(np.sum(labels == LABEL_UP)),
                int(np.sum(labels == LABEL_FLAT)),
                int(np.sum(labels == LABEL_DOWN)),
            )
        return outputs


def _regime_threshold_factor(
    *,
    feature_matrix: FeatureMatrix,
    config: LabelConfig,
    horizon_days: int,
) -> np.ndarray:
    if not config.regime_adjustment_enabled:
        return np.ones(feature_matrix.row_count, dtype=float)

    names = feature_matrix.feature_names
    values = feature_matrix.values

    def _col(name: str) -> np.ndarray:
        if name not in names:
            return np.full(feature_matrix.row_count, np.nan, dtype=float)
        idx = names.index(name)
        return values[:, idx]

    ma_spread = np.abs(_col("ma_spread_5_20"))
    macd_hist = np.abs(_col("macd_hist"))
    drawdown = np.abs(_col("drawdown_20"))
    vol60 = _col("vol_60")
    vol20 = feature_matrix.hist_vol_20

    vol_ratio = np.full(feature_matrix.row_count, np.nan, dtype=float)
    valid_vol = np.isfinite(vol20) & np.isfinite(vol60) & (np.abs(vol60) > 1e-12)
    vol_ratio[valid_vol] = vol20[valid_vol] / vol60[valid_vol]

    trend_strength = np.full(feature_matrix.row_count, np.nan, dtype=float)
    valid_trend = np.isfinite(ma_spread) & np.isfinite(macd_hist) & np.isfinite(vol20)
    trend_strength[valid_trend] = (
        ma_spread[valid_trend] + 0.5 * macd_hist[valid_trend]
    ) / np.maximum(vol20[valid_trend], 1e-6)

    factor = np.ones(feature_matrix.row_count, dtype=float)
    high_vol = np.isfinite(vol_ratio) & (vol_ratio >= 1.10)
    low_vol = np.isfinite(vol_ratio) & (vol_ratio <= 0.90)
    strong_trend = np.isfinite(trend_strength) & (trend_strength >= 0.55)
    deep_drawdown = np.isfinite(drawdown) & (drawdown >= 0.08)

    factor[high_vol] *= config.vol_high_scale(horizon_days)
    factor[low_vol] *= config.vol_low_scale(horizon_days)
    factor[strong_trend] *= config.trend_strong_scale(horizon_days)
    factor[deep_drawdown] *= config.drawdown_deep_scale(horizon_days)

    # Optional extra narrowing: for long horizon, high volatility with strong trend
    # can be directional but gets over-labeled as uncertain when thresholds are wide.
    # Default keeps behaviour unchanged; set <1.0 via config to narrow only this regime.
    if horizon_days > 20:
        hv_strong = high_vol & strong_trend
        if np.any(hv_strong):
            factor[hv_strong] *= float(np.clip(config.long_high_vol_strong_trend_scale, 0.7, 1.0))
    return np.clip(factor, 0.72, 1.18)


def _future_return(close: np.ndarray, horizon_days: int) -> np.ndarray:
    out = np.full(close.shape[0], np.nan, dtype=float)
    if close.shape[0] <= horizon_days:
        return out
    numerator = close[horizon_days:]
    denominator = close[:-horizon_days]
    valid = np.abs(denominator) > 1e-12
    values = np.full(denominator.shape[0], np.nan, dtype=float)
    values[valid] = numerator[valid] / denominator[valid] - 1.0
    out[:-horizon_days] = values
    return out


def base_probabilities(labels: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Compute [down, flat, up] base frequencies from training labels."""
    valid_labels = labels[valid_mask]
    if valid_labels.shape[0] == 0:
        return np.asarray([1 / 3, 1 / 3, 1 / 3], dtype=float)
    counts = np.bincount(valid_labels, minlength=3).astype(float)
    total = counts.sum()
    if total <= 0:
        return np.asarray([1 / 3, 1 / 3, 1 / 3], dtype=float)
    return counts / total


def regime_from_vol(
    hist_vol_20: np.ndarray,
    valid_mask: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, Mapping[str, float]]:
    """Assign low/mid/high regime by volatility tertiles."""
    out = np.full(values.shape[0], "mid_vol", dtype=object)
    train_vol = hist_vol_20[valid_mask & np.isfinite(hist_vol_20)]
    if train_vol.shape[0] == 0:
        return out, {"q33": 0.0, "q66": 0.0, "q90": 0.0}

    q33 = float(np.quantile(train_vol, 0.33))
    q66 = float(np.quantile(train_vol, 0.66))
    q90 = float(np.quantile(train_vol, 0.90))
    out[values <= q33] = "low_vol"
    out[(values > q33) & (values <= q66)] = "mid_vol"
    out[values > q66] = "high_vol"
    return out, {"q33": q33, "q66": q66, "q90": q90}
