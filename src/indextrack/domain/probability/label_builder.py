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

    def as_map(self) -> dict[int, float]:
        return {5: self.k_5, 20: self.k_20, 60: self.k_60}


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
            threshold_up = threshold_component * k_h
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
