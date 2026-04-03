"""Map quantile predictions to constrained multi-class probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np

SQRT2 = sqrt(2.0)
QUANTILE_GAP_TO_STD = 2.5632


@dataclass(frozen=True)
class ProbabilityMapperConfig:
    """Probability transform and constraint config."""

    sigma_floor_5: float = 0.0030
    sigma_floor_20: float = 0.0060
    sigma_floor_60: float = 0.0100
    lambda_5: float = 0.80
    lambda_20: float = 1.00
    lambda_60: float = 1.00
    calibration_blend_5: float = 1.0
    calibration_blend_20: float = 0.0
    calibration_blend_60: float = 0.0
    max_calibration_shift_5: float = 0.55
    max_calibration_shift_20: float = 0.0
    max_calibration_shift_60: float = 0.0
    base_prob_recent_weight_5: float = 0.0
    base_prob_recent_weight_20: float = 0.0
    base_prob_recent_weight_60: float = 0.0
    base_prob_recent_window_5: int = 90
    base_prob_recent_window_20: int = 140
    base_prob_recent_window_60: int = 220
    prob_cap: float = 0.72
    display_prob_cap: float = 0.90
    display_uncertain_threshold: float = 0.34
    display_high_conf_threshold: float = 0.62
    display_min_confidence: float = 0.22
    display_regime_shift_threshold: float = 0.28
    display_cross_horizon_extreme_threshold: float = 0.85
    display_label_margin_threshold: float = 0.06
    display_label_uncertain_confidence: float = 0.05
    display_label_mild_confidence: float = 0.15
    display_label_strong_top1_threshold: float = 0.60
    display_label_strong_margin_threshold: float = 0.20
    eps: float = 1e-8

    def sigma_floor(self, horizon_days: int) -> float:
        if horizon_days <= 5:
            return self.sigma_floor_5
        if horizon_days <= 20:
            return self.sigma_floor_20
        return self.sigma_floor_60

    def lambda_h(self, horizon_days: int) -> float:
        if horizon_days <= 5:
            return self.lambda_5
        if horizon_days <= 20:
            return self.lambda_20
        return self.lambda_60

    def calibration_blend(self, horizon_days: int) -> float:
        if horizon_days <= 5:
            return float(np.clip(self.calibration_blend_5, 0.0, 1.0))
        if horizon_days <= 20:
            return float(np.clip(self.calibration_blend_20, 0.0, 1.0))
        return float(np.clip(self.calibration_blend_60, 0.0, 1.0))

    def max_calibration_shift(self, horizon_days: int) -> float:
        if horizon_days <= 5:
            return float(np.clip(self.max_calibration_shift_5, 0.0, 1.0))
        if horizon_days <= 20:
            return float(np.clip(self.max_calibration_shift_20, 0.0, 1.0))
        return float(np.clip(self.max_calibration_shift_60, 0.0, 1.0))

    def base_prob_recent_weight(self, horizon_days: int) -> float:
        if horizon_days <= 5:
            value = self.base_prob_recent_weight_5
        elif horizon_days <= 20:
            value = self.base_prob_recent_weight_20
        else:
            value = self.base_prob_recent_weight_60
        return float(np.clip(value, 0.0, 1.0))

    def base_prob_recent_window(self, horizon_days: int) -> int:
        if horizon_days <= 5:
            value = self.base_prob_recent_window_5
        elif horizon_days <= 20:
            value = self.base_prob_recent_window_20
        else:
            value = self.base_prob_recent_window_60
        return max(20, int(value))


def quantiles_to_raw_probabilities(
    *,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    threshold_up: np.ndarray,
    threshold_down: np.ndarray,
    sigma_floor: float,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert quantile predictions into raw down/flat/up probabilities."""
    q10_adj = np.minimum(q10, q50)
    q90_adj = np.maximum(q90, q50)
    spread = (q90_adj - q10_adj) / QUANTILE_GAP_TO_STD
    sigma = np.maximum(spread, sigma_floor)
    sigma = np.maximum(sigma, eps)
    mu = q50

    z_down = (threshold_down - mu) / sigma
    z_up = (threshold_up - mu) / sigma
    p_down = _normal_cdf(z_down)
    p_up = 1.0 - _normal_cdf(z_up)
    p_flat = 1.0 - p_down - p_up
    p_flat = np.maximum(p_flat, 0.0)
    probs = np.column_stack([p_down, p_flat, p_up])
    probs = _normalize_rows(probs, eps=eps)
    return probs[:, 0], probs[:, 1], probs[:, 2], mu, sigma


def calibrate_and_constrain(
    *,
    raw_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    base_probs: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    lambda_h: float,
    regimes: np.ndarray,
    config: ProbabilityMapperConfig,
) -> np.ndarray:
    """Apply shrinkage and cap constraints on calibrated probabilities."""
    blend = lambda_h * calibrated_probs + (1.0 - lambda_h) * base_probs
    signal_strength = np.abs(mu) / np.maximum(sigma, config.eps)
    extra_shrink = np.clip((0.55 - signal_strength) * 0.5, 0.0, 0.35)
    blend = (1.0 - extra_shrink[:, None]) * blend + extra_shrink[:, None] * base_probs

    out = blend.copy()
    for idx in range(out.shape[0]):
        allow_cap_break = str(regimes[idx]) == "high_vol_extreme"
        if not allow_cap_break:
            out[idx] = _cap_single_row(out[idx], cap=config.prob_cap, base_probs=base_probs)
    out = _normalize_rows(np.clip(out, 0.0, 1.0), eps=config.eps)
    return out


def blend_calibrated_probabilities(
    *,
    raw_probs: np.ndarray,
    model_calibrated_probs: np.ndarray,
    blend_weight: float,
    max_shift: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """Blend calibrator output with raw probabilities and limit abrupt flips."""
    blend = float(np.clip(blend_weight, 0.0, 1.0))
    shift_cap = float(np.clip(max_shift, 0.0, 1.0))
    merged = blend * model_calibrated_probs + (1.0 - blend) * raw_probs
    delta = np.clip(merged - raw_probs, -shift_cap, shift_cap)
    adjusted = raw_probs + delta
    adjusted = np.clip(adjusted, 0.0, 1.0)
    return _normalize_rows(adjusted, eps=eps)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    z_arr = np.asarray(z, dtype=float)
    if z_arr.ndim == 0:
        return np.asarray(0.5 * (1.0 + erf(float(z_arr) / SQRT2)), dtype=float)
    flat = z_arr.reshape(-1)
    values = np.fromiter(
        (0.5 * (1.0 + erf(float(item) / SQRT2)) for item in flat),
        dtype=float,
        count=flat.shape[0],
    )
    return values.reshape(z_arr.shape)


def _normalize_rows(probs: np.ndarray, eps: float) -> np.ndarray:
    row_sum = probs.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum <= eps, 1.0, row_sum)
    out = probs / row_sum
    out = np.clip(out, 0.0, 1.0)
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum <= eps, 1.0, row_sum)
    return out / row_sum


def _cap_single_row(row: np.ndarray, cap: float, base_probs: np.ndarray) -> np.ndarray:
    out = row.copy()
    for _ in range(6):
        max_value = float(np.max(out))
        if max_value <= cap + 1e-12:
            break
        max_idx = int(np.argmax(out))
        overflow = out[max_idx] - cap
        out[max_idx] = cap
        targets = [idx for idx in range(out.shape[0]) if idx != max_idx]
        weights = base_probs[targets]
        if float(np.sum(weights)) <= 1e-12:
            weights = np.asarray([1.0] * len(targets), dtype=float)
        weights = weights / float(np.sum(weights))
        for local_idx, target in enumerate(targets):
            out[target] += overflow * weights[local_idx]
    out = np.clip(out, 0.0, cap)
    s = float(np.sum(out))
    if s <= 1e-12:
        return np.asarray([1 / 3, 1 / 3, 1 / 3], dtype=float)
    return out / s
