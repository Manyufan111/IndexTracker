"""Map quantile predictions to constrained multi-class probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from typing import Mapping, Sequence

import numpy as np

SQRT2 = sqrt(2.0)
QUANTILE_GAP_TO_STD = 2.5632


@dataclass(frozen=True)
class ProbabilityMapperConfig:
    """Probability transform and constraint config."""

    sigma_floor_5: float = 0.0030
    sigma_floor_20: float = 0.0075
    sigma_floor_60: float = 0.0100
    lambda_5: float = 0.80
    lambda_20: float = 1.00
    lambda_60: float = 1.00
    short_guard_enabled: bool = False
    short_guard_margin: float = 0.15
    short_guard_uncertain_max: float = 0.35
    short_guard_mu_sigma_min: float = 0.80
    short_guard_alpha: float = 0.70
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
    display_regime_shift_uncertain_boost_5: float = 0.15
    display_regime_shift_uncertain_boost_20: float = 0.12
    display_regime_shift_uncertain_boost_60: float = 0.08
    display_high_vol_uncertain_boost_5: float = 0.05
    display_high_vol_uncertain_boost_20: float = 0.04
    display_high_vol_uncertain_boost_60: float = 0.03
    display_max_total_uncertain_boost_5: float = 0.12
    display_max_total_uncertain_boost_20: float = 0.09
    display_max_total_uncertain_boost_60: float = 0.06
    display_uncertain_ceiling_5: float = 0.95
    display_uncertain_ceiling_20: float = 0.95
    display_uncertain_ceiling_60: float = 0.75
    display_raw_uncertain_soft_threshold: float = 0.65
    display_raw_uncertain_hard_threshold: float = 0.75
    display_label_margin_threshold: float = 0.06
    display_label_uncertain_confidence: float = 0.05
    display_label_mild_confidence: float = 0.15
    display_label_uncertain_lean_gap: float = 0.12
    display_label_uncertain_lean_max_uncertain: float = 0.70
    display_label_strong_top1_threshold: float = 0.60
    display_label_strong_margin_threshold: float = 0.20
    display_binary_blend_5: float = 1.00
    display_binary_blend_20: float = 0.35
    display_binary_blend_60: float = 0.45
    display_binary_max_shift_5: float = 0.45
    display_binary_max_shift_20: float = 0.22
    display_binary_max_shift_60: float = 0.22
    display_binary_overcorrection_gap: float = 0.55
    display_binary_overcorrection_shrink: float = 0.70
    display_binary_flip_rate_threshold_20: float = 0.40
    display_binary_flip_rate_threshold_60: float = 0.45
    raw_temperature_5: float = 1.00
    raw_temperature_20: float = 1.35
    raw_temperature_60: float = 1.05
    raw_uncertainty_floor_5: float = 0.00
    raw_uncertainty_floor_20: float = 0.08
    raw_uncertainty_floor_60: float = 0.02
    raw_class_cap_5: float = 1.00
    raw_class_cap_20: float = 0.82
    raw_class_cap_60: float = 0.90
    raw_extreme_trigger_5: float = 1.00
    raw_extreme_trigger_20: float = 0.85
    raw_extreme_trigger_60: float = 0.92
    raw_extreme_uncertainty_boost_5: float = 0.00
    raw_extreme_uncertainty_boost_20: float = 0.14
    raw_extreme_uncertainty_boost_60: float = 0.06
    raw_prob_mapping_mode: str = "piecewise"
    model_quantile_levels: tuple[float, ...] = (
        0.05,
        0.10,
        0.15,
        0.25,
        0.50,
        0.75,
        0.85,
        0.90,
        0.95,
    )
    dynamic_lambda_enabled: bool = True
    dynamic_lambda_strength_5: float = 0.18
    dynamic_lambda_strength_20: float = 0.24
    dynamic_lambda_strength_60: float = 0.28
    dynamic_lambda_min_5: float = 0.60
    dynamic_lambda_min_20: float = 0.68
    dynamic_lambda_min_60: float = 0.72
    dynamic_lambda_regime_shift_penalty: float = 0.20
    display_use_binary_calibrator: bool = False
    display_uncertain_step_cap_5: float = 0.02
    display_uncertain_step_cap_20: float = 0.015
    display_uncertain_step_cap_60: float = 0.012
    display_guardrail_margin_freeze: float = 0.18
    display_allow_top1_reorder: bool = False
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

    def display_binary_blend(self, horizon_days: int) -> float:
        if horizon_days <= 5:
            value = self.display_binary_blend_5
        elif horizon_days <= 20:
            value = self.display_binary_blend_20
        else:
            value = self.display_binary_blend_60
        return float(np.clip(value, 0.0, 1.0))

    def display_binary_max_shift(self, horizon_days: int) -> float:
        if horizon_days <= 5:
            value = self.display_binary_max_shift_5
        elif horizon_days <= 20:
            value = self.display_binary_max_shift_20
        else:
            value = self.display_binary_max_shift_60
        return float(np.clip(value, 0.0, 1.0))

    def display_binary_flip_rate_threshold(self, horizon_days: int) -> float:
        if horizon_days <= 5:
            return 1.0
        if horizon_days <= 20:
            value = self.display_binary_flip_rate_threshold_20
        else:
            value = self.display_binary_flip_rate_threshold_60
        return float(np.clip(value, 0.0, 1.0))

    def display_regime_shift_uncertain_boost(self, horizon_days: int) -> float:
        if horizon_days <= 5:
            value = self.display_regime_shift_uncertain_boost_5
        elif horizon_days <= 20:
            value = self.display_regime_shift_uncertain_boost_20
        else:
            value = self.display_regime_shift_uncertain_boost_60
        return float(np.clip(value, 0.0, 0.25))

    def display_high_vol_uncertain_boost(self, horizon_days: int) -> float:
        if horizon_days <= 5:
            value = self.display_high_vol_uncertain_boost_5
        elif horizon_days <= 20:
            value = self.display_high_vol_uncertain_boost_20
        else:
            value = self.display_high_vol_uncertain_boost_60
        return float(np.clip(value, 0.0, 0.20))

    def display_max_total_uncertain_boost(self, horizon_days: int) -> float:
        if horizon_days <= 5:
            value = self.display_max_total_uncertain_boost_5
        elif horizon_days <= 20:
            value = self.display_max_total_uncertain_boost_20
        else:
            value = self.display_max_total_uncertain_boost_60
        return float(np.clip(value, 0.0, 0.30))

    def display_uncertain_ceiling(self, horizon_days: int) -> float:
        if horizon_days <= 5:
            value = self.display_uncertain_ceiling_5
        elif horizon_days <= 20:
            value = self.display_uncertain_ceiling_20
        else:
            value = self.display_uncertain_ceiling_60
        return float(np.clip(value, 0.0, 0.99))

    def raw_temperature(self, horizon_days: int) -> float:
        if horizon_days <= 5:
            value = self.raw_temperature_5
        elif horizon_days <= 20:
            value = self.raw_temperature_20
        else:
            value = self.raw_temperature_60
        return max(float(value), 1.0)

    def raw_uncertainty_floor(self, horizon_days: int) -> float:
        if horizon_days <= 5:
            value = self.raw_uncertainty_floor_5
        elif horizon_days <= 20:
            value = self.raw_uncertainty_floor_20
        else:
            value = self.raw_uncertainty_floor_60
        return float(np.clip(value, 0.0, 1.0))

    def raw_class_cap(self, horizon_days: int) -> float:
        if horizon_days <= 5:
            value = self.raw_class_cap_5
        elif horizon_days <= 20:
            value = self.raw_class_cap_20
        else:
            value = self.raw_class_cap_60
        return float(np.clip(value, 0.34, 1.0))

    def raw_extreme_trigger(self, horizon_days: int) -> float:
        if horizon_days <= 5:
            value = self.raw_extreme_trigger_5
        elif horizon_days <= 20:
            value = self.raw_extreme_trigger_20
        else:
            value = self.raw_extreme_trigger_60
        return float(np.clip(value, 0.34, 1.0))

    def raw_extreme_uncertainty_boost(self, horizon_days: int) -> float:
        if horizon_days <= 5:
            value = self.raw_extreme_uncertainty_boost_5
        elif horizon_days <= 20:
            value = self.raw_extreme_uncertainty_boost_20
        else:
            value = self.raw_extreme_uncertainty_boost_60
        return float(np.clip(value, 0.0, 0.5))

    def normalized_quantile_levels(self) -> tuple[float, ...]:
        cleaned = sorted(
            {
                float(np.clip(level, 0.01, 0.99))
                for level in self.model_quantile_levels
                if np.isfinite(level)
            }
        )
        mandatory = [0.10, 0.50, 0.90]
        for item in mandatory:
            if all(abs(item - value) > 1e-8 for value in cleaned):
                cleaned.append(item)
        cleaned = sorted(cleaned)
        return tuple(cleaned)

    def normalized_raw_prob_mapping_mode(self) -> str:
        mode = self.raw_prob_mapping_mode.strip().lower()
        if mode in {"piecewise", "gaussian"}:
            return mode
        return "piecewise"

    def dynamic_lambda_strength(self, horizon_days: int) -> float:
        if horizon_days <= 5:
            value = self.dynamic_lambda_strength_5
        elif horizon_days <= 20:
            value = self.dynamic_lambda_strength_20
        else:
            value = self.dynamic_lambda_strength_60
        return float(np.clip(value, 0.0, 0.6))

    def dynamic_lambda_min(self, horizon_days: int) -> float:
        if horizon_days <= 5:
            value = self.dynamic_lambda_min_5
        elif horizon_days <= 20:
            value = self.dynamic_lambda_min_20
        else:
            value = self.dynamic_lambda_min_60
        return float(np.clip(value, 0.0, 1.0))

    def display_uncertain_step_cap(self, horizon_days: int) -> float:
        if horizon_days <= 5:
            value = self.display_uncertain_step_cap_5
        elif horizon_days <= 20:
            value = self.display_uncertain_step_cap_20
        else:
            value = self.display_uncertain_step_cap_60
        return float(np.clip(value, 0.0, 0.10))


def quantiles_to_raw_probabilities(
    *,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    threshold_up: np.ndarray,
    threshold_down: np.ndarray,
    sigma_floor: float,
    quantile_predictions: Mapping[float, np.ndarray] | None = None,
    quantile_levels: Sequence[float] | None = None,
    mapping_mode: str = "gaussian",
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert quantile predictions into raw down/flat/up probabilities."""
    q10_adj = np.minimum(q10, q50)
    q90_adj = np.maximum(q90, q50)
    spread = (q90_adj - q10_adj) / QUANTILE_GAP_TO_STD
    sigma = np.maximum(spread, sigma_floor)
    sigma = np.maximum(sigma, eps)
    mu = q50

    mode = mapping_mode.strip().lower()
    use_piecewise = (
        mode == "piecewise"
        and quantile_predictions is not None
        and len(quantile_predictions) >= 5
    )
    if use_piecewise:
        p_down, p_up = _piecewise_tail_probabilities(
            threshold_down=threshold_down,
            threshold_up=threshold_up,
            quantile_predictions=quantile_predictions,
            quantile_levels=quantile_levels,
            eps=eps,
        )
    else:
        z_down = (threshold_down - mu) / sigma
        z_up = (threshold_up - mu) / sigma
        p_down = _normal_cdf(z_down)
        p_up = 1.0 - _normal_cdf(z_up)
    p_flat = 1.0 - p_down - p_up
    p_flat = np.maximum(p_flat, 0.0)
    probs = np.column_stack([p_down, p_flat, p_up])
    probs = _normalize_rows(probs, eps=eps)
    return probs[:, 0], probs[:, 1], probs[:, 2], mu, sigma


def refine_raw_probabilities(
    *,
    probs: np.ndarray,
    horizon_days: int,
    config: ProbabilityMapperConfig,
) -> np.ndarray:
    """Apply raw-level anti-extreme refinement before calibration."""
    out = _normalize_rows(np.clip(probs, 0.0, 1.0), eps=config.eps)
    temp = config.raw_temperature(horizon_days)
    if temp > 1.000001:
        out = _temperature_smooth(out, temperature=temp, eps=config.eps)

    floor = config.raw_uncertainty_floor(horizon_days)
    if floor > config.eps:
        out = _enforce_uncertainty_floor(out, floor=floor, eps=config.eps)

    trigger = config.raw_extreme_trigger(horizon_days)
    boost = config.raw_extreme_uncertainty_boost(horizon_days)
    if boost > config.eps and trigger < 0.999999:
        out = _boost_uncertainty_for_extremes(
            out,
            trigger=trigger,
            max_boost=boost,
            eps=config.eps,
        )

    class_cap = config.raw_class_cap(horizon_days)
    if class_cap < 0.999999:
        base = np.asarray([0.30, 0.40, 0.30], dtype=float)
        for idx in range(out.shape[0]):
            out[idx] = _cap_single_row(out[idx], cap=class_cap, base_probs=base)
    return _normalize_rows(np.clip(out, 0.0, 1.0), eps=config.eps)


def calibrate_and_constrain(
    *,
    raw_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    base_probs: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    lambda_h: float,
    horizon_days: int,
    regime_shift_score: float,
    regimes: np.ndarray,
    config: ProbabilityMapperConfig,
) -> np.ndarray:
    """Apply shrinkage and cap constraints on calibrated probabilities."""
    base = np.asarray(base_probs, dtype=float).reshape(1, 3)
    base = _normalize_rows(np.clip(base, 0.0, 1.0), eps=config.eps)
    base = np.repeat(base, calibrated_probs.shape[0], axis=0)

    if config.dynamic_lambda_enabled:
        sorted_probs = np.sort(calibrated_probs, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        uncertainty = np.clip(calibrated_probs[:, 1], 0.0, 1.0)
        signal_norm = np.abs(mu) / (np.abs(mu) + np.maximum(sigma, config.eps))
        quality = np.clip(
            0.50 * margin + 0.30 * (1.0 - uncertainty) + 0.20 * signal_norm,
            0.0,
            1.0,
        )
        strength = config.dynamic_lambda_strength(horizon_days)
        lambda_eff = lambda_h * ((1.0 - strength) + strength * quality)
        regime_penalty = float(
            np.clip(regime_shift_score, 0.0, 1.0) * np.clip(config.dynamic_lambda_regime_shift_penalty, 0.0, 0.8)
        )
        lambda_eff = lambda_eff * (1.0 - regime_penalty)
        lambda_eff = np.clip(lambda_eff, config.dynamic_lambda_min(horizon_days), 1.0)
        blend = lambda_eff[:, None] * calibrated_probs + (1.0 - lambda_eff[:, None]) * base
    else:
        blend = lambda_h * calibrated_probs + (1.0 - lambda_h) * base
        signal_strength = np.abs(mu) / np.maximum(sigma, config.eps)
        extra_shrink = np.clip((0.55 - signal_strength) * 0.5, 0.0, 0.35)
        blend = (1.0 - extra_shrink[:, None]) * blend + extra_shrink[:, None] * base

    out = blend.copy()
    for idx in range(out.shape[0]):
        allow_cap_break = str(regimes[idx]) == "high_vol_extreme"
        if not allow_cap_break:
            out[idx] = _cap_single_row(out[idx], cap=config.prob_cap, base_probs=base[idx])
    out = _normalize_rows(np.clip(out, 0.0, 1.0), eps=config.eps)
    return out


def _piecewise_tail_probabilities(
    *,
    threshold_down: np.ndarray,
    threshold_up: np.ndarray,
    quantile_predictions: Mapping[float, np.ndarray],
    quantile_levels: Sequence[float] | None,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    quantile_pairs = _normalize_quantile_pairs(
        quantile_predictions=quantile_predictions,
        quantile_levels=quantile_levels,
    )
    levels = np.asarray([item[0] for item in quantile_pairs], dtype=float)
    values = np.column_stack([item[1] for item in quantile_pairs])
    values = _enforce_quantile_monotonic(values, eps=eps)
    p_down = np.zeros(values.shape[0], dtype=float)
    p_up = np.zeros(values.shape[0], dtype=float)
    for idx in range(values.shape[0]):
        cdf_down = _piecewise_cdf_single(
            x=float(threshold_down[idx]),
            q_levels=levels,
            q_values=values[idx],
            eps=eps,
        )
        cdf_up = _piecewise_cdf_single(
            x=float(threshold_up[idx]),
            q_levels=levels,
            q_values=values[idx],
            eps=eps,
        )
        p_down[idx] = float(np.clip(cdf_down, 0.0, 1.0))
        p_up[idx] = float(np.clip(1.0 - cdf_up, 0.0, 1.0))
    return p_down, p_up


def _normalize_quantile_pairs(
    *,
    quantile_predictions: Mapping[float, np.ndarray],
    quantile_levels: Sequence[float] | None,
) -> list[tuple[float, np.ndarray]]:
    target_levels: list[float] = []
    if quantile_levels is not None:
        target_levels = [float(value) for value in quantile_levels if np.isfinite(value)]
    if not target_levels:
        target_levels = [float(value) for value in quantile_predictions.keys()]
    target_levels = sorted({float(np.clip(value, 0.01, 0.99)) for value in target_levels})

    level_to_values: dict[float, np.ndarray] = {}
    for raw_level, raw_values in quantile_predictions.items():
        level = float(np.clip(raw_level, 0.01, 0.99))
        arr = np.asarray(raw_values, dtype=float).reshape(-1)
        level_to_values[level] = arr

    pairs: list[tuple[float, np.ndarray]] = []
    for level in target_levels:
        if level in level_to_values:
            pairs.append((level, level_to_values[level]))
    if len(pairs) < 5:
        pairs = sorted(level_to_values.items(), key=lambda item: item[0])
    if not pairs:
        raise ValueError("quantile_predictions 为空，无法构建分段分布")

    row_count = pairs[0][1].shape[0]
    checked: list[tuple[float, np.ndarray]] = []
    for level, arr in pairs:
        if arr.shape[0] != row_count:
            raise ValueError("quantile_predictions 行数不一致")
        checked.append((level, arr))
    return checked


def _enforce_quantile_monotonic(values: np.ndarray, *, eps: float) -> np.ndarray:
    out = values.copy()
    out = np.maximum.accumulate(out, axis=1)
    for col in range(1, out.shape[1]):
        too_close = out[:, col] <= out[:, col - 1]
        if np.any(too_close):
            out[too_close, col] = out[too_close, col - 1] + eps
    return out


def _piecewise_cdf_single(
    *,
    x: float,
    q_levels: np.ndarray,
    q_values: np.ndarray,
    eps: float,
) -> float:
    if q_values.shape[0] < 2:
        return float(np.clip(q_levels[0], 0.0, 1.0))
    if x <= q_values[0]:
        slope = (q_levels[1] - q_levels[0]) / max(q_values[1] - q_values[0], eps)
        value = q_levels[0] + (x - q_values[0]) * slope
        return float(np.clip(value, 0.0, 1.0))
    if x >= q_values[-1]:
        slope = (q_levels[-1] - q_levels[-2]) / max(q_values[-1] - q_values[-2], eps)
        value = q_levels[-1] + (x - q_values[-1]) * slope
        return float(np.clip(value, 0.0, 1.0))
    value = float(np.interp(x, q_values, q_levels))
    return float(np.clip(value, 0.0, 1.0))


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


def _temperature_smooth(probs: np.ndarray, *, temperature: float, eps: float) -> np.ndarray:
    temp = max(float(temperature), 1.0)
    logits = np.log(np.clip(probs, eps, 1.0)) / temp
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    denom = np.sum(exp_values, axis=1, keepdims=True)
    denom = np.where(denom <= eps, 1.0, denom)
    return exp_values / denom


def _enforce_uncertainty_floor(probs: np.ndarray, *, floor: float, eps: float) -> np.ndarray:
    out = probs.copy()
    floor_value = float(np.clip(floor, 0.0, 1.0))
    for idx in range(out.shape[0]):
        current = float(out[idx, 1])
        if current >= floor_value:
            continue
        need = floor_value - current
        direction_mass = max(float(out[idx, 0] + out[idx, 2]), eps)
        shrink = min(need, direction_mass)
        if shrink <= eps:
            continue
        scale = max((direction_mass - shrink) / direction_mass, 0.0)
        out[idx, 0] *= scale
        out[idx, 2] *= scale
        out[idx, 1] = current + shrink
    return _normalize_rows(out, eps=eps)


def _boost_uncertainty_for_extremes(
    probs: np.ndarray,
    *,
    trigger: float,
    max_boost: float,
    eps: float,
) -> np.ndarray:
    out = probs.copy()
    for idx in range(out.shape[0]):
        top1 = float(np.max(out[idx]))
        if top1 <= trigger:
            continue
        denom = max(1.0 - trigger, eps)
        ratio = min((top1 - trigger) / denom, 1.0)
        add = max_boost * ratio
        direction_mass = max(float(out[idx, 0] + out[idx, 2]), eps)
        add = min(add, direction_mass)
        if add <= eps:
            continue
        scale = max((direction_mass - add) / direction_mass, 0.0)
        out[idx, 0] *= scale
        out[idx, 2] *= scale
        out[idx, 1] += add
    return _normalize_rows(out, eps=eps)


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
