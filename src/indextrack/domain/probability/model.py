"""End-to-end market probability model with OOF calibration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import logging
from math import sqrt
from typing import Any, Mapping, Sequence

import numpy as np

from indextrack.app.models import Candle, Horizon, TrendSignals
from indextrack.domain.probability.calibrator import (
    BinaryLogitCalibrator,
    CalibratorConfig,
    SoftmaxCalibrator,
)
from indextrack.domain.probability.evaluator import (
    BinaryEvaluationMetrics,
    EvaluationMetrics,
    binary_metrics_to_dict,
    evaluate_binary_probabilities,
    evaluate_probabilities,
    metrics_to_dict,
)
from indextrack.domain.probability.feature_engineering import FeatureMatrix, ProbabilityFeatureEngine
from indextrack.domain.probability.label_builder import (
    HORIZON_DAYS,
    LABEL_DOWN,
    LABEL_FLAT,
    LABEL_UP,
    LabelBuilder,
    LabelConfig,
    base_probabilities,
    regime_from_vol,
)
from indextrack.domain.probability.probability_mapper import (
    ProbabilityMapperConfig,
    blend_calibrated_probabilities,
    calibrate_and_constrain,
    quantiles_to_raw_probabilities,
    refine_raw_probabilities,
)
from indextrack.domain.probability.quantile_model import (
    QuantileLinearRegressor,
    QuantileModelConfig,
    TimeSeriesSplitConfig,
    expanding_time_series_splits,
)

LOGGER = logging.getLogger("indextrack.probability.model")


class ProbabilityModelError(RuntimeError):
    """Raised when probability model cannot provide valid output."""


@dataclass(frozen=True)
class HorizonProbabilityOutput:
    """Probability output for one horizon at one timestamp."""

    horizon: Horizon
    horizon_days: int
    prob_down_raw: float
    prob_flat_raw: float
    prob_up_raw: float
    prob_down_calibrated: float
    prob_flat_calibrated: float
    prob_up_calibrated: float
    prob_down: float
    prob_flat: float
    prob_up: float
    threshold_up: float
    threshold_down: float
    regime: str
    mu: float
    sigma: float
    confidence: float
    display_prob_up_raw: float = 0.0
    display_prob_down_raw: float = 0.0
    display_prob_uncertain_raw: float = 0.0
    display_prob_up_calibrated: float = 0.0
    display_prob_down_calibrated: float = 0.0
    display_prob_uncertain_calibrated: float = 0.0
    display_prob_up_post_shrink: float = 0.0
    display_prob_down_post_shrink: float = 0.0
    display_prob_uncertain_post_shrink: float = 0.0
    display_prob_up_post_uncertainty_boost: float = 0.0
    display_prob_down_post_uncertainty_boost: float = 0.0
    display_prob_uncertain_post_uncertainty_boost: float = 0.0
    display_prob_up: float = 0.0
    display_prob_uncertain: float = 0.0
    display_prob_down: float = 0.0
    raw_uncertain: float = 0.0
    final_uncertain: float = 0.0
    regime_shift_boost_delta: float = 0.0
    high_vol_boost_delta: float = 0.0
    total_uncertain_boost_delta: float = 0.0
    uncertain_ceiling_applied: bool = False
    signal_strength: float = 0.0
    display_confidence: float = 0.0
    display_state: str = "uncertain"
    internal_state: str = "uncertain"
    label: str = "uncertain"
    headline_label: str = "不确定"
    final_state_label: str = "uncertain"
    display_label: str = "uncertain"
    top1_prob: float = 0.0
    top2_prob: float = 0.0
    margin: float = 0.0
    display_warning: str = ""
    warning_level: str = "none"
    warning_code: str = ""
    warning_message: str = ""


@dataclass(frozen=True)
class BacktestOutput:
    """Model backtest/evaluation output."""

    metrics_by_horizon: Mapping[Horizon, EvaluationMetrics]
    diagnostics_by_horizon: Mapping[Horizon, "HorizonDiagnostics"]

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for horizon, metrics in self.metrics_by_horizon.items():
            payload = metrics_to_dict(metrics)
            diagnostics = self.diagnostics_by_horizon.get(horizon)
            if diagnostics is not None:
                payload["diagnostics"] = diagnostics.to_dict()
            out[horizon] = payload
        return out


@dataclass(frozen=True)
class ProbabilityChainMetrics:
    """Stage-wise OOF evaluation metrics (raw -> calibrated -> final)."""

    raw: EvaluationMetrics
    calibrated: EvaluationMetrics
    final: EvaluationMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": metrics_to_dict(self.raw),
            "calibrated": metrics_to_dict(self.calibrated),
            "final": metrics_to_dict(self.final),
        }


@dataclass(frozen=True)
class BinaryDirectionChainMetrics:
    """Stage-wise directional binary metrics (raw -> calibrated -> final)."""

    raw: BinaryEvaluationMetrics
    calibrated: BinaryEvaluationMetrics
    final: BinaryEvaluationMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": binary_metrics_to_dict(self.raw),
            "calibrated": binary_metrics_to_dict(self.calibrated),
            "final": binary_metrics_to_dict(self.final),
        }


@dataclass(frozen=True)
class HorizonDiagnostics:
    """Detailed diagnostics for one horizon."""

    horizon: Horizon
    horizon_days: int
    train_rows: int
    oof_rows: int
    train_label_distribution: Mapping[str, float]
    oof_label_distribution: Mapping[str, float]
    label_distribution: Mapping[str, float]
    base_probs: Mapping[str, float]
    chain_metrics: ProbabilityChainMetrics
    chain_shift: Mapping[str, float]
    calibration_mode: str
    calibration_blend: float
    max_calibration_shift: float
    lambda_h: float
    base_prob_recent_weight: float
    base_prob_recent_window: int
    sigma_floor: float
    k_h: float
    threshold_up_stats: Mapping[str, float]
    threshold_down_stats: Mapping[str, float]
    q10_stats: Mapping[str, float]
    q50_stats: Mapping[str, float]
    q90_stats: Mapping[str, float]
    sigma_stats: Mapping[str, float]
    recent_label_distribution: Mapping[str, float] = field(default_factory=dict)
    train_label_distribution_by_regime: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    oof_label_distribution_by_regime: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    threshold_uncertain_bins: list[Mapping[str, float]] = field(default_factory=list)
    calibration_flip_summary: Mapping[str, float] = field(default_factory=dict)
    binary_chain_metrics: Mapping[str, Any] = field(default_factory=dict)
    binary_pipeline: Mapping[str, Any] = field(default_factory=dict)
    short_guard_stats: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "horizon_days": self.horizon_days,
            "train_rows": self.train_rows,
            "oof_rows": self.oof_rows,
            "train_label_distribution": dict(self.train_label_distribution),
            "oof_label_distribution": dict(self.oof_label_distribution),
            "label_distribution": dict(self.label_distribution),
            "base_probs": dict(self.base_probs),
            "chain_metrics": self.chain_metrics.to_dict(),
            "chain_shift": dict(self.chain_shift),
            "calibration_mode": self.calibration_mode,
            "calibration_blend": self.calibration_blend,
            "max_calibration_shift": self.max_calibration_shift,
            "lambda_h": self.lambda_h,
            "base_prob_recent_weight": self.base_prob_recent_weight,
            "base_prob_recent_window": self.base_prob_recent_window,
            "sigma_floor": self.sigma_floor,
            "k_h": self.k_h,
            "threshold_up_stats": dict(self.threshold_up_stats),
            "threshold_down_stats": dict(self.threshold_down_stats),
            "q10_stats": dict(self.q10_stats),
            "q50_stats": dict(self.q50_stats),
            "q90_stats": dict(self.q90_stats),
            "sigma_stats": dict(self.sigma_stats),
            "recent_label_distribution": dict(self.recent_label_distribution),
            "train_label_distribution_by_regime": {
                key: dict(value) for key, value in self.train_label_distribution_by_regime.items()
            },
            "oof_label_distribution_by_regime": {
                key: dict(value) for key, value in self.oof_label_distribution_by_regime.items()
            },
            "threshold_uncertain_bins": [dict(item) for item in self.threshold_uncertain_bins],
            "calibration_flip_summary": dict(self.calibration_flip_summary),
            "binary_chain_metrics": dict(self.binary_chain_metrics),
            "binary_pipeline": dict(self.binary_pipeline),
            "short_guard_stats": dict(self.short_guard_stats),
        }


@dataclass(frozen=True)
class MarketProbabilityModelConfig:
    """Combined configuration for the probability model pipeline."""

    label: LabelConfig = field(default_factory=LabelConfig)
    quantile: QuantileModelConfig = field(default_factory=QuantileModelConfig)
    calibrator: CalibratorConfig = field(default_factory=CalibratorConfig)
    mapper: ProbabilityMapperConfig = field(default_factory=ProbabilityMapperConfig)
    split: TimeSeriesSplitConfig = field(default_factory=TimeSeriesSplitConfig)


@dataclass
class _HorizonState:
    horizon: Horizon
    horizon_days: int
    k_h: float
    lambda_h: float
    sigma_floor: float
    quantile_models: Mapping[float, QuantileLinearRegressor]
    quantile_levels: tuple[float, ...]
    q10_model: QuantileLinearRegressor
    q50_model: QuantileLinearRegressor
    q90_model: QuantileLinearRegressor
    calibrator: SoftmaxCalibrator | None
    binary_calibrator: BinaryLogitCalibrator | None
    calibration_mode: str
    calibration_blend: float
    max_calibration_shift: float
    base_probs: np.ndarray
    vol_cutoffs: Mapping[str, float]
    metrics: EvaluationMetrics
    binary_mode: str
    binary_cap: float
    regime_shift_score: float
    binary_chain_metrics: Mapping[str, Any]
    diagnostics: HorizonDiagnostics


class MarketProbabilityModel:
    """Probability model with OOF calibration and reliability constraints."""

    def __init__(self, config: MarketProbabilityModelConfig | None = None) -> None:
        self.config = config or MarketProbabilityModelConfig()
        self._feature_engine = ProbabilityFeatureEngine()
        self._label_builder = LabelBuilder(self.config.label)
        self._states: dict[Horizon, _HorizonState] = {}
        self._backtest_output: BacktestOutput | None = None
        self._fit_signature: tuple[int, str] | None = None

    def fit(self, candles: Sequence[Candle]) -> "MarketProbabilityModel":
        feature_matrix = self._feature_engine.transform(candles)
        labels_by_horizon = self._label_builder.build(feature_matrix)
        self._states = {}
        metrics_by_horizon: dict[Horizon, EvaluationMetrics] = {}
        diagnostics_by_horizon: dict[Horizon, HorizonDiagnostics] = {}

        finite_feature_mask = np.isfinite(feature_matrix.values).all(axis=1)
        for horizon, label_bundle in labels_by_horizon.items():
            valid_mask = label_bundle.valid_mask & finite_feature_mask
            indices = np.where(valid_mask)[0]
            if indices.shape[0] < (self.config.split.min_train_size + self.config.split.min_valid_size):
                raise ProbabilityModelError(
                    f"{horizon} 样本不足，无法进行 OOF + 校准建模。有效样本={indices.shape[0]}"
                )

            x = feature_matrix.values[indices]
            y_reg = label_bundle.future_return[indices]
            y_cls = label_bundle.labels[indices]
            threshold_up = label_bundle.threshold_up[indices]
            threshold_down = label_bundle.threshold_down[indices]
            vol_values = feature_matrix.hist_vol_20[indices]
            global_base_probs = base_probabilities(y_cls, np.ones_like(y_cls, dtype=bool))
            base_probs = global_base_probs
            recent_weight = self.config.mapper.base_prob_recent_weight(label_bundle.horizon_days)
            if recent_weight > 1e-8:
                recent_window = self.config.mapper.base_prob_recent_window(label_bundle.horizon_days)
                recent_labels = y_cls[-min(recent_window, y_cls.shape[0]) :]
                recent_base_probs = base_probabilities(
                    recent_labels,
                    np.ones_like(recent_labels, dtype=bool),
                )
                base_probs = (1.0 - recent_weight) * global_base_probs + recent_weight * recent_base_probs
                base_probs = base_probs / max(float(np.sum(base_probs)), self.config.mapper.eps)

            oof_raw = np.full((indices.shape[0], 3), np.nan, dtype=float)
            oof_mu = np.full(indices.shape[0], np.nan, dtype=float)
            oof_sigma = np.full(indices.shape[0], np.nan, dtype=float)
            quantile_levels = self.config.mapper.normalized_quantile_levels()
            q10_level = _nearest_quantile_level(quantile_levels, 0.10)
            q50_level = _nearest_quantile_level(quantile_levels, 0.50)
            q90_level = _nearest_quantile_level(quantile_levels, 0.90)
            oof_quantiles: dict[float, np.ndarray] = {
                level: np.full(indices.shape[0], np.nan, dtype=float)
                for level in quantile_levels
            }
            mapping_mode = self.config.mapper.normalized_raw_prob_mapping_mode()

            split_count = 0
            for train_idx, valid_idx in expanding_time_series_splits(indices.shape[0], self.config.split):
                split_count += 1
                fold_predictions: dict[float, np.ndarray] = {}
                for level in quantile_levels:
                    model = QuantileLinearRegressor(level, self.config.quantile).fit(
                        x[train_idx], y_reg[train_idx]
                    )
                    fold_predictions[level] = model.predict(x[valid_idx])

                q10_pred = fold_predictions[q10_level]
                q50_pred = fold_predictions[q50_level]
                q90_pred = fold_predictions[q90_level]
                down_raw, flat_raw, up_raw, mu, sigma = quantiles_to_raw_probabilities(
                    q10=q10_pred,
                    q50=q50_pred,
                    q90=q90_pred,
                    threshold_up=threshold_up[valid_idx],
                    threshold_down=threshold_down[valid_idx],
                    sigma_floor=self.config.mapper.sigma_floor(label_bundle.horizon_days),
                    quantile_predictions=fold_predictions,
                    quantile_levels=quantile_levels,
                    mapping_mode=mapping_mode,
                    eps=self.config.mapper.eps,
                )
                raw_fold = np.column_stack([down_raw, flat_raw, up_raw])
                raw_fold = refine_raw_probabilities(
                    probs=raw_fold,
                    horizon_days=label_bundle.horizon_days,
                    config=self.config.mapper,
                )
                oof_raw[valid_idx] = raw_fold
                oof_mu[valid_idx] = mu
                oof_sigma[valid_idx] = sigma
                for level, pred_values in fold_predictions.items():
                    oof_quantiles[level][valid_idx] = pred_values

            if split_count == 0:
                raise ProbabilityModelError(f"{horizon} 未生成有效时间序列切分")

            oof_mask = np.isfinite(oof_raw).all(axis=1)
            oof_count = int(np.sum(oof_mask))
            if oof_count < max(12, self.config.split.min_valid_size):
                raise ProbabilityModelError(f"{horizon} OOF 样本不足，无法进行稳定建模。oof={oof_count}")

            base_calibration_mode = self.config.calibrator.normalized_mode_for_horizon_days(
                label_bundle.horizon_days
            )
            calibration_mode = base_calibration_mode
            log_raw_full = np.log(np.clip(oof_raw[oof_mask], self.config.mapper.eps, 1.0))
            calibrator: SoftmaxCalibrator | None = None
            model_calibrated_oof = oof_raw[oof_mask]
            if base_calibration_mode != "none":
                if oof_count < 30:
                    calibration_mode = f"{base_calibration_mode}_auto_disabled_oof_lt_30"
                else:
                    calibration_ratio = self.config.calibrator.recent_oof_ratio(label_bundle.horizon_days)
                    log_raw_fit = log_raw_full
                    y_fit = y_cls[oof_mask]
                    if calibration_ratio < 0.999:
                        keep_rows = max(30, int(round(oof_count * calibration_ratio)))
                        log_raw_fit = log_raw_full[-keep_rows:]
                        y_fit = y_fit[-keep_rows:]
                        calibration_mode = f"{base_calibration_mode}_recent_{calibration_ratio:.2f}"
                    calibrator = SoftmaxCalibrator(self.config.calibrator).fit(log_raw_fit, y_fit)
                    model_calibrated_oof = calibrator.predict_proba(log_raw_full)
                    if base_calibration_mode == "conservative":
                        model_calibrated_oof = _temperature_smooth_probs(
                            model_calibrated_oof,
                            temperature=max(self.config.calibrator.conservative_temperature, 1.0),
                            eps=self.config.mapper.eps,
                        )

            calibration_blend = self.config.mapper.calibration_blend(label_bundle.horizon_days)
            max_calibration_shift = self.config.mapper.max_calibration_shift(label_bundle.horizon_days)
            if calibration_mode.startswith("none") or "auto_disabled" in calibration_mode:
                calibration_blend = 0.0
                max_calibration_shift = 0.0
            calibrated_oof = blend_calibrated_probabilities(
                raw_probs=oof_raw[oof_mask],
                model_calibrated_probs=model_calibrated_oof,
                blend_weight=calibration_blend,
                max_shift=max_calibration_shift,
                eps=self.config.mapper.eps,
            )
            strong_mask_oof: np.ndarray | None = None
            if self.config.mapper.short_guard_enabled and label_bundle.horizon_days <= 5:
                calibrated_oof, strong_mask_oof = _apply_short_guard(
                    raw_probs=oof_raw[oof_mask],
                    calibrated_probs=calibrated_oof,
                    mu=oof_mu[oof_mask],
                    sigma=oof_sigma[oof_mask],
                    horizon_days=label_bundle.horizon_days,
                    config=self.config.mapper,
                    eps=self.config.mapper.eps,
                )
            calibrated_oof = _apply_short_guard(
                raw_probs=oof_raw[oof_mask],
                calibrated_probs=calibrated_oof,
                mu=oof_mu[oof_mask],
                sigma=oof_sigma[oof_mask],
                horizon_days=label_bundle.horizon_days,
                config=self.config.mapper,
                eps=self.config.mapper.eps,
            )

            regime_labels, vol_cutoffs = regime_from_vol(
                feature_matrix.hist_vol_20[indices],
                np.ones(indices.shape[0], dtype=bool),
                vol_values,
            )
            regime_labels = self._extend_extreme_regime(
                regimes=regime_labels,
                vol_values=vol_values,
                mu=oof_mu,
                sigma=oof_sigma,
                q90=vol_cutoffs.get("q90", 0.0),
            )
            train_label_dist = _label_distribution(y_cls)
            oof_label_dist = _label_distribution(y_cls[oof_mask])
            regime_shift_score = (
                abs(train_label_dist["pct_up"] - oof_label_dist["pct_up"])
                + abs(train_label_dist["pct_down"] - oof_label_dist["pct_down"])
            )
            train_core_mask = ~oof_mask
            if int(np.sum(train_core_mask)) == 0:
                train_core_mask = np.ones_like(train_core_mask, dtype=bool)
            recent_window = min(
                self.config.mapper.base_prob_recent_window(label_bundle.horizon_days),
                y_cls.shape[0],
            )
            recent_label_dist = _label_distribution(y_cls[-recent_window:])
            train_by_regime = _label_distribution_by_regime(
                labels=y_cls,
                regimes=regime_labels,
                mask=train_core_mask,
            )
            oof_by_regime = _label_distribution_by_regime(
                labels=y_cls,
                regimes=regime_labels,
                mask=oof_mask,
            )
            threshold_uncertain_bins = _threshold_uncertain_relation(
                labels=y_cls,
                threshold_abs=np.abs(threshold_up),
                mask=np.isfinite(threshold_up),
                bins=5,
            )
            calibration_flip_summary = _calibration_flip_summary(
                raw_probs=oof_raw[oof_mask],
                calibrated_probs=calibrated_oof,
                labels=y_cls[oof_mask],
                eps=self.config.mapper.eps,
            )

            lambda_h = self.config.mapper.lambda_h(label_bundle.horizon_days)
            if _is_raw_only_pipeline(
                calibration_mode=calibration_mode,
                calibration_blend=calibration_blend,
                max_calibration_shift=max_calibration_shift,
                lambda_h=lambda_h,
            ):
                final_oof = oof_raw[oof_mask]
            else:
                if isinstance(calibrated_oof, tuple):
                    calibrated_oof = calibrated_oof[0]
                calibrated_oof = _ensure_prob_matrix(calibrated_oof)
                final_oof = calibrate_and_constrain(
                    raw_probs=oof_raw[oof_mask],
                    calibrated_probs=calibrated_oof,
                    base_probs=base_probs,
                    mu=oof_mu[oof_mask],
                    sigma=oof_sigma[oof_mask],
                    lambda_h=lambda_h,
                    horizon_days=label_bundle.horizon_days,
                    regime_shift_score=float(regime_shift_score),
                    regimes=regime_labels[oof_mask],
                    config=self.config.mapper,
                )

            if isinstance(calibrated_oof, (list, tuple)) and not isinstance(calibrated_oof, np.ndarray):
                try:
                    calibrated_oof = np.asarray(calibrated_oof[0], dtype=float)
                except Exception:
                    calibrated_oof = np.asarray(calibrated_oof, dtype=float)
            raw_metrics = evaluate_probabilities(
                probs=_ensure_prob_matrix(oof_raw[oof_mask]),
                labels=y_cls[oof_mask],
                regimes=regime_labels[oof_mask],
            )
            calibrated_metrics = evaluate_probabilities(
                probs=_ensure_prob_matrix(calibrated_oof),
                labels=y_cls[oof_mask],
                regimes=regime_labels[oof_mask],
            )
            short_guard_stats: dict[str, float] = {}
            if label_bundle.horizon_days <= 5:
                short_guard_stats = _short_guard_metrics(
                    raw_probs=oof_raw[oof_mask],
                    guard_probs=calibrated_oof,
                    labels=y_cls[oof_mask],
                    strong_mask=strong_mask_oof if strong_mask_oof is not None else np.zeros_like(oof_mask, dtype=bool),
                    eps=self.config.mapper.eps,
                )
            metrics = evaluate_probabilities(
                probs=_ensure_prob_matrix(final_oof),
                labels=y_cls[oof_mask],
                regimes=regime_labels[oof_mask],
            )
            chain_metrics = ProbabilityChainMetrics(
                raw=raw_metrics,
                calibrated=calibrated_metrics,
                final=metrics,
            )
            base_probs_dict = _probs_to_dict(base_probs)
            chain_shift = _chain_shift_summary(
                raw_probs=oof_raw[oof_mask],
                calibrated_probs=calibrated_oof,
                final_probs=final_oof,
            )
            (
                binary_calibrator,
                binary_mode,
                binary_chain_metrics,
                binary_pipeline,
            ) = self._fit_binary_display_pipeline(
                horizon=horizon,
                horizon_days=label_bundle.horizon_days,
                final_probs=final_oof,
                labels=y_cls[oof_mask],
            )
            binary_pipeline = {
                **binary_pipeline,
                "regime_shift_score": float(regime_shift_score),
            }
            diagnostics = HorizonDiagnostics(
                horizon=horizon,
                horizon_days=label_bundle.horizon_days,
                train_rows=int(y_cls.shape[0]),
                oof_rows=oof_count,
                train_label_distribution=train_label_dist,
                oof_label_distribution=oof_label_dist,
                label_distribution=oof_label_dist,
                base_probs=base_probs_dict,
                chain_metrics=chain_metrics,
                chain_shift=chain_shift,
                calibration_mode=calibration_mode,
                calibration_blend=calibration_blend,
                max_calibration_shift=max_calibration_shift,
                lambda_h=lambda_h,
                base_prob_recent_weight=self.config.mapper.base_prob_recent_weight(label_bundle.horizon_days),
                base_prob_recent_window=self.config.mapper.base_prob_recent_window(label_bundle.horizon_days),
                sigma_floor=self.config.mapper.sigma_floor(label_bundle.horizon_days),
                k_h=self._k_h(label_bundle.horizon_days),
                threshold_up_stats=_summary_stats(threshold_up),
                threshold_down_stats=_summary_stats(threshold_down),
                q10_stats=_summary_stats(oof_quantiles[q10_level][oof_mask]),
                q50_stats=_summary_stats(oof_quantiles[q50_level][oof_mask]),
                q90_stats=_summary_stats(oof_quantiles[q90_level][oof_mask]),
                sigma_stats=_summary_stats(oof_sigma[oof_mask]),
                recent_label_distribution=recent_label_dist,
                train_label_distribution_by_regime=train_by_regime,
                oof_label_distribution_by_regime=oof_by_regime,
                threshold_uncertain_bins=threshold_uncertain_bins,
                calibration_flip_summary=calibration_flip_summary,
                binary_chain_metrics=binary_chain_metrics.to_dict(),
                binary_pipeline=binary_pipeline,
                short_guard_stats=short_guard_stats,
            )

            full_quantile_models: dict[float, QuantileLinearRegressor] = {}
            for level in quantile_levels:
                full_quantile_models[level] = QuantileLinearRegressor(level, self.config.quantile).fit(x, y_reg)
            full_q10 = full_quantile_models[q10_level]
            full_q50 = full_quantile_models[q50_level]
            full_q90 = full_quantile_models[q90_level]
            self._states[horizon] = _HorizonState(
                horizon=horizon,
                horizon_days=label_bundle.horizon_days,
                k_h=self._k_h(label_bundle.horizon_days),
                lambda_h=lambda_h,
                sigma_floor=self.config.mapper.sigma_floor(label_bundle.horizon_days),
                quantile_models=full_quantile_models,
                quantile_levels=quantile_levels,
                q10_model=full_q10,
                q50_model=full_q50,
                q90_model=full_q90,
                calibrator=calibrator,
                binary_calibrator=binary_calibrator,
                calibration_mode=calibration_mode,
                calibration_blend=calibration_blend,
                max_calibration_shift=max_calibration_shift,
                base_probs=base_probs,
                vol_cutoffs=vol_cutoffs,
                metrics=metrics,
                binary_mode=binary_mode,
                binary_cap=self.config.mapper.display_prob_cap,
                regime_shift_score=float(regime_shift_score),
                binary_chain_metrics=binary_chain_metrics.to_dict(),
                diagnostics=diagnostics,
            )
            metrics_by_horizon[horizon] = metrics
            diagnostics_by_horizon[horizon] = diagnostics
            LOGGER.info(
                (
                "horizon_fit_done horizon=%s rows=%s oof_rows=%s "
                    "train_label(up/flat/down)=%.3f/%.3f/%.3f "
                    "oof_label(up/flat/down)=%.3f/%.3f/%.3f "
                    "raw(logloss=%.6f,brier=%.6f) "
                    "cal(logloss=%.6f,brier=%.6f) "
                    "final(logloss=%.6f,brier=%.6f) "
                    "k=%.3f lambda=%.3f blend=%.3f shift_cap=%.3f sigma_floor=%.4f"
                ),
                horizon,
                x.shape[0],
                oof_count,
                diagnostics.train_label_distribution["pct_up"],
                diagnostics.train_label_distribution["pct_flat"],
                diagnostics.train_label_distribution["pct_down"],
                diagnostics.oof_label_distribution["pct_up"],
                diagnostics.oof_label_distribution["pct_flat"],
                diagnostics.oof_label_distribution["pct_down"],
                raw_metrics.log_loss,
                raw_metrics.brier_score,
                calibrated_metrics.log_loss,
                calibrated_metrics.brier_score,
                metrics.log_loss,
                metrics.brier_score,
                diagnostics.k_h,
                diagnostics.lambda_h,
                diagnostics.calibration_blend,
                diagnostics.max_calibration_shift,
                diagnostics.sigma_floor,
            )

        self._backtest_output = BacktestOutput(
            metrics_by_horizon=metrics_by_horizon,
            diagnostics_by_horizon=diagnostics_by_horizon,
        )
        ordered = sorted(candles, key=lambda item: item.trade_date)
        self._fit_signature = (len(ordered), ordered[-1].trade_date.isoformat())
        return self

    def predict_proba(self, candles: Sequence[Candle]) -> dict[Horizon, HorizonProbabilityOutput]:
        self._ensure_fitted(candles)
        feature_matrix = self._feature_engine.transform(candles)
        latest_idx = self._latest_feature_index(feature_matrix)
        x_latest = feature_matrix.values[latest_idx : latest_idx + 1]
        vol_latest = feature_matrix.hist_vol_20[latest_idx]
        if not np.isfinite(vol_latest):
            raise ProbabilityModelError("最新样本波动率不可用，无法预测概率")

        outputs: dict[Horizon, HorizonProbabilityOutput] = {}
        for horizon in ("short", "mid", "long"):
            state = self._states[horizon]
            threshold_up = state.k_h * vol_latest * sqrt(float(state.horizon_days))
            threshold_down = -threshold_up

            quantile_predictions: dict[float, np.ndarray] = {
                level: model.predict(x_latest)
                for level, model in state.quantile_models.items()
            }
            q10_level = _nearest_quantile_level(state.quantile_levels, 0.10)
            q50_level = _nearest_quantile_level(state.quantile_levels, 0.50)
            q90_level = _nearest_quantile_level(state.quantile_levels, 0.90)
            q10 = quantile_predictions[q10_level]
            q50 = quantile_predictions[q50_level]
            q90 = quantile_predictions[q90_level]
            down_raw, flat_raw, up_raw, mu, sigma = quantiles_to_raw_probabilities(
                q10=q10,
                q50=q50,
                q90=q90,
                threshold_up=np.asarray([threshold_up], dtype=float),
                threshold_down=np.asarray([threshold_down], dtype=float),
                sigma_floor=state.sigma_floor,
                quantile_predictions=quantile_predictions,
                quantile_levels=state.quantile_levels,
                mapping_mode=self.config.mapper.normalized_raw_prob_mapping_mode(),
                eps=self.config.mapper.eps,
            )
            raw_probs = np.column_stack([down_raw, flat_raw, up_raw])
            raw_probs = refine_raw_probabilities(
                probs=raw_probs,
                horizon_days=state.horizon_days,
                config=self.config.mapper,
            )
            model_calibrated = raw_probs
            if state.calibrator is not None:
                model_calibrated = state.calibrator.predict_proba(
                    np.log(np.clip(raw_probs, self.config.mapper.eps, 1.0))
                )
                if state.calibration_mode.startswith("conservative"):
                    model_calibrated = _temperature_smooth_probs(
                        model_calibrated,
                        temperature=max(self.config.calibrator.conservative_temperature, 1.0),
                        eps=self.config.mapper.eps,
                    )
            calibrated = blend_calibrated_probabilities(
                raw_probs=raw_probs,
                model_calibrated_probs=model_calibrated,
                blend_weight=state.calibration_blend,
                max_shift=state.max_calibration_shift,
                eps=self.config.mapper.eps,
            )
            if self.config.mapper.short_guard_enabled and state.horizon_days <= 5:
                calibrated, _ = _apply_short_guard(
                    raw_probs=raw_probs,
                    calibrated_probs=calibrated,
                    mu=mu,
                    sigma=sigma,
                    horizon_days=state.horizon_days,
                    config=self.config.mapper,
                    eps=self.config.mapper.eps,
                )
            regimes = np.asarray(
                [
                    self._single_regime(
                        vol_value=vol_latest,
                        cutoffs=state.vol_cutoffs,
                        mu=float(mu[0]),
                        sigma=float(sigma[0]),
                    )
                ],
                dtype=object,
            )
            if _is_raw_only_pipeline(
                calibration_mode=state.calibration_mode,
                calibration_blend=state.calibration_blend,
                max_calibration_shift=state.max_calibration_shift,
                lambda_h=state.lambda_h,
            ):
                final_probs = raw_probs
            else:
                if isinstance(calibrated, tuple):
                    calibrated = calibrated[0]
                calibrated = _ensure_prob_matrix(calibrated)
                final_probs = calibrate_and_constrain(
                    raw_probs=raw_probs,
                    calibrated_probs=calibrated,
                    base_probs=state.base_probs,
                    mu=mu,
                    sigma=sigma,
                    lambda_h=state.lambda_h,
                    horizon_days=state.horizon_days,
                    regime_shift_score=state.regime_shift_score,
                    regimes=regimes,
                    config=self.config.mapper,
                )
            final_down, final_flat, final_up = final_probs[0]
            cal_down, cal_flat, cal_up = calibrated[0]
            confidence = float(np.max(final_probs[0]))
            display = self._build_directional_display(
                state=state,
                raw_probs=raw_probs[0],
                calibrated_probs=calibrated[0],
                final_probs=final_probs[0],
                regime=str(regimes[0]),
            )
            outputs[horizon] = HorizonProbabilityOutput(
                horizon=horizon,
                horizon_days=state.horizon_days,
                prob_down_raw=float(raw_probs[0, LABEL_DOWN]),
                prob_flat_raw=float(raw_probs[0, LABEL_FLAT]),
                prob_up_raw=float(raw_probs[0, LABEL_UP]),
                prob_down_calibrated=float(cal_down),
                prob_flat_calibrated=float(cal_flat),
                prob_up_calibrated=float(cal_up),
                prob_down=float(final_down),
                prob_flat=float(final_flat),
                prob_up=float(final_up),
                threshold_up=float(threshold_up),
                threshold_down=float(threshold_down),
                regime=str(regimes[0]),
                mu=float(mu[0]),
                sigma=float(sigma[0]),
                confidence=confidence,
                display_prob_up_raw=float(display["raw_up"]),
                display_prob_down_raw=float(display["raw_down"]),
                display_prob_uncertain_raw=float(display["raw_uncertain"]),
                display_prob_up_calibrated=float(display["cal_up"]),
                display_prob_down_calibrated=float(display["cal_down"]),
                display_prob_uncertain_calibrated=float(display["cal_uncertain"]),
                display_prob_up_post_shrink=float(display["post_shrink_up"]),
                display_prob_down_post_shrink=float(display["post_shrink_down"]),
                display_prob_uncertain_post_shrink=float(display["post_shrink_uncertain"]),
                display_prob_up_post_uncertainty_boost=float(display["post_uncertainty_boost_up"]),
                display_prob_down_post_uncertainty_boost=float(display["post_uncertainty_boost_down"]),
                display_prob_uncertain_post_uncertainty_boost=float(
                    display["post_uncertainty_boost_uncertain"]
                ),
                display_prob_up=float(display["final_up"]),
                display_prob_uncertain=float(display["uncertain"]),
                display_prob_down=float(display["final_down"]),
                raw_uncertain=float(display["raw_uncertain"]),
                final_uncertain=float(display["final_uncertain"]),
                regime_shift_boost_delta=float(display["regime_shift_boost_delta"]),
                high_vol_boost_delta=float(display["high_vol_boost_delta"]),
                total_uncertain_boost_delta=float(display["total_uncertain_boost_delta"]),
                uncertain_ceiling_applied=bool(display["uncertain_ceiling_applied"]),
                signal_strength=float(display["signal_strength"]),
                display_confidence=float(display["confidence"]),
                display_state=str(display["state"]),
                internal_state=str(display["internal_state"]),
                label=str(display["label"]),
                headline_label=str(display["headline_label"]),
                final_state_label=str(display["state"]),
                display_label=str(display["display_label"]),
                top1_prob=float(display["top1_prob"]),
                top2_prob=float(display["top2_prob"]),
                margin=float(display["margin"]),
                display_warning=str(display["warning"]),
                warning_level=str(display["warning_level"]),
                warning_code=str(display["warning_code"]),
                warning_message=str(display["warning_message"]),
            )
        return self._apply_cross_horizon_sanity(outputs)

    def predict(self, candles: Sequence[Candle]) -> TrendSignals:
        probs = self.predict_proba(candles)
        short = _direction_from_probs(probs["short"])
        mid = _direction_from_probs(probs["mid"])
        long = _direction_from_probs(probs["long"])
        return TrendSignals(short=short, mid=mid, long=long)

    def backtest(self, candles: Sequence[Candle]) -> BacktestOutput:
        self._ensure_fitted(candles)
        if self._backtest_output is None:
            raise ProbabilityModelError("backtest 结果不可用")
        return self._backtest_output

    def diagnostics(self, candles: Sequence[Candle]) -> dict[Horizon, HorizonDiagnostics]:
        """Return per-horizon diagnostics from latest fit."""
        self._ensure_fitted(candles)
        return {key: value.diagnostics for key, value in self._states.items()}

    def _ensure_fitted(self, candles: Sequence[Candle]) -> None:
        ordered = sorted(candles, key=lambda item: item.trade_date)
        signature = (len(ordered), ordered[-1].trade_date.isoformat())
        if self._fit_signature != signature or not self._states:
            self.fit(candles)

    @staticmethod
    def _latest_feature_index(feature_matrix: FeatureMatrix) -> int:
        finite_mask = np.isfinite(feature_matrix.values).all(axis=1) & np.isfinite(feature_matrix.hist_vol_20)
        valid_indices = np.where(finite_mask)[0]
        if valid_indices.shape[0] == 0:
            raise ProbabilityModelError("无可用于预测的特征行")
        return int(valid_indices[-1])

    def _k_h(self, horizon_days: int) -> float:
        if horizon_days <= 5:
            return self.config.label.k_5
        if horizon_days <= 20:
            return self.config.label.k_20
        return self.config.label.k_60

    @staticmethod
    def _extend_extreme_regime(
        *,
        regimes: np.ndarray,
        vol_values: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        q90: float,
    ) -> np.ndarray:
        out = regimes.astype(object).copy()
        signal = np.abs(mu) / np.maximum(sigma, 1e-8)
        extreme = (vol_values >= q90) & (signal >= 1.20) & np.isfinite(signal)
        out[extreme] = "high_vol_extreme"
        return out

    @staticmethod
    def _single_regime(
        *,
        vol_value: float,
        cutoffs: Mapping[str, float],
        mu: float,
        sigma: float,
    ) -> str:
        q33 = float(cutoffs.get("q33", 0.0))
        q66 = float(cutoffs.get("q66", 0.0))
        q90 = float(cutoffs.get("q90", 0.0))
        if vol_value <= q33:
            regime = "low_vol"
        elif vol_value <= q66:
            regime = "mid_vol"
        else:
            regime = "high_vol"
        signal = abs(mu) / max(sigma, 1e-8)
        if regime == "high_vol" and vol_value >= q90 and signal >= 1.20:
            return "high_vol_extreme"
        return regime

    def _fit_binary_display_pipeline(
        self,
        *,
        horizon: Horizon,
        horizon_days: int,
        final_probs: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[
        BinaryLogitCalibrator | None,
        str,
        BinaryDirectionChainMetrics,
        dict[str, Any],
    ]:
        directional_mask = (labels == LABEL_UP) | (labels == LABEL_DOWN)
        directional_count = int(np.sum(directional_mask))
        if directional_count < 30:
            fallback = self._empty_binary_chain()
            return (
                None,
                "disabled_insufficient_directional_samples",
                fallback,
                {
                    "enabled": False,
                    "mode": "disabled_insufficient_directional_samples",
                    "directional_oof_rows": directional_count,
                    "reason": "insufficient_directional_samples",
                    "prob_cap": float(self.config.mapper.display_prob_cap),
                },
            )

        directional_probs = final_probs[directional_mask]
        directional_labels = labels[directional_mask]
        y_bin = (directional_labels == LABEL_UP).astype(int)
        up_count = int(np.sum(y_bin == 1))
        down_count = int(np.sum(y_bin == 0))
        if up_count == 0 or down_count == 0:
            fallback = self._empty_binary_chain()
            return (
                None,
                "disabled_single_class",
                fallback,
                {
                    "enabled": False,
                    "mode": "disabled_single_class",
                    "directional_oof_rows": directional_count,
                    "reason": "single_class",
                    "prob_cap": float(self.config.mapper.display_prob_cap),
                },
            )

        raw_up = _directional_conditional_up(
            up_values=directional_probs[:, LABEL_UP],
            down_values=directional_probs[:, LABEL_DOWN],
            eps=self.config.mapper.eps,
        )
        x_all = _binary_direction_features(
            prob_up_conditional=raw_up,
            prob_flat=directional_probs[:, LABEL_FLAT],
            eps=self.config.mapper.eps,
        )
        raw_metrics = evaluate_binary_probabilities(prob_up=raw_up, labels_binary_up=y_bin, bins=10)
        calibrator: BinaryLogitCalibrator | None = None
        mode = "enabled"
        cal_up = raw_up.copy()

        ratio = self.config.calibrator.recent_oof_ratio(horizon_days)
        fit_rows = directional_count
        if directional_count >= 30:
            fit_x = x_all
            fit_y = y_bin
            if ratio < 0.999:
                keep_rows = max(30, int(round(directional_count * ratio)))
                fit_rows = keep_rows
                fit_x = x_all[-keep_rows:]
                fit_y = y_bin[-keep_rows:]
                mode = f"enabled_recent_{ratio:.2f}"
            try:
                calibrator = BinaryLogitCalibrator(self.config.calibrator).fit(fit_x, fit_y)
                cal_up = calibrator.predict_up_prob(x_all)
            except ValueError:
                calibrator = None
                mode = "disabled_fit_error"
                cal_up = raw_up.copy()

        calibrated_metrics = evaluate_binary_probabilities(prob_up=cal_up, labels_binary_up=y_bin, bins=10)
        raw_to_cal_flip_rate = float(np.mean((raw_up >= 0.5) != (cal_up >= 0.5)))
        cap = float(np.clip(self.config.mapper.display_prob_cap, 0.50, 0.99))
        final_up = np.clip(cal_up, 1.0 - cap, cap)
        final_metrics = evaluate_binary_probabilities(prob_up=final_up, labels_binary_up=y_bin, bins=10)

        # Avoid forcing a degraded binary calibrator into production for mid/long.
        if horizon in {"mid", "long"} and calibrator is not None:
            if calibrated_metrics.log_loss > raw_metrics.log_loss + 1e-4:
                calibrator = None
                mode = "auto_disabled_degrade"
                cal_up = raw_up.copy()
                calibrated_metrics = raw_metrics
                final_up = np.clip(cal_up, 1.0 - cap, cap)
                final_metrics = evaluate_binary_probabilities(
                    prob_up=final_up,
                    labels_binary_up=y_bin,
                    bins=10,
                )
                raw_to_cal_flip_rate = 0.0
            else:
                flip_threshold = self.config.mapper.display_binary_flip_rate_threshold(horizon_days)
                if raw_to_cal_flip_rate > flip_threshold + 1e-8:
                    calibrator = None
                    mode = "auto_disabled_flip_rate"
                    cal_up = raw_up.copy()
                    calibrated_metrics = raw_metrics
                    final_up = np.clip(cal_up, 1.0 - cap, cap)
                    final_metrics = evaluate_binary_probabilities(
                        prob_up=final_up,
                        labels_binary_up=y_bin,
                        bins=10,
                    )
                    raw_to_cal_flip_rate = 0.0

        chain = BinaryDirectionChainMetrics(
            raw=raw_metrics,
            calibrated=calibrated_metrics,
            final=final_metrics,
        )
        info = {
            "enabled": calibrator is not None,
            "mode": mode,
            "directional_oof_rows": directional_count,
            "fit_rows": fit_rows,
            "label_up": float(up_count / directional_count),
            "label_down": float(down_count / directional_count),
            "prob_cap": float(self.config.mapper.display_prob_cap),
            "uncertain_threshold": float(self.config.mapper.display_uncertain_threshold),
            "high_conf_threshold": float(self.config.mapper.display_high_conf_threshold),
            "raw_to_cal_flip_rate": raw_to_cal_flip_rate,
            "flip_rate_threshold": self.config.mapper.display_binary_flip_rate_threshold(horizon_days),
            "blend_weight": self.config.mapper.display_binary_blend(horizon_days),
            "max_shift": self.config.mapper.display_binary_max_shift(horizon_days),
        }
        return calibrator, mode, chain, info

    @staticmethod
    def _empty_binary_chain() -> BinaryDirectionChainMetrics:
        zeros = BinaryEvaluationMetrics(log_loss=0.0, brier_score=0.0, accuracy=0.0)
        return BinaryDirectionChainMetrics(raw=zeros, calibrated=zeros, final=zeros)

    def _build_directional_display(
        self,
        *,
        state: _HorizonState,
        raw_probs: np.ndarray,
        calibrated_probs: np.ndarray,
        final_probs: np.ndarray,
        regime: str,
    ) -> dict[str, Any]:
        raw_down, raw_flat, raw_up = _normalize_down_flat_up(raw_probs, eps=self.config.mapper.eps)
        cal_down, cal_flat, cal_up = _normalize_down_flat_up(calibrated_probs, eps=self.config.mapper.eps)
        final_down, final_flat, final_up = _normalize_down_flat_up(final_probs, eps=self.config.mapper.eps)
        raw_uncertain = raw_flat
        cal_uncertain = cal_flat

        warnings: list[str] = []
        direction_mass = max(1.0 - final_flat, self.config.mapper.eps)
        final_up_cond = _directional_conditional_up(
            up_values=np.asarray([final_up]),
            down_values=np.asarray([final_down]),
            eps=self.config.mapper.eps,
        )[0]

        if self.config.mapper.display_use_binary_calibrator and state.binary_calibrator is not None:
            features = _binary_direction_features(
                prob_up_conditional=np.asarray([final_up_cond]),
                prob_flat=np.asarray([final_flat]),
                eps=self.config.mapper.eps,
            )
            model_up_cond = float(state.binary_calibrator.predict_up_prob(features)[0])
            blend_weight = self.config.mapper.display_binary_blend(state.horizon_days)
            max_shift = self.config.mapper.display_binary_max_shift(state.horizon_days)
            mixed_up_cond = blend_weight * model_up_cond + (1.0 - blend_weight) * final_up_cond
            delta = float(np.clip(mixed_up_cond - final_up_cond, -max_shift, max_shift))
            final_up_cond = float(np.clip(final_up_cond + delta, 0.0, 1.0))

        cap = float(np.clip(state.binary_cap, 0.50, 0.99))
        cap_high = cap
        cap_low = 1.0 - cap
        if final_up_cond > cap_high + 1e-12:
            final_up_cond = cap_high
            warnings.append("direction_cap_up")
        elif final_up_cond < cap_low - 1e-12:
            final_up_cond = cap_low
            warnings.append("direction_cap_down")

        final_up = direction_mass * final_up_cond
        final_down = direction_mass * (1.0 - final_up_cond)
        final_flat = 1.0 - final_up - final_down
        final_up, final_down, final_flat = _normalize_up_down_uncertain(
            up=final_up,
            down=final_down,
            uncertain=final_flat,
            eps=self.config.mapper.eps,
        )

        pre_guard = np.asarray([final_up, final_down, final_flat], dtype=float)
        pre_guard_top_idx = int(np.argmax(pre_guard))
        pre_guard_sorted = np.sort(pre_guard)
        pre_guard_margin = float(max(pre_guard_sorted[-1] - pre_guard_sorted[-2], 0.0))

        raw_boost_multiplier = _raw_uncertainty_boost_multiplier(
            raw_uncertain=raw_uncertain,
            soft_threshold=self.config.mapper.display_raw_uncertain_soft_threshold,
            hard_threshold=self.config.mapper.display_raw_uncertain_hard_threshold,
        )
        step_cap = self.config.mapper.display_uncertain_step_cap(state.horizon_days)
        total_boost_cap = min(
            self.config.mapper.display_max_total_uncertain_boost(state.horizon_days),
            step_cap * 2.0,
        )
        boost_remaining = float(max(total_boost_cap, 0.0))
        regime_shift_boost_delta = 0.0
        high_vol_boost_delta = 0.0

        post_shrink_up = final_up
        post_shrink_down = final_down
        post_shrink_uncertain = final_flat
        post_boost_up = final_up
        post_boost_down = final_down
        post_boost_uncertain = final_flat

        allow_uncertain_boost = (
            pre_guard_top_idx != 2
            and pre_guard_margin < self.config.mapper.display_guardrail_margin_freeze
        )

        if allow_uncertain_boost and state.regime_shift_score >= self.config.mapper.display_regime_shift_threshold:
            final_up_cond = _directional_conditional_up(
                up_values=np.asarray([final_up]),
                down_values=np.asarray([final_down]),
                eps=self.config.mapper.eps,
            )[0]
            shrink = min(
                0.18,
                0.06 + 0.30 * (state.regime_shift_score - self.config.mapper.display_regime_shift_threshold),
            )
            shrink = max(0.0, shrink)
            final_up_cond = 0.5 + (final_up_cond - 0.5) * (1.0 - shrink)
            mass = max(1.0 - final_flat, self.config.mapper.eps)
            final_up = mass * final_up_cond
            final_down = mass * (1.0 - final_up_cond)

            desired_boost = (
                self.config.mapper.display_regime_shift_uncertain_boost(state.horizon_days)
                * 0.25
                * raw_boost_multiplier
            )
            actual_boost = min(
                desired_boost,
                step_cap,
                boost_remaining,
                max(0.0, 0.95 - final_flat),
            )
            if actual_boost > self.config.mapper.eps:
                final_flat += actual_boost
                mass = max(1.0 - final_flat, self.config.mapper.eps)
                final_up = mass * final_up_cond
                final_down = mass * (1.0 - final_up_cond)
                boost_remaining = max(0.0, boost_remaining - actual_boost)
                regime_shift_boost_delta = float(actual_boost)
                warnings.append("regime_shift_shrink")

        post_shrink_up = final_up
        post_shrink_down = final_down
        post_shrink_uncertain = final_flat

        if (
            allow_uncertain_boost
            and regime in {"high_vol", "high_vol_extreme"}
            and abs(_directional_conditional_up(
                up_values=np.asarray([final_up]),
                down_values=np.asarray([final_down]),
                eps=self.config.mapper.eps,
            )[0] - 0.5) < 0.14
        ):
            final_up_cond = _directional_conditional_up(
                up_values=np.asarray([final_up]),
                down_values=np.asarray([final_down]),
                eps=self.config.mapper.eps,
            )[0]
            desired_boost = (
                self.config.mapper.display_high_vol_uncertain_boost(state.horizon_days)
                * 0.25
                * raw_boost_multiplier
            )
            actual_boost = min(
                desired_boost,
                step_cap,
                boost_remaining,
                max(0.0, 0.95 - final_flat),
            )
            if actual_boost > self.config.mapper.eps:
                final_flat += actual_boost
                mass = max(1.0 - final_flat, self.config.mapper.eps)
                final_up = mass * final_up_cond
                final_down = mass * (1.0 - final_up_cond)
                boost_remaining = max(0.0, boost_remaining - actual_boost)
                high_vol_boost_delta = float(actual_boost)
                warnings.append("high_vol_uncertain_boost")

        post_boost_up = final_up
        post_boost_down = final_down
        post_boost_uncertain = final_flat

        uncertain_ceiling_applied = False
        uncertain_ceiling = self.config.mapper.display_uncertain_ceiling(state.horizon_days)
        if final_flat > uncertain_ceiling + 1e-12:
            overflow = final_flat - uncertain_ceiling
            final_flat = uncertain_ceiling
            direction_after = max(final_up + final_down, self.config.mapper.eps)
            final_up += overflow * (final_up / direction_after)
            final_down += overflow * (final_down / direction_after)
            uncertain_ceiling_applied = True
            warnings.append("uncertain_ceiling_applied")

        final_up, final_down, final_flat = _normalize_up_down_uncertain(
            up=final_up,
            down=final_down,
            uncertain=final_flat,
            eps=self.config.mapper.eps,
        )

        if (
            (not self.config.mapper.display_allow_top1_reorder)
            and pre_guard_top_idx in {0, 1}
            and int(np.argmax(np.asarray([final_up, final_down, final_flat], dtype=float))) != pre_guard_top_idx
        ):
            final_up = float(pre_guard[0])
            final_down = float(pre_guard[1])
            final_flat = float(pre_guard[2])
            post_boost_up = final_up
            post_boost_down = final_down
            post_boost_uncertain = final_flat
            warnings.append("display_top1_guard_revert")
            regime_shift_boost_delta = 0.0
            high_vol_boost_delta = 0.0

        final_mass = max(final_up + final_down, self.config.mapper.eps)
        final_up_cond = final_up / final_mass
        signal_strength = final_mass * abs(final_up_cond - 0.5) * 2.0
        signal_strength = float(np.clip(signal_strength, 0.0, 1.0))
        total_uncertain_boost_delta = float(max(0.0, final_flat - raw_uncertain))

        state_label = "uncertain"
        if (
            final_flat < self.config.mapper.display_uncertain_threshold
            and signal_strength >= self.config.mapper.display_label_uncertain_confidence
        ):
            if final_up_cond >= self.config.mapper.display_high_conf_threshold:
                state_label = "high-confidence up"
            elif final_up_cond <= (1.0 - self.config.mapper.display_high_conf_threshold):
                state_label = "high-confidence down"
            else:
                state_label = "uncertain"
        probs = np.asarray([final_up, final_flat, final_down], dtype=float)
        sorted_probs = np.sort(probs)
        top1_prob = float(sorted_probs[-1])
        top2_prob = float(sorted_probs[-2])
        margin = float(max(top1_prob - top2_prob, 0.0))
        display_label = _resolve_display_label(
            final_state_label=state_label,
            prob_up=float(final_up),
            prob_down=float(final_down),
            signal_strength=signal_strength,
            top1_prob=top1_prob,
            margin=margin,
            warning_text=";".join(warnings),
            config=self.config.mapper,
        )
        internal_state = state_label
        label = _resolve_direction_label(
            internal_state=internal_state,
            prob_up=float(final_up),
            prob_down=float(final_down),
        )
        headline_label = _headline_label_from_display(display_label)
        warning_code = ";".join(warnings)
        warning_level = _warning_level_from_codes(warning_code)
        warning_message = _warning_message_from_codes(warning_code)

        return {
            "raw_up": float(raw_up),
            "raw_down": float(raw_down),
            "raw_uncertain": float(raw_uncertain),
            "cal_up": float(cal_up),
            "cal_down": float(cal_down),
            "cal_uncertain": float(cal_uncertain),
            "post_shrink_up": float(post_shrink_up),
            "post_shrink_down": float(post_shrink_down),
            "post_shrink_uncertain": float(post_shrink_uncertain),
            "post_uncertainty_boost_up": float(post_boost_up),
            "post_uncertainty_boost_down": float(post_boost_down),
            "post_uncertainty_boost_uncertain": float(post_boost_uncertain),
            "final_up": float(final_up),
            "final_down": float(final_down),
            "uncertain": float(final_flat),
            "final_uncertain": float(final_flat),
            "regime_shift_boost_delta": float(regime_shift_boost_delta),
            "high_vol_boost_delta": float(high_vol_boost_delta),
            "total_uncertain_boost_delta": total_uncertain_boost_delta,
            "uncertain_ceiling_applied": bool(uncertain_ceiling_applied),
            "signal_strength": signal_strength,
            "confidence": signal_strength,
            "state": state_label,
            "internal_state": internal_state,
            "label": label,
            "headline_label": headline_label,
            "top1_prob": top1_prob,
            "top2_prob": top2_prob,
            "margin": margin,
            "display_label": display_label,
            "warning": warning_code,
            "warning_level": warning_level,
            "warning_code": warning_code,
            "warning_message": warning_message,
        }

    def _apply_cross_horizon_sanity(
        self,
        outputs: dict[Horizon, HorizonProbabilityOutput],
    ) -> dict[Horizon, HorizonProbabilityOutput]:
        mid = outputs.get("mid")
        long = outputs.get("long")
        if mid is None or long is None:
            return outputs

        mid_cond = _directional_conditional_up(
            up_values=np.asarray([mid.display_prob_up]),
            down_values=np.asarray([mid.display_prob_down]),
            eps=self.config.mapper.eps,
        )[0]
        long_cond = _directional_conditional_up(
            up_values=np.asarray([long.display_prob_up]),
            down_values=np.asarray([long.display_prob_down]),
            eps=self.config.mapper.eps,
        )[0]
        extreme = float(np.clip(self.config.mapper.display_cross_horizon_extreme_threshold, 0.55, 0.99))
        opposite_extreme = (
            (mid_cond >= extreme and long_cond <= (1.0 - extreme))
            or (mid_cond <= (1.0 - extreme) and long_cond >= extreme)
        )
        if not opposite_extreme:
            return outputs
        if (
            mid.margin >= self.config.mapper.display_guardrail_margin_freeze
            or long.margin >= self.config.mapper.display_guardrail_margin_freeze
        ):
            return outputs

        adjusted = dict(outputs)
        for key in ("mid", "long"):
            item = adjusted[key]
            cond = _directional_conditional_up(
                up_values=np.asarray([item.display_prob_up]),
                down_values=np.asarray([item.display_prob_down]),
                eps=self.config.mapper.eps,
            )[0]
            boost_cap = min(
                self.config.mapper.display_uncertain_step_cap(item.horizon_days),
                0.02,
            )
            uncertain = min(0.95, item.display_prob_uncertain + boost_cap)
            cond = 0.5 + (cond - 0.5) * 0.85
            mass = max(1.0 - uncertain, self.config.mapper.eps)
            up = mass * cond
            down = mass * (1.0 - cond)
            signal_strength = float(np.clip(mass * abs(cond - 0.5) * 2.0, 0.0, 1.0))
            warning = item.display_warning
            if warning:
                warning = f"{warning};cross_horizon_conflict_shrink"
            else:
                warning = "cross_horizon_conflict_shrink"
            state_label = "uncertain"
            if (
                uncertain < self.config.mapper.display_uncertain_threshold
                and signal_strength >= self.config.mapper.display_label_uncertain_confidence
            ):
                if cond >= self.config.mapper.display_high_conf_threshold:
                    state_label = "high-confidence up"
                elif cond <= (1.0 - self.config.mapper.display_high_conf_threshold):
                    state_label = "high-confidence down"
            probs = np.asarray([up, uncertain, down], dtype=float)
            sorted_probs = np.sort(probs)
            top1_prob = float(sorted_probs[-1])
            top2_prob = float(sorted_probs[-2])
            margin = float(max(top1_prob - top2_prob, 0.0))
            display_label = _resolve_display_label(
                final_state_label=state_label,
                prob_up=float(up),
                prob_down=float(down),
                signal_strength=signal_strength,
                top1_prob=top1_prob,
                margin=margin,
                warning_text=warning,
                config=self.config.mapper,
            )
            internal_state = state_label
            label = _resolve_direction_label(
                internal_state=internal_state,
                prob_up=float(up),
                prob_down=float(down),
            )
            headline_label = _headline_label_from_display(display_label)
            warning_level = _warning_level_from_codes(warning)
            warning_message = _warning_message_from_codes(warning)
            adjusted[key] = replace(
                item,
                display_prob_up=up,
                display_prob_down=down,
                display_prob_uncertain=uncertain,
                signal_strength=signal_strength,
                display_confidence=signal_strength,
                display_state=state_label,
                internal_state=internal_state,
                label=label,
                headline_label=headline_label,
                final_state_label=state_label,
                display_label=display_label,
                top1_prob=top1_prob,
                top2_prob=top2_prob,
                margin=margin,
                display_warning=warning,
                warning_level=warning_level,
                warning_code=warning,
                warning_message=warning_message,
                final_uncertain=uncertain,
                total_uncertain_boost_delta=float(max(0.0, uncertain - item.raw_uncertain)),
            )
        return adjusted


def _direction_from_probs(output: HorizonProbabilityOutput) -> str:
    if output.display_label == "uncertain":
        return "sideways"
    if output.display_label in {"mild_up", "strong_up"}:
        return "uptrend"
    if output.display_label in {"mild_down", "strong_down"}:
        return "downtrend"
    return "uptrend" if output.display_prob_up >= output.display_prob_down else "downtrend"


def _directional_conditional_up(
    *,
    up_values: np.ndarray,
    down_values: np.ndarray,
    eps: float,
) -> np.ndarray:
    up = np.clip(up_values, 0.0, 1.0)
    down = np.clip(down_values, 0.0, 1.0)
    total = np.maximum(up + down, eps)
    cond = up / total
    return np.clip(cond, 0.0, 1.0)


def _binary_direction_features(
    *,
    prob_up_conditional: np.ndarray,
    prob_flat: np.ndarray,
    eps: float,
) -> np.ndarray:
    clipped_cond = np.clip(prob_up_conditional, eps, 1.0 - eps)
    logit = np.log(clipped_cond / (1.0 - clipped_cond))
    flat = np.clip(prob_flat, eps, 1.0)
    log_flat = np.log(flat)
    uncertainty = np.clip(flat, 0.0, 1.0)
    return np.column_stack([logit, log_flat, uncertainty])


def _resolve_display_label(
    *,
    final_state_label: str,
    prob_up: float,
    prob_down: float,
    signal_strength: float,
    top1_prob: float,
    margin: float,
    warning_text: str,
    config: ProbabilityMapperConfig,
) -> str:
    if final_state_label == "uncertain":
        uncertain_prob = float(np.clip(1.0 - prob_up - prob_down, 0.0, 1.0))
        direction_up = prob_up >= prob_down
        direction_gap = abs(prob_up - prob_down)
        if (
            uncertain_prob <= config.display_label_uncertain_lean_max_uncertain
            and direction_gap >= config.display_label_uncertain_lean_gap
            and signal_strength >= config.display_label_uncertain_confidence
        ):
            return "mild_up" if direction_up else "mild_down"
        return "uncertain"
    if signal_strength < config.display_label_uncertain_confidence:
        return "uncertain"
    direction_up = prob_up >= prob_down
    if margin < config.display_label_margin_threshold:
        return "mild_up" if direction_up else "mild_down"
    if signal_strength < config.display_label_mild_confidence:
        return "mild_up" if direction_up else "mild_down"
    if (
        top1_prob < config.display_label_strong_top1_threshold
        or margin < config.display_label_strong_margin_threshold
    ):
        return "mild_up" if direction_up else "mild_down"
    if "regime_shift_shrink" in warning_text:
        return "mild_up" if direction_up else "mild_down"
    return "strong_up" if direction_up else "strong_down"


def _resolve_direction_label(
    *,
    internal_state: str,
    prob_up: float,
    prob_down: float,
) -> str:
    state = internal_state.strip().lower()
    if state == "uncertain":
        return "uncertain"
    if "up" in state:
        return "up"
    if "down" in state:
        return "down"
    if abs(prob_up - prob_down) < 1e-12:
        return "uncertain"
    return "up" if prob_up > prob_down else "down"


def _headline_label_from_display(display_label: str) -> str:
    mapping = {
        "strong_up": "明确看多",
        "mild_up": "偏多但置信一般",
        "uncertain": "中性/不确定",
        "mild_down": "偏空但置信一般",
        "strong_down": "明确看空",
    }
    key = display_label.strip().lower()
    if key in mapping:
        return mapping[key]
    if "up" in key:
        return "偏上"
    if "down" in key:
        return "偏下"
    return "不确定"


def _warning_level_from_codes(codes: str) -> str:
    tokens = [token.strip() for token in codes.split(";") if token.strip()]
    if not tokens:
        return "none"
    strong_codes = {
        "direction_cap_up",
        "direction_cap_down",
        "cross_horizon_conflict_shrink",
        "display_top1_guard_revert",
    }
    if any(token in strong_codes for token in tokens):
        return "warning"
    return "info"


def _warning_message_from_codes(codes: str) -> str:
    tokens = [token.strip() for token in codes.split(";") if token.strip()]
    if not tokens:
        return ""
    mapping = {
        "direction_cap_up": "方向概率触发上限压缩，已降低极端化输出。",
        "direction_cap_down": "方向概率触发下限压缩，已降低极端化输出。",
        "regime_shift_shrink": "检测到市场状态漂移，已自动收缩置信度。",
        "high_vol_uncertain_boost": "当前高波动环境，不确定权重已提升。",
        "uncertain_ceiling_applied": "不确定概率触发上限约束，已回补到方向概率。",
        "binary_overcorrection_shrink": "检测到方向校准过度翻转，已回拉到更稳健区间。",
        "cross_horizon_conflict_shrink": "中长期信号冲突，已执行跨周期降置信处理。",
        "display_top1_guard_revert": "展示层触发排序保护，已回退到原始方向排序。",
    }
    messages = [mapping.get(token, token) for token in tokens]
    return "；".join(messages)


def _temperature_smooth_probs(probs: np.ndarray, *, temperature: float, eps: float) -> np.ndarray:
    temp = max(float(temperature), 1.0)
    if temp <= 1.000001:
        return probs
    logits = np.log(np.clip(probs, eps, 1.0)) / temp
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    denom = np.sum(exp_values, axis=1, keepdims=True)
    denom = np.where(denom <= eps, 1.0, denom)
    return exp_values / denom


def _raw_uncertainty_boost_multiplier(
    *,
    raw_uncertain: float,
    soft_threshold: float,
    hard_threshold: float,
) -> float:
    """Decay uncertainty boosts when raw uncertainty is already high."""
    raw = float(np.clip(raw_uncertain, 0.0, 1.0))
    soft = float(np.clip(soft_threshold, 0.0, 1.0))
    hard = float(np.clip(hard_threshold, soft + 1e-6, 1.0))
    if raw <= soft:
        return 1.0
    if raw >= hard:
        return 0.0
    span = max(hard - soft, 1e-6)
    return float(np.clip((hard - raw) / span, 0.0, 1.0))


def _nearest_quantile_level(levels: Sequence[float], target: float) -> float:
    if not levels:
        raise ValueError("quantile levels 不能为空")
    arr = np.asarray(list(levels), dtype=float)
    idx = int(np.argmin(np.abs(arr - float(target))))
    return float(arr[idx])


def _normalize_down_flat_up(
    probs: np.ndarray,
    *,
    eps: float,
) -> tuple[float, float, float]:
    down = float(np.clip(float(probs[LABEL_DOWN]), 0.0, 1.0))
    flat = float(np.clip(float(probs[LABEL_FLAT]), 0.0, 1.0))
    up = float(np.clip(float(probs[LABEL_UP]), 0.0, 1.0))
    total = down + flat + up
    if total <= eps:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return (down / total, flat / total, up / total)


def _normalize_up_down_uncertain(
    *,
    up: float,
    down: float,
    uncertain: float,
    eps: float,
) -> tuple[float, float, float]:
    up_v = float(np.clip(up, 0.0, 1.0))
    down_v = float(np.clip(down, 0.0, 1.0))
    uncertain_v = float(np.clip(uncertain, 0.0, 1.0))
    total = up_v + down_v + uncertain_v
    if total <= eps:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return (up_v / total, down_v / total, uncertain_v / total)


def _label_distribution(labels: np.ndarray) -> dict[str, float]:
    counts = np.bincount(labels.astype(int), minlength=3).astype(float)
    total = float(np.sum(counts))
    if total <= 1e-12:
        return {
            "count": 0.0,
            "pct_up": 0.0,
            "pct_flat": 0.0,
            "pct_down": 0.0,
        }
    return {
        "count": total,
        "pct_up": float(counts[LABEL_UP] / total),
        "pct_flat": float(counts[LABEL_FLAT] / total),
        "pct_down": float(counts[LABEL_DOWN] / total),
    }


def _label_distribution_by_regime(
    *,
    labels: np.ndarray,
    regimes: np.ndarray,
    mask: np.ndarray,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if labels.shape[0] == 0:
        return out
    valid = mask.astype(bool) & np.isfinite(labels)
    if int(np.sum(valid)) == 0:
        return out
    regime_values = regimes[valid].astype(object)
    label_values = labels[valid].astype(int)
    unique_regimes = sorted({str(item) for item in regime_values})
    for regime in unique_regimes:
        regime_mask = np.asarray([str(item) == regime for item in regime_values], dtype=bool)
        if int(np.sum(regime_mask)) == 0:
            continue
        out[regime] = _label_distribution(label_values[regime_mask])
    return out


def _threshold_uncertain_relation(
    *,
    labels: np.ndarray,
    threshold_abs: np.ndarray,
    mask: np.ndarray,
    bins: int,
) -> list[dict[str, float]]:
    valid = mask.astype(bool) & np.isfinite(threshold_abs) & np.isfinite(labels)
    if int(np.sum(valid)) < 20:
        return []
    values = threshold_abs[valid]
    y = labels[valid].astype(int)
    bin_count = max(3, int(bins))
    edges = np.quantile(values, np.linspace(0.0, 1.0, bin_count + 1))
    edges = np.asarray(edges, dtype=float)
    for idx in range(1, edges.shape[0]):
        if edges[idx] <= edges[idx - 1]:
            edges[idx] = edges[idx - 1] + 1e-9
    rows: list[dict[str, float]] = []
    for idx in range(bin_count):
        low = float(edges[idx])
        high = float(edges[idx + 1])
        if idx == bin_count - 1:
            in_bin = (values >= low) & (values <= high)
        else:
            in_bin = (values >= low) & (values < high)
        count = int(np.sum(in_bin))
        if count == 0:
            continue
        uncertain_rate = float(np.mean(y[in_bin] == LABEL_FLAT))
        rows.append(
            {
                "bucket": idx,
                "low": low,
                "high": high,
                "count": float(count),
                "uncertain_rate": uncertain_rate,
            }
        )
    return rows


def _calibration_flip_summary(
    *,
    raw_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    labels: np.ndarray,
    eps: float,
) -> dict[str, float]:
    if raw_probs.shape[0] == 0:
        return {}
    try:
        raw_major = np.argmax(raw_probs, axis=1)
        cal_major = np.argmax(calibrated_probs, axis=1)
    except ValueError:
        return {}
    raw_top1 = np.max(raw_probs, axis=1)
    cal_top1 = np.max(calibrated_probs, axis=1)
    sorted_raw = np.sort(raw_probs, axis=1)
    raw_margin = sorted_raw[:, -1] - sorted_raw[:, -2]
    flip_mask = raw_major != cal_major
    flip_count = int(np.sum(flip_mask))

    y = labels.astype(int)
    idx = np.arange(y.shape[0], dtype=int)
    raw_ll = -np.log(np.clip(raw_probs[idx, y], eps, 1.0))
    cal_ll = -np.log(np.clip(calibrated_probs[idx, y], eps, 1.0))
    onehot = np.eye(3)[y]
    raw_brier = np.sum((raw_probs - onehot) ** 2, axis=1)
    cal_brier = np.sum((calibrated_probs - onehot) ** 2, axis=1)

    strong_mask = (raw_margin >= 0.20) & (raw_major != LABEL_FLAT)
    strong_count = int(np.sum(strong_mask))
    strong_to_uncertain = (
        float(np.mean(cal_major[strong_mask] == LABEL_FLAT))
        if strong_count > 0
        else 0.0
    )
    flip_ll_improve = (
        float(np.mean(cal_ll[flip_mask] < raw_ll[flip_mask]))
        if flip_count > 0
        else 0.0
    )
    flip_brier_improve = (
        float(np.mean(cal_brier[flip_mask] < raw_brier[flip_mask]))
        if flip_count > 0
        else 0.0
    )
    return {
        "flip_rate": float(np.mean(flip_mask)),
        "flip_count": float(flip_count),
        "raw_top1_prob_mean": float(np.mean(raw_top1)),
        "cal_top1_prob_mean": float(np.mean(cal_top1)),
        "flip_logloss_improve_rate": flip_ll_improve,
        "flip_brier_improve_rate": flip_brier_improve,
        "strong_direction_to_uncertain_rate": strong_to_uncertain,
    }


def _short_guard_metrics(
    *,
    raw_probs: np.ndarray,
    guard_probs: np.ndarray,
    labels: np.ndarray,
    strong_mask: np.ndarray,
    eps: float,
) -> dict[str, float]:
    if strong_mask.shape[0] == 0:
        return {}
    y = labels.astype(int)
    idx = np.arange(y.shape[0], dtype=int)
    raw_major = np.argmax(raw_probs, axis=1)
    guard_major = np.argmax(guard_probs, axis=1)
    flip_mask = raw_major != guard_major

    def _logloss(probs: np.ndarray, mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        return float(
            np.mean(-np.log(np.clip(probs[mask][np.arange(np.sum(mask)), y[mask]], eps, 1.0)))
        )

    def _brier(probs: np.ndarray, mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        onehot = np.eye(3)[y[mask]]
        return float(np.mean(np.sum((probs[mask] - onehot) ** 2, axis=1)))

    trigger = strong_mask
    if not np.any(trigger):
        return {"trigger_rate": 0.0}
    return {
        "trigger_rate": float(np.mean(trigger)),
        "trigger_count": float(np.sum(trigger)),
        "trigger_flip_rate_raw_to_guard": float(np.mean(flip_mask[trigger])),
        "trigger_raw_top1_acc": float(np.mean(raw_major[trigger] == y[trigger])),
        "trigger_guard_top1_acc": float(np.mean(guard_major[trigger] == y[trigger])),
        "trigger_raw_logloss": _logloss(raw_probs, trigger),
        "trigger_guard_logloss": _logloss(guard_probs, trigger),
        "trigger_raw_brier": _brier(raw_probs, trigger),
        "trigger_guard_brier": _brier(guard_probs, trigger),
        "trigger_guard_to_uncertain": float(np.mean(guard_major[trigger] == LABEL_FLAT)),
    }


def _ensure_prob_matrix(probs: np.ndarray) -> np.ndarray:
    """Ensure probabilities are 2D (n,3); transpose if shape (3,n)."""
    arr = np.asarray(probs, dtype=float)
    if arr.ndim == 2 and arr.shape[1] != 3 and arr.shape[0] == 3:
        arr = arr.T
    return arr


def _probs_to_dict(values: np.ndarray) -> dict[str, float]:
    return {
        "down": float(values[LABEL_DOWN]),
        "flat": float(values[LABEL_FLAT]),
        "up": float(values[LABEL_UP]),
    }


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.shape[0] == 0:
        return {
            "count": 0.0,
            "min": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "mean": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }
    return {
        "count": float(finite.shape[0]),
        "min": float(np.min(finite)),
        "p25": float(np.quantile(finite, 0.25)),
        "p50": float(np.quantile(finite, 0.50)),
        "mean": float(np.mean(finite)),
        "p75": float(np.quantile(finite, 0.75)),
        "p90": float(np.quantile(finite, 0.90)),
        "max": float(np.max(finite)),
    }


def _apply_short_guard(
    *,
    raw_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    horizon_days: int,
    config: ProbabilityMapperConfig,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend calibrated probs back toward raw for strong short signals only. Returns (guarded, mask)."""
    if not config.short_guard_enabled or horizon_days > 5:
        return calibrated_probs, np.zeros(raw_probs.shape[0], dtype=bool)
    raw = np.clip(raw_probs, eps, 1.0)
    cal = np.clip(calibrated_probs, eps, 1.0)
    margin = np.abs(raw[:, LABEL_UP] - raw[:, LABEL_DOWN])
    raw_uncertain = raw[:, LABEL_FLAT]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu_sigma = np.abs(mu / np.maximum(sigma, eps)).astype(float)
    strong_mask = (
        (margin > config.short_guard_margin)
        & (raw_uncertain < config.short_guard_uncertain_max)
        & (mu_sigma > config.short_guard_mu_sigma_min)
    )
    if not np.any(strong_mask):
        return calibrated_probs, strong_mask
    alpha = float(np.clip(config.short_guard_alpha, 0.0, 1.0))
    guarded = cal.copy()
    guarded[strong_mask] = alpha * cal[strong_mask] + (1.0 - alpha) * raw[strong_mask]
    return guarded, strong_mask


def _chain_shift_summary(
    *,
    raw_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    final_probs: np.ndarray,
) -> dict[str, float]:
    raw_major = np.argmax(raw_probs, axis=1)
    cal_major = np.argmax(calibrated_probs, axis=1)
    final_major = np.argmax(final_probs, axis=1)
    return {
        "mean_abs_shift_raw_to_cal": float(np.mean(np.abs(calibrated_probs - raw_probs))),
        "mean_abs_shift_cal_to_final": float(np.mean(np.abs(final_probs - calibrated_probs))),
        "direction_flip_rate_raw_to_cal": float(np.mean(raw_major != cal_major)),
        "direction_flip_rate_cal_to_final": float(np.mean(cal_major != final_major)),
    }


def _is_raw_only_pipeline(
    *,
    calibration_mode: str,
    calibration_blend: float,
    max_calibration_shift: float,
    lambda_h: float,
) -> bool:
    mode = calibration_mode.strip().lower()
    return (
        mode.startswith("none")
        and abs(calibration_blend) <= 1e-12
        and abs(max_calibration_shift) <= 1e-12
        and abs(lambda_h - 1.0) <= 1e-12
    )
