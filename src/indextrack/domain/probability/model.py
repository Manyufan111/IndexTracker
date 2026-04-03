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
    display_prob_up_calibrated: float = 0.0
    display_prob_down_calibrated: float = 0.0
    display_prob_up: float = 0.0
    display_prob_uncertain: float = 0.0
    display_prob_down: float = 0.0
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
    binary_chain_metrics: Mapping[str, Any] = field(default_factory=dict)
    binary_pipeline: Mapping[str, Any] = field(default_factory=dict)

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
            "binary_chain_metrics": dict(self.binary_chain_metrics),
            "binary_pipeline": dict(self.binary_pipeline),
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
            oof_q10 = np.full(indices.shape[0], np.nan, dtype=float)
            oof_q50 = np.full(indices.shape[0], np.nan, dtype=float)
            oof_q90 = np.full(indices.shape[0], np.nan, dtype=float)

            split_count = 0
            for train_idx, valid_idx in expanding_time_series_splits(indices.shape[0], self.config.split):
                split_count += 1
                q10_model = QuantileLinearRegressor(0.10, self.config.quantile).fit(
                    x[train_idx], y_reg[train_idx]
                )
                q50_model = QuantileLinearRegressor(0.50, self.config.quantile).fit(
                    x[train_idx], y_reg[train_idx]
                )
                q90_model = QuantileLinearRegressor(0.90, self.config.quantile).fit(
                    x[train_idx], y_reg[train_idx]
                )

                q10_pred = q10_model.predict(x[valid_idx])
                q50_pred = q50_model.predict(x[valid_idx])
                q90_pred = q90_model.predict(x[valid_idx])
                down_raw, flat_raw, up_raw, mu, sigma = quantiles_to_raw_probabilities(
                    q10=q10_pred,
                    q50=q50_pred,
                    q90=q90_pred,
                    threshold_up=threshold_up[valid_idx],
                    threshold_down=threshold_down[valid_idx],
                    sigma_floor=self.config.mapper.sigma_floor(label_bundle.horizon_days),
                    eps=self.config.mapper.eps,
                )
                oof_raw[valid_idx] = np.column_stack([down_raw, flat_raw, up_raw])
                oof_mu[valid_idx] = mu
                oof_sigma[valid_idx] = sigma
                oof_q10[valid_idx] = q10_pred
                oof_q50[valid_idx] = q50_pred
                oof_q90[valid_idx] = q90_pred

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

            lambda_h = self.config.mapper.lambda_h(label_bundle.horizon_days)
            if _is_raw_only_pipeline(
                calibration_mode=calibration_mode,
                calibration_blend=calibration_blend,
                max_calibration_shift=max_calibration_shift,
                lambda_h=lambda_h,
            ):
                final_oof = oof_raw[oof_mask]
            else:
                final_oof = calibrate_and_constrain(
                    raw_probs=oof_raw[oof_mask],
                    calibrated_probs=calibrated_oof,
                    base_probs=base_probs,
                    mu=oof_mu[oof_mask],
                    sigma=oof_sigma[oof_mask],
                    lambda_h=lambda_h,
                    regimes=regime_labels[oof_mask],
                    config=self.config.mapper,
                )

            raw_metrics = evaluate_probabilities(
                probs=oof_raw[oof_mask],
                labels=y_cls[oof_mask],
                regimes=regime_labels[oof_mask],
            )
            calibrated_metrics = evaluate_probabilities(
                probs=calibrated_oof,
                labels=y_cls[oof_mask],
                regimes=regime_labels[oof_mask],
            )
            metrics = evaluate_probabilities(
                probs=final_oof,
                labels=y_cls[oof_mask],
                regimes=regime_labels[oof_mask],
            )
            chain_metrics = ProbabilityChainMetrics(
                raw=raw_metrics,
                calibrated=calibrated_metrics,
                final=metrics,
            )
            train_label_dist = _label_distribution(y_cls)
            oof_label_dist = _label_distribution(y_cls[oof_mask])
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
            regime_shift_score = (
                abs(train_label_dist["pct_up"] - oof_label_dist["pct_up"])
                + abs(train_label_dist["pct_down"] - oof_label_dist["pct_down"])
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
                q10_stats=_summary_stats(oof_q10[oof_mask]),
                q50_stats=_summary_stats(oof_q50[oof_mask]),
                q90_stats=_summary_stats(oof_q90[oof_mask]),
                sigma_stats=_summary_stats(oof_sigma[oof_mask]),
                binary_chain_metrics=binary_chain_metrics.to_dict(),
                binary_pipeline=binary_pipeline,
            )

            full_q10 = QuantileLinearRegressor(0.10, self.config.quantile).fit(x, y_reg)
            full_q50 = QuantileLinearRegressor(0.50, self.config.quantile).fit(x, y_reg)
            full_q90 = QuantileLinearRegressor(0.90, self.config.quantile).fit(x, y_reg)
            self._states[horizon] = _HorizonState(
                horizon=horizon,
                horizon_days=label_bundle.horizon_days,
                k_h=self._k_h(label_bundle.horizon_days),
                lambda_h=lambda_h,
                sigma_floor=self.config.mapper.sigma_floor(label_bundle.horizon_days),
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

            q10 = state.q10_model.predict(x_latest)
            q50 = state.q50_model.predict(x_latest)
            q90 = state.q90_model.predict(x_latest)
            down_raw, flat_raw, up_raw, mu, sigma = quantiles_to_raw_probabilities(
                q10=q10,
                q50=q50,
                q90=q90,
                threshold_up=np.asarray([threshold_up], dtype=float),
                threshold_down=np.asarray([threshold_down], dtype=float),
                sigma_floor=state.sigma_floor,
                eps=self.config.mapper.eps,
            )
            raw_probs = np.column_stack([down_raw, flat_raw, up_raw])
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
                final_probs = calibrate_and_constrain(
                    raw_probs=raw_probs,
                    calibrated_probs=calibrated,
                    base_probs=state.base_probs,
                    mu=mu,
                    sigma=sigma,
                    lambda_h=state.lambda_h,
                    regimes=regimes,
                    config=self.config.mapper,
                )
            final_down, final_flat, final_up = final_probs[0]
            cal_down, cal_flat, cal_up = calibrated[0]
            confidence = float(np.max(final_probs[0]))
            display = self._build_directional_display(
                state=state,
                final_probs=final_probs[0],
                regime=str(regimes[0]),
            )
            outputs[horizon] = HorizonProbabilityOutput(
                horizon=horizon,
                horizon_days=state.horizon_days,
                prob_down_raw=float(down_raw[0]),
                prob_flat_raw=float(flat_raw[0]),
                prob_up_raw=float(up_raw[0]),
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
                display_prob_up_calibrated=float(display["cal_up"]),
                display_prob_down_calibrated=float(display["cal_down"]),
                display_prob_up=float(display["final_up"]),
                display_prob_uncertain=float(display["uncertain"]),
                display_prob_down=float(display["final_down"]),
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
        final_probs: np.ndarray,
        regime: str,
    ) -> dict[str, float | str]:
        prob_down = float(np.clip(final_probs[LABEL_DOWN], 0.0, 1.0))
        prob_flat = float(np.clip(final_probs[LABEL_FLAT], 0.0, 1.0))
        prob_up = float(np.clip(final_probs[LABEL_UP], 0.0, 1.0))
        prob_sum = prob_down + prob_flat + prob_up
        if prob_sum <= self.config.mapper.eps:
            prob_down, prob_flat, prob_up = 1 / 3, 1 / 3, 1 / 3
        else:
            prob_down /= prob_sum
            prob_flat /= prob_sum
            prob_up /= prob_sum

        raw_up_cond = _directional_conditional_up(
            up_values=np.asarray([prob_up]),
            down_values=np.asarray([prob_down]),
            eps=self.config.mapper.eps,
        )[0]
        direction_mass = max(1.0 - prob_flat, self.config.mapper.eps)
        raw_up = direction_mass * raw_up_cond
        raw_down = direction_mass * (1.0 - raw_up_cond)

        cal_up_cond = raw_up_cond
        if state.binary_calibrator is not None:
            features = _binary_direction_features(
                prob_up_conditional=np.asarray([raw_up_cond]),
                prob_flat=np.asarray([prob_flat]),
                eps=self.config.mapper.eps,
            )
            cal_up_cond = float(state.binary_calibrator.predict_up_prob(features)[0])
        cal_up = direction_mass * cal_up_cond
        cal_down = direction_mass * (1.0 - cal_up_cond)

        final_up_cond = cal_up_cond
        final_flat = prob_flat
        warnings: list[str] = []
        cap = float(np.clip(state.binary_cap, 0.50, 0.99))
        cap_high = cap
        cap_low = 1.0 - cap
        if final_up_cond > cap_high + 1e-12:
            final_up_cond = cap_high
            warnings.append("direction_cap_up")
        elif final_up_cond < cap_low - 1e-12:
            final_up_cond = cap_low
            warnings.append("direction_cap_down")

        if state.regime_shift_score >= self.config.mapper.display_regime_shift_threshold:
            shrink = min(
                0.42,
                0.18 + 0.55 * (state.regime_shift_score - self.config.mapper.display_regime_shift_threshold),
            )
            shrink = max(0.0, shrink)
            if max(final_up_cond, 1.0 - final_up_cond) >= 0.70:
                final_up_cond = 0.5 + (final_up_cond - 0.5) * (1.0 - shrink)
                final_flat = min(0.95, final_flat + 0.15 * shrink)
                warnings.append("regime_shift_shrink")

        if regime in {"high_vol", "high_vol_extreme"} and abs(final_up_cond - 0.5) < 0.18:
            final_flat = min(0.95, final_flat + 0.05)
            warnings.append("high_vol_uncertain_boost")

        final_mass = max(1.0 - final_flat, self.config.mapper.eps)
        final_up = final_mass * final_up_cond
        final_down = final_mass * (1.0 - final_up_cond)
        signal_strength = final_mass * abs(final_up_cond - 0.5) * 2.0
        signal_strength = float(np.clip(signal_strength, 0.0, 1.0))

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
            "cal_up": float(cal_up),
            "cal_down": float(cal_down),
            "final_up": float(final_up),
            "final_down": float(final_down),
            "uncertain": float(final_flat),
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

        adjusted = dict(outputs)
        for key in ("mid", "long"):
            item = adjusted[key]
            cond = _directional_conditional_up(
                up_values=np.asarray([item.display_prob_up]),
                down_values=np.asarray([item.display_prob_down]),
                eps=self.config.mapper.eps,
            )[0]
            uncertain = min(0.95, item.display_prob_uncertain + 0.08)
            cond = 0.5 + (cond - 0.5) * 0.65
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
        "strong_up": "上行",
        "mild_up": "偏上",
        "uncertain": "不确定",
        "mild_down": "偏下",
        "strong_down": "下行",
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
        "cross_horizon_conflict_shrink": "中长期信号冲突，已执行跨周期降置信处理。",
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
