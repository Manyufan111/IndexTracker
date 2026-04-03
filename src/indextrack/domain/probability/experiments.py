"""Ablation and sensitivity experiments for quantile probability model."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Sequence

from indextrack.app.models import Candle, Horizon
from indextrack.domain.probability.calibrator import CalibratorConfig
from indextrack.domain.probability.model import (
    HorizonDiagnostics,
    HorizonProbabilityOutput,
    MarketProbabilityModel,
    MarketProbabilityModelConfig,
    ProbabilityModelError,
)
from indextrack.domain.probability.probability_mapper import ProbabilityMapperConfig


@dataclass(frozen=True)
class AblationResult:
    """One ablation run summary for one horizon."""

    horizon: Horizon
    group: str
    variant: str
    ok: bool
    note: str
    k_h: float
    lambda_h: float
    calibration_mode: str
    calibration_blend_h: float
    max_calibration_shift_h: float
    sigma_floor_h: float
    raw_log_loss: float
    calibrated_log_loss: float
    final_log_loss: float
    raw_brier: float
    calibrated_brier: float
    final_brier: float
    flip_rate_raw_to_cal: float
    label_up: float
    label_flat: float
    label_down: float
    final_up: float
    final_flat: float
    final_down: float

    @property
    def score(self) -> float:
        # Reliability + stability, accuracy is intentionally not the primary objective.
        return (
            self.final_log_loss
            + 0.60 * self.final_brier
            + 0.20 * self.flip_rate_raw_to_cal
        )


def run_ablation_suite(
    *,
    candles: Sequence[Candle],
    base_config: MarketProbabilityModelConfig,
) -> tuple[list[AblationResult], dict[Horizon, AblationResult] | None]:
    """Run horizon-specific ablations and return all rows + per-horizon recommendation."""
    scenarios: list[tuple[Horizon, str, str, Callable[[MarketProbabilityModelConfig], MarketProbabilityModelConfig]]] = [
        ("short", "short_pipeline", "raw", _short_raw_config),
        ("short", "short_pipeline", "raw_plus_calibration", _short_raw_plus_calibration_config),
        ("short", "short_pipeline", "full_chain", _short_full_chain_config),
        ("mid", "mid_pipeline", "raw", _mid_raw_config),
        ("mid", "mid_pipeline", "raw_plus_shrinkage", _mid_raw_plus_shrinkage_config),
        ("mid", "mid_pipeline", "raw_plus_current_calibration", _mid_raw_plus_current_calibration_config),
        ("mid", "mid_pipeline", "raw_plus_conservative_calibration", _mid_raw_plus_conservative_calibration_config),
        ("long", "long_pipeline", "raw", _long_raw_config),
        ("long", "long_pipeline", "raw_plus_mild_shrinkage", _long_raw_plus_mild_shrinkage_config),
        ("long", "long_pipeline", "raw_plus_current_calibration", _long_raw_plus_current_calibration_config),
        ("long", "long_pipeline", "raw_plus_conservative_calibration", _long_raw_plus_conservative_calibration_config),
        ("long", "long_threshold", "current", lambda cfg: cfg),
        ("long", "long_threshold", "slightly_narrower_k60", lambda cfg: _with_k60(cfg, cfg.label.k_60 * 0.90)),
        ("long", "long_threshold", "narrower_k60", lambda cfg: _with_k60(cfg, cfg.label.k_60 * 0.80)),
        ("long", "long_sigma_floor", "current", lambda cfg: cfg),
        ("long", "long_sigma_floor", "higher_sigma_floor60", lambda cfg: _with_sigma_floor60(cfg, min(0.05, cfg.mapper.sigma_floor_60 * 1.25))),
        ("long", "long_sigma_floor", "lower_sigma_floor60", lambda cfg: _with_sigma_floor60(cfg, max(0.002, cfg.mapper.sigma_floor_60 * 0.80))),
        ("long", "long_lambda", "current", lambda cfg: cfg),
        ("long", "long_lambda", "weaker_shrinkage", lambda cfg: _with_lambda60(cfg, min(1.0, cfg.mapper.lambda_60 + 0.05))),
        ("long", "long_lambda", "stronger_shrinkage", lambda cfg: _with_lambda60(cfg, max(0.65, cfg.mapper.lambda_60 - 0.10))),
        ("mid", "calibration_recent_window", "recent_conservative_50", lambda cfg: _with_recent_calibration(cfg, "mid", 0.50)),
        ("long", "calibration_recent_window", "recent_conservative_45", lambda cfg: _with_recent_calibration(cfg, "long", 0.45)),
        ("long", "prior_adjustment", "recent_prior_blend_45", lambda cfg: _with_long_recent_prior(cfg, weight=0.45)),
        ("long", "prior_adjustment", "recent_prior_blend_65", lambda cfg: _with_long_recent_prior(cfg, weight=0.65)),
    ]

    results: list[AblationResult] = []
    for horizon, group, variant, builder in scenarios:
        cfg = builder(base_config)
        try:
            model = MarketProbabilityModel(cfg).fit(candles)
            diagnostics = model.diagnostics(candles)[horizon]
            latest = model.predict_proba(candles)[horizon]
            results.append(
                _build_result(
                    horizon=horizon,
                    group=group,
                    variant=variant,
                    config=cfg,
                    diagnostics=diagnostics,
                    latest=latest,
                )
            )
        except (ProbabilityModelError, ValueError) as exc:
            results.append(
                _failed_result(
                    horizon=horizon,
                    group=group,
                    variant=variant,
                    config=cfg,
                    note=str(exc),
                )
            )

    ok_results = [item for item in results if item.ok]
    if not ok_results:
        return results, None
    recommended: dict[Horizon, AblationResult] = {}
    for horizon in ("short", "mid", "long"):
        candidates = [item for item in ok_results if item.horizon == horizon]
        if not candidates:
            continue
        recommended[horizon] = min(candidates, key=lambda item: (item.score, item.final_log_loss, item.final_brier))
    return results, (recommended or None)


def format_ablation_table(results: Sequence[AblationResult]) -> str:
    """Render compact plain-text table for CLI logs."""
    headers = [
        "hz",
        "group",
        "variant",
        "status",
        "k",
        "lam",
        "cal_mode",
        "blend",
        "raw_ll",
        "cal_ll",
        "fin_ll",
        "raw_br",
        "cal_br",
        "fin_br",
        "flip",
        "label(up/flat/down)",
        "final(up/flat/down)",
    ]
    rows = [headers]
    for item in results:
        rows.append(
            [
                item.horizon,
                item.group,
                item.variant,
                "ok" if item.ok else "fail",
                f"{item.k_h:.3f}",
                f"{item.lambda_h:.3f}",
                item.calibration_mode,
                f"{item.calibration_blend_h:.2f}",
                f"{item.raw_log_loss:.4f}",
                f"{item.calibrated_log_loss:.4f}",
                f"{item.final_log_loss:.4f}",
                f"{item.raw_brier:.4f}",
                f"{item.calibrated_brier:.4f}",
                f"{item.final_brier:.4f}",
                f"{item.flip_rate_raw_to_cal:.3f}",
                f"{item.label_up:.2f}/{item.label_flat:.2f}/{item.label_down:.2f}",
                f"{item.final_up:.2f}/{item.final_flat:.2f}/{item.final_down:.2f}",
            ]
        )
    widths = [max(len(row[idx]) for row in rows) for idx in range(len(headers))]
    formatted: list[str] = []
    for ridx, row in enumerate(rows):
        line = " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row))
        formatted.append(line)
        if ridx == 0:
            formatted.append("-+-".join("-" * width for width in widths))
    return "\n".join(formatted)


def _build_result(
    *,
    horizon: Horizon,
    group: str,
    variant: str,
    config: MarketProbabilityModelConfig,
    diagnostics: HorizonDiagnostics,
    latest: HorizonProbabilityOutput,
) -> AblationResult:
    horizon_days = diagnostics.horizon_days
    return AblationResult(
        horizon=horizon,
        group=group,
        variant=variant,
        ok=True,
        note="",
        k_h=_k_for_horizon(config, horizon_days),
        lambda_h=config.mapper.lambda_h(horizon_days),
        calibration_mode=diagnostics.calibration_mode,
        calibration_blend_h=diagnostics.calibration_blend,
        max_calibration_shift_h=diagnostics.max_calibration_shift,
        sigma_floor_h=config.mapper.sigma_floor(horizon_days),
        raw_log_loss=diagnostics.chain_metrics.raw.log_loss,
        calibrated_log_loss=diagnostics.chain_metrics.calibrated.log_loss,
        final_log_loss=diagnostics.chain_metrics.final.log_loss,
        raw_brier=diagnostics.chain_metrics.raw.brier_score,
        calibrated_brier=diagnostics.chain_metrics.calibrated.brier_score,
        final_brier=diagnostics.chain_metrics.final.brier_score,
        flip_rate_raw_to_cal=float(diagnostics.chain_shift["direction_flip_rate_raw_to_cal"]),
        label_up=float(diagnostics.oof_label_distribution["pct_up"]),
        label_flat=float(diagnostics.oof_label_distribution["pct_flat"]),
        label_down=float(diagnostics.oof_label_distribution["pct_down"]),
        final_up=float(latest.prob_up),
        final_flat=float(latest.prob_flat),
        final_down=float(latest.prob_down),
    )


def _failed_result(
    *,
    horizon: Horizon,
    group: str,
    variant: str,
    config: MarketProbabilityModelConfig,
    note: str,
) -> AblationResult:
    horizon_days = {"short": 5, "mid": 20, "long": 60}[horizon]
    return AblationResult(
        horizon=horizon,
        group=group,
        variant=variant,
        ok=False,
        note=note,
        k_h=_k_for_horizon(config, horizon_days),
        lambda_h=config.mapper.lambda_h(horizon_days),
        calibration_mode=config.calibrator.normalized_mode_for_horizon_days(horizon_days),
        calibration_blend_h=config.mapper.calibration_blend(horizon_days),
        max_calibration_shift_h=config.mapper.max_calibration_shift(horizon_days),
        sigma_floor_h=config.mapper.sigma_floor(horizon_days),
        raw_log_loss=0.0,
        calibrated_log_loss=0.0,
        final_log_loss=0.0,
        raw_brier=0.0,
        calibrated_brier=0.0,
        final_brier=0.0,
        flip_rate_raw_to_cal=1.0,
        label_up=0.0,
        label_flat=0.0,
        label_down=0.0,
        final_up=0.0,
        final_flat=0.0,
        final_down=0.0,
    )


def _with_k60(config: MarketProbabilityModelConfig, value: float) -> MarketProbabilityModelConfig:
    return replace(config, label=replace(config.label, k_60=max(0.20, min(1.20, value))))


def _with_lambda60(config: MarketProbabilityModelConfig, value: float) -> MarketProbabilityModelConfig:
    mapper = replace(config.mapper, lambda_60=max(0.0, min(1.0, value)))
    return replace(config, mapper=mapper)


def _with_lambda20(config: MarketProbabilityModelConfig, value: float) -> MarketProbabilityModelConfig:
    mapper = replace(config.mapper, lambda_20=max(0.0, min(1.0, value)))
    return replace(config, mapper=mapper)


def _with_sigma_floor60(config: MarketProbabilityModelConfig, value: float) -> MarketProbabilityModelConfig:
    mapper = replace(config.mapper, sigma_floor_60=max(1e-4, min(0.20, value)))
    return replace(config, mapper=mapper)


def _with_calibrator_mode(config: MarketProbabilityModelConfig, mode: str) -> MarketProbabilityModelConfig:
    return replace(config, calibrator=replace(config.calibrator, mode=mode))


def _set_horizon_calibration_mode(
    config: MarketProbabilityModelConfig,
    horizon: Horizon,
    mode: str,
) -> MarketProbabilityModelConfig:
    calibrator: CalibratorConfig = config.calibrator
    if horizon == "short":
        new_cal = replace(calibrator, mode_5=mode)
    elif horizon == "mid":
        new_cal = replace(calibrator, mode_20=mode)
    else:
        new_cal = replace(calibrator, mode_60=mode)
    return replace(config, calibrator=new_cal)


def _set_horizon_recent_oof_ratio(
    config: MarketProbabilityModelConfig,
    horizon: Horizon,
    ratio: float,
) -> MarketProbabilityModelConfig:
    clipped = max(0.05, min(1.0, ratio))
    calibrator: CalibratorConfig = config.calibrator
    if horizon == "short":
        new_cal = replace(calibrator, recent_oof_ratio_5=clipped)
    elif horizon == "mid":
        new_cal = replace(calibrator, recent_oof_ratio_20=clipped)
    else:
        new_cal = replace(calibrator, recent_oof_ratio_60=clipped)
    return replace(config, calibrator=new_cal)


def _set_horizon_blend(
    config: MarketProbabilityModelConfig,
    horizon: Horizon,
    blend: float,
    shift: float,
) -> MarketProbabilityModelConfig:
    mapper: ProbabilityMapperConfig = config.mapper
    if horizon == "short":
        new_mapper = replace(
            mapper,
            calibration_blend_5=max(0.0, min(1.0, blend)),
            max_calibration_shift_5=max(0.0, min(1.0, shift)),
        )
    elif horizon == "mid":
        new_mapper = replace(
            mapper,
            calibration_blend_20=max(0.0, min(1.0, blend)),
            max_calibration_shift_20=max(0.0, min(1.0, shift)),
        )
    else:
        new_mapper = replace(
            mapper,
            calibration_blend_60=max(0.0, min(1.0, blend)),
            max_calibration_shift_60=max(0.0, min(1.0, shift)),
        )
    return replace(config, mapper=new_mapper)


def _set_horizon_lambda(config: MarketProbabilityModelConfig, horizon: Horizon, value: float) -> MarketProbabilityModelConfig:
    mapper: ProbabilityMapperConfig = config.mapper
    lam = max(0.0, min(1.0, value))
    if horizon == "short":
        new_mapper = replace(mapper, lambda_5=lam)
    elif horizon == "mid":
        new_mapper = replace(mapper, lambda_20=lam)
    else:
        new_mapper = replace(mapper, lambda_60=lam)
    return replace(config, mapper=new_mapper)


def _set_raw_like_output(config: MarketProbabilityModelConfig) -> MarketProbabilityModelConfig:
    mapper = replace(config.mapper, prob_cap=1.0)
    return replace(config, mapper=mapper)


def _short_raw_config(config: MarketProbabilityModelConfig) -> MarketProbabilityModelConfig:
    out = _set_raw_like_output(config)
    out = _set_horizon_calibration_mode(out, "short", "none")
    out = _set_horizon_blend(out, "short", 0.0, 0.0)
    out = _set_horizon_lambda(out, "short", 1.0)
    return out


def _short_raw_plus_calibration_config(config: MarketProbabilityModelConfig) -> MarketProbabilityModelConfig:
    out = _set_raw_like_output(config)
    out = _set_horizon_calibration_mode(out, "short", "conservative")
    out = _set_horizon_blend(out, "short", 1.0, 1.0)
    out = _set_horizon_lambda(out, "short", 1.0)
    return out


def _short_full_chain_config(config: MarketProbabilityModelConfig) -> MarketProbabilityModelConfig:
    out = _set_horizon_calibration_mode(config, "short", "conservative")
    out = _set_horizon_blend(out, "short", max(config.mapper.calibration_blend_5, 0.85), max(config.mapper.max_calibration_shift_5, 0.50))
    return out


def _mid_raw_config(config: MarketProbabilityModelConfig) -> MarketProbabilityModelConfig:
    out = _set_raw_like_output(config)
    out = _set_horizon_calibration_mode(out, "mid", "none")
    out = _set_horizon_blend(out, "mid", 0.0, 0.0)
    out = _set_horizon_lambda(out, "mid", 1.0)
    return out


def _mid_raw_plus_shrinkage_config(config: MarketProbabilityModelConfig) -> MarketProbabilityModelConfig:
    out = _mid_raw_config(config)
    return _with_lambda20(out, 0.92)


def _mid_raw_plus_current_calibration_config(config: MarketProbabilityModelConfig) -> MarketProbabilityModelConfig:
    out = _set_raw_like_output(config)
    out = _set_horizon_calibration_mode(out, "mid", config.calibrator.normalized_mode())
    out = _set_horizon_blend(out, "mid", 0.90, 0.45)
    out = _set_horizon_lambda(out, "mid", 1.0)
    return out


def _mid_raw_plus_conservative_calibration_config(config: MarketProbabilityModelConfig) -> MarketProbabilityModelConfig:
    out = _set_raw_like_output(config)
    out = _set_horizon_calibration_mode(out, "mid", "conservative")
    out = _set_horizon_blend(out, "mid", 0.45, 0.25)
    out = _set_horizon_lambda(out, "mid", 1.0)
    return out


def _long_raw_config(config: MarketProbabilityModelConfig) -> MarketProbabilityModelConfig:
    out = _set_raw_like_output(config)
    out = _set_horizon_calibration_mode(out, "long", "none")
    out = _set_horizon_blend(out, "long", 0.0, 0.0)
    out = _set_horizon_lambda(out, "long", 1.0)
    return out


def _long_raw_plus_mild_shrinkage_config(config: MarketProbabilityModelConfig) -> MarketProbabilityModelConfig:
    out = _long_raw_config(config)
    return _set_horizon_lambda(out, "long", 0.97)


def _long_raw_plus_current_calibration_config(config: MarketProbabilityModelConfig) -> MarketProbabilityModelConfig:
    out = _set_raw_like_output(config)
    out = _set_horizon_calibration_mode(out, "long", config.calibrator.normalized_mode())
    out = _set_horizon_blend(out, "long", 0.55, 0.35)
    out = _set_horizon_lambda(out, "long", 1.0)
    return out


def _long_raw_plus_conservative_calibration_config(config: MarketProbabilityModelConfig) -> MarketProbabilityModelConfig:
    out = _set_raw_like_output(config)
    out = _set_horizon_calibration_mode(out, "long", "conservative")
    out = _set_horizon_blend(out, "long", 0.28, 0.18)
    out = _set_horizon_lambda(out, "long", 1.0)
    return out


def _with_recent_calibration(
    config: MarketProbabilityModelConfig,
    horizon: Horizon,
    ratio: float,
) -> MarketProbabilityModelConfig:
    out = _set_raw_like_output(config)
    out = _set_horizon_calibration_mode(out, horizon, "conservative")
    out = _set_horizon_recent_oof_ratio(out, horizon, ratio)
    blend = 0.40 if horizon == "mid" else 0.30
    shift = 0.22 if horizon == "mid" else 0.18
    out = _set_horizon_blend(out, horizon, blend, shift)
    out = _set_horizon_lambda(out, horizon, 1.0)
    return out


def _with_long_recent_prior(config: MarketProbabilityModelConfig, *, weight: float) -> MarketProbabilityModelConfig:
    mapper = replace(
        config.mapper,
        base_prob_recent_weight_60=max(0.0, min(1.0, weight)),
        base_prob_recent_window_60=180,
    )
    out = replace(config, mapper=mapper)
    out = _set_horizon_calibration_mode(out, "long", "none")
    out = _set_horizon_blend(out, "long", 0.0, 0.0)
    out = _set_horizon_lambda(out, "long", 0.97)
    return out


def _k_for_horizon(config: MarketProbabilityModelConfig, horizon_days: int) -> float:
    if horizon_days <= 5:
        return float(config.label.k_5)
    if horizon_days <= 20:
        return float(config.label.k_20)
    return float(config.label.k_60)
