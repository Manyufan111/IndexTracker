#!/usr/bin/env python3
"""Export structured diagnostics for quantile probability model."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import numpy as np

from indextrack.app.orchestrator import AnalysisOrchestrator, DataUnavailableError
from indextrack.cli import _build_probability_model_config
from indextrack.domain.probability.calibrator import SoftmaxCalibrator
from indextrack.domain.probability.evaluator import evaluate_probabilities, metrics_to_dict
from indextrack.domain.probability.experiments import run_ablation_suite
from indextrack.domain.probability.feature_engineering import ProbabilityFeatureEngine
from indextrack.domain.probability.label_builder import (
    LABEL_DOWN,
    LABEL_FLAT,
    LABEL_UP,
    LabelBuilder,
    base_probabilities,
    regime_from_vol,
)
from indextrack.domain.probability.model import MarketProbabilityModel
from indextrack.domain.probability.probability_mapper import (
    ProbabilityMapperConfig,
    blend_calibrated_probabilities,
    calibrate_and_constrain,
    quantiles_to_raw_probabilities,
    refine_raw_probabilities,
)
from indextrack.domain.probability.quantile_model import (
    QuantileLinearRegressor,
    expanding_time_series_splits,
)
from indextrack.infra.config import RuntimeConfig, load_runtime_config
from indextrack.infra.freshness import FreshnessGuard
from indextrack.infra.providers.primary import YahooFinancePrimaryProvider
from indextrack.infra.providers.router import ProviderRouter
from indextrack.infra.providers.secondary import YahooFinanceSecondaryProvider
from indextrack.infra.repository.sqlite_repo import SQLiteRepository

HORIZONS = ("short", "mid", "long")
LABEL_TO_NAME = {LABEL_DOWN: "down", LABEL_FLAT: "flat", LABEL_UP: "up"}
DISPLAY_HEADLINE_MAP = {
    "strong_up": "明确看多",
    "mild_up": "偏多但置信一般",
    "uncertain": "中性/不确定",
    "mild_down": "偏空但置信一般",
    "strong_down": "明确看空",
}
WARNING_MESSAGE_MAP = {
    "direction_cap_up": "方向概率触发上限压缩，已降低极端化输出。",
    "direction_cap_down": "方向概率触发下限压缩，已降低极端化输出。",
    "regime_shift_shrink": "检测到市场状态漂移，已自动收缩置信度。",
    "high_vol_uncertain_boost": "当前高波动环境，不确定权重已提升。",
    "uncertain_ceiling_applied": "不确定概率触发上限约束，已回补到方向概率。",
    "binary_overcorrection_shrink": "检测到方向校准过度翻转，已回拉到更稳健区间。",
    "cross_horizon_conflict_shrink": "中长期信号冲突，已执行跨周期降置信处理。",
    "display_top1_guard_revert": "展示层触发排序保护，已回退到原始方向排序。",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Export quantile diagnostics")
    parser.add_argument("--symbol", default="SP500", choices=["SP500", "NASDAQ"])
    parser.add_argument("--period", default="1Y")
    parser.add_argument("--db-path", default=".data/indextrack.db")
    parser.add_argument("--timezone", default=None)
    parser.add_argument("--output-dir", default=".data/diagnostics")
    parser.add_argument("--recent-rows", type=int, default=12)
    parser.add_argument("--short-extreme-limit", type=int, default=20)
    parser.add_argument("--short-v2-min-count", type=int, default=20)
    parser.add_argument("--short-v2-acc-thresh", type=float, default=0.05)
    parser.add_argument("--short-v2-logloss-thresh", type=float, default=0.03)
    parser.add_argument("--short-v2-flip-thresh", type=float, default=0.20)
    args = parser.parse_args()

    runtime = load_runtime_config()
    timezone_name = args.timezone or runtime.timezone
    timezone = ZoneInfo(timezone_name)
    end = datetime.now(timezone).date()
    trading_days = _period_to_trading_days(args.period)
    lookback_trading_days = max(trading_days, 260)
    lookback_calendar_days = int(lookback_trading_days * 1.7)
    start = end - timedelta(days=lookback_calendar_days)

    provider_router = ProviderRouter(
        primary=YahooFinancePrimaryProvider(request_timeout_sec=runtime.request_timeout_sec),
        secondary=YahooFinanceSecondaryProvider(request_timeout_sec=runtime.request_timeout_sec),
        retries=1,
    )
    repository = SQLiteRepository(db_path=args.db_path)
    freshness_guard = FreshnessGuard(timezone_name=timezone_name)
    orchestrator = AnalysisOrchestrator(
        provider_router=provider_router,
        repository=repository,
        freshness_guard=freshness_guard,
    )

    try:
        outcome = orchestrator.fetch_market_data(
            symbol=args.symbol,
            start=start,
            end=end,
            lookback_days=lookback_calendar_days,
        )
    except DataUnavailableError as exc:
        raise SystemExit(f"数据不可用: {exc}") from exc

    model_config = _build_probability_model_config(runtime)
    diagnostics = _build_quantile_diagnostics(
        candles=outcome.candles,
        model_config=model_config,
        recent_rows=max(args.recent_rows, 5),
        short_extreme_limit=max(args.short_extreme_limit, 1),
    )
    short_v2_exports = _build_short_guard_v2_exports(
        short_detail=diagnostics["horizon_details"]["short"],
        feature_matrix=ProbabilityFeatureEngine().transform(outcome.candles),
        symbol=args.symbol,
        min_count=max(args.short_v2_min_count, 1),
        acc_thresh=float(args.short_v2_acc_thresh),
        logloss_thresh=float(args.short_v2_logloss_thresh),
        flip_thresh=float(args.short_v2_flip_thresh),
    )

    now_tag = datetime.now(timezone).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"{args.symbol.lower()}_quantile_summary_{now_tag}.csv"
    detail_path = output_dir / f"{args.symbol.lower()}_quantile_detailed_report_{now_tag}.md"
    sample_path = output_dir / f"{args.symbol.lower()}_quantile_sample_predictions_{now_tag}.csv"
    short_v2_samples_path = output_dir / f"{args.symbol.lower()}_short_subset_samples_{now_tag}.csv"
    short_v2_bucket_path = output_dir / f"{args.symbol.lower()}_short_subset_bucket_report_{now_tag}.csv"
    short_v2_candidates_path = output_dir / f"{args.symbol.lower()}_short_guard_v2_candidates_{now_tag}.csv"
    json_path = output_dir / f"{args.symbol.lower()}_quantile_diagnostics_{now_tag}.json"
    ablation_path = output_dir / f"{args.symbol.lower()}_quantile_ablation_{now_tag}.csv"

    _write_summary_csv(summary_path, diagnostics["summary_rows"])
    _write_sample_csv(sample_path, diagnostics["sample_rows"])
    _write_short_v2_samples_csv(short_v2_samples_path, short_v2_exports["sample_rows"])
    _write_short_v2_bucket_csv(short_v2_bucket_path, short_v2_exports["bucket_rows"])
    _write_short_v2_candidates_csv(short_v2_candidates_path, short_v2_exports["candidate_rows"])
    _write_ablation_csv(ablation_path, diagnostics["ablation_rows"])
    _write_json(
        json_path,
        {
            "symbol": args.symbol,
            "period": args.period,
            "timezone": timezone_name,
            "data_status": {
                "source": outcome.data_status.source,
                "last_trade_date": outcome.data_status.last_trade_date.isoformat(),
                "fetched_at": outcome.data_status.fetched_at.isoformat(),
                "is_fresh": bool(outcome.data_status.is_fresh),
                "note": outcome.data_status.note,
                "used_cache_fallback": outcome.used_cache_fallback,
            },
            "summary_rows": diagnostics["summary_rows"],
            "horizon_details": diagnostics["horizon_details"],
            "short_guard_v2": short_v2_exports,
            "ablation": diagnostics["ablation"],
            "ablation_rows": diagnostics["ablation_rows"],
            "generated_at": datetime.now(timezone).isoformat(),
        },
    )
    _write_detail_markdown(
        detail_path=detail_path,
        symbol=args.symbol,
        period=args.period,
        data_status=outcome.data_status,
        used_cache_fallback=outcome.used_cache_fallback,
        diagnostics=diagnostics,
        summary_path=summary_path,
        sample_path=sample_path,
        short_v2_samples_path=short_v2_samples_path,
        short_v2_bucket_path=short_v2_bucket_path,
        short_v2_candidates_path=short_v2_candidates_path,
        json_path=json_path,
        ablation_path=ablation_path,
    )
    latest_index_path = output_dir / "diagnostics_latest.md"
    _write_latest_index(
        path=latest_index_path,
        output_dir=output_dir,
        current_bundle={
            "symbol": args.symbol,
            "period": args.period,
            "summary": summary_path,
            "detailed": detail_path,
            "samples": sample_path,
            "short_v2_samples": short_v2_samples_path,
            "short_v2_bucket": short_v2_bucket_path,
            "short_v2_candidates": short_v2_candidates_path,
            "json": json_path,
            "ablation": ablation_path,
            "generated_at": datetime.now(timezone).isoformat(),
            "source": outcome.data_status.source,
            "last_trade_date": outcome.data_status.last_trade_date.isoformat(),
            "cache_fallback": bool(outcome.used_cache_fallback),
        },
    )

    print("导出完成:")
    print(f"- summary: {summary_path}")
    print(f"- detailed report: {detail_path}")
    print(f"- sample predictions csv: {sample_path}")
    print(f"- short v2 samples csv: {short_v2_samples_path}")
    print(f"- short v2 bucket report csv: {short_v2_bucket_path}")
    print(f"- short v2 candidates csv: {short_v2_candidates_path}")
    print(f"- diagnostics json: {json_path}")
    print(f"- ablation csv: {ablation_path}")
    print(f"- latest index: {latest_index_path}")
    return 0


def _build_quantile_diagnostics(
    *,
    candles: list[Any],
    model_config: Any,
    recent_rows: int,
    short_extreme_limit: int,
) -> dict[str, Any]:
    feature_engine = ProbabilityFeatureEngine()
    feature_matrix = feature_engine.transform(candles)
    labels_by_horizon = LabelBuilder(model_config.label).build(feature_matrix)
    finite_feature_mask = np.isfinite(feature_matrix.values).all(axis=1)

    horizon_details: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for horizon in HORIZONS:
        bundle = labels_by_horizon[horizon]
        valid_mask = bundle.valid_mask & finite_feature_mask
        global_indices = np.where(valid_mask)[0]
        if global_indices.shape[0] < (
            model_config.split.min_train_size + model_config.split.min_valid_size + 10
        ):
            raise RuntimeError(
                f"{horizon} 样本不足，无法导出稳定诊断。有效样本={global_indices.shape[0]}"
            )
        detail = _evaluate_one_horizon(
            horizon=horizon,
            global_indices=global_indices,
            feature_matrix=feature_matrix,
            labels=bundle.labels,
            future_return=bundle.future_return,
            threshold_up=bundle.threshold_up,
            threshold_down=bundle.threshold_down,
            config=model_config,
            recent_rows=recent_rows,
        )
        horizon_details[horizon] = detail
        summary_rows.append(_to_summary_row(horizon, detail))
        sample_rows.extend(detail["sample_rows"])

    short_extremes = _extract_short_extremes(horizon_details["short"], short_extreme_limit)
    horizon_details["short"]["extreme_samples"] = short_extremes
    horizon_details["long"]["long_focus"] = _build_long_focus(horizon_details["long"], model_config.mapper)

    ablation_rows, ablation_summary = _build_ablation(candles=candles, base_config=model_config)
    lambda_sensitivity = _build_lambda_sensitivity(candles=candles, base_config=model_config)
    ablation_summary["lambda_sensitivity"] = lambda_sensitivity

    return {
        "summary_rows": summary_rows,
        "horizon_details": horizon_details,
        "sample_rows": sample_rows,
        "ablation_rows": ablation_rows,
        "ablation": ablation_summary,
    }


def _evaluate_one_horizon(
    *,
    horizon: str,
    global_indices: np.ndarray,
    feature_matrix: Any,
    labels: np.ndarray,
    future_return: np.ndarray,
    threshold_up: np.ndarray,
    threshold_down: np.ndarray,
    config: Any,
    recent_rows: int,
) -> dict[str, Any]:
    split_cfg = config.split
    mapper_cfg: ProbabilityMapperConfig = config.mapper
    horizon_days = {"short": 5, "mid": 20, "long": 60}[horizon]
    quantile_levels = mapper_cfg.normalized_quantile_levels()
    q10_level = _nearest_quantile_level_export(quantile_levels, 0.10)
    q50_level = _nearest_quantile_level_export(quantile_levels, 0.50)
    q90_level = _nearest_quantile_level_export(quantile_levels, 0.90)
    mapping_mode = mapper_cfg.normalized_raw_prob_mapping_mode()

    n_total = int(global_indices.shape[0])
    test_size = _choose_test_size(n_total, split_cfg.min_train_size, split_cfg.min_valid_size)
    if n_total - test_size < split_cfg.min_train_size + split_cfg.min_valid_size:
        raise RuntimeError(
            f"{horizon} 无法构造 train/valid/test。n_total={n_total}, test_size={test_size}"
        )
    dev_global = global_indices[:-test_size]
    test_global = global_indices[-test_size:]

    x_dev = feature_matrix.values[dev_global]
    y_reg_dev = future_return[dev_global]
    y_cls_dev = labels[dev_global]
    threshold_up_dev = threshold_up[dev_global]
    threshold_down_dev = threshold_down[dev_global]
    vol_dev = feature_matrix.hist_vol_20[dev_global]

    x_test = feature_matrix.values[test_global]
    y_cls_test = labels[test_global]
    threshold_up_test = threshold_up[test_global]
    threshold_down_test = threshold_down[test_global]
    vol_test = feature_matrix.hist_vol_20[test_global]

    oof_raw = np.full((x_dev.shape[0], 3), np.nan, dtype=float)
    oof_quantiles: dict[float, np.ndarray] = {
        level: np.full(x_dev.shape[0], np.nan, dtype=float)
        for level in quantile_levels
    }
    oof_sigma = np.full(x_dev.shape[0], np.nan, dtype=float)
    oof_mu = np.full(x_dev.shape[0], np.nan, dtype=float)
    oof_seen = np.zeros(x_dev.shape[0], dtype=bool)

    for train_idx, valid_idx in expanding_time_series_splits(x_dev.shape[0], split_cfg):
        fold_predictions: dict[float, np.ndarray] = {}
        for level in quantile_levels:
            model = QuantileLinearRegressor(level, config.quantile).fit(
                x_dev[train_idx],
                y_reg_dev[train_idx],
            )
            fold_predictions[level] = model.predict(x_dev[valid_idx])
        q10_pred = fold_predictions[q10_level]
        q50_pred = fold_predictions[q50_level]
        q90_pred = fold_predictions[q90_level]
        down_raw, flat_raw, up_raw, mu, sigma = quantiles_to_raw_probabilities(
            q10=q10_pred,
            q50=q50_pred,
            q90=q90_pred,
            threshold_up=threshold_up_dev[valid_idx],
            threshold_down=threshold_down_dev[valid_idx],
            sigma_floor=mapper_cfg.sigma_floor(horizon_days),
            quantile_predictions=fold_predictions,
            quantile_levels=quantile_levels,
            mapping_mode=mapping_mode,
            eps=mapper_cfg.eps,
        )
        raw_fold = np.column_stack([down_raw, flat_raw, up_raw])
        raw_fold = refine_raw_probabilities(
            probs=raw_fold,
            horizon_days=horizon_days,
            config=mapper_cfg,
        )
        oof_raw[valid_idx] = raw_fold
        for level, values in fold_predictions.items():
            oof_quantiles[level][valid_idx] = values
        oof_sigma[valid_idx] = sigma
        oof_mu[valid_idx] = mu
        oof_seen[valid_idx] = True

    oof_mask = np.isfinite(oof_raw).all(axis=1)
    oof_count = int(np.sum(oof_mask))
    if oof_count < max(10, split_cfg.min_valid_size):
        raise RuntimeError(f"{horizon} OOF 样本不足，无法导出完整校准诊断。oof_count={oof_count}")

    train_core_mask = ~oof_seen
    if int(np.sum(train_core_mask)) == 0:
        train_core_mask = np.ones_like(train_core_mask, dtype=bool)

    base_probs = base_probabilities(y_cls_dev, np.ones(y_cls_dev.shape[0], dtype=bool))
    mode = config.calibrator.normalized_mode_for_horizon_days(horizon_days)
    effective_mode = mode
    calibrator: SoftmaxCalibrator | None = None

    valid_raw = oof_raw[oof_mask]
    if mode != "none" and oof_count >= 30:
        log_valid_raw = np.log(np.clip(valid_raw, mapper_cfg.eps, 1.0))
        y_valid = y_cls_dev[oof_mask]
        calibration_ratio = config.calibrator.recent_oof_ratio(horizon_days)
        if calibration_ratio < 0.999:
            keep_rows = max(30, int(round(oof_count * calibration_ratio)))
            log_fit = log_valid_raw[-keep_rows:]
            y_fit = y_valid[-keep_rows:]
            effective_mode = f"{mode}_recent_{calibration_ratio:.2f}"
        else:
            log_fit = log_valid_raw
            y_fit = y_valid
        calibrator = SoftmaxCalibrator(config.calibrator).fit(log_fit, y_fit)
        valid_model_cal = calibrator.predict_proba(log_valid_raw)
        if mode == "conservative":
            valid_model_cal = _temperature_smooth_probs(
                valid_model_cal,
                temperature=max(config.calibrator.conservative_temperature, 1.0),
                eps=mapper_cfg.eps,
            )
    else:
        if mode != "none" and oof_count < 30:
            effective_mode = f"{mode}_auto_disabled_oof_lt_30"
        valid_model_cal = valid_raw

    calibration_blend = mapper_cfg.calibration_blend(horizon_days)
    max_shift = mapper_cfg.max_calibration_shift(horizon_days)
    if mode == "none" or "auto_disabled" in effective_mode:
        calibration_blend = 0.0
        max_shift = 0.0
    valid_cal = blend_calibrated_probabilities(
        raw_probs=valid_raw,
        model_calibrated_probs=valid_model_cal,
        blend_weight=calibration_blend,
        max_shift=max_shift,
        eps=mapper_cfg.eps,
    )
    valid_guard, valid_trigger_mask = _apply_short_guard_export(
        raw_probs=valid_raw,
        calibrated_probs=valid_cal,
        mu=oof_mu[oof_mask],
        sigma=oof_sigma[oof_mask],
        horizon_days=horizon_days,
        mapper_cfg=mapper_cfg,
        eps=mapper_cfg.eps,
    )

    regime_dev, vol_cutoffs = regime_from_vol(
        vol_dev,
        np.ones(vol_dev.shape[0], dtype=bool),
        vol_dev,
    )
    regime_dev = _extend_extreme_regime(
        regimes=regime_dev,
        vol_values=vol_dev,
        mu=oof_mu,
        sigma=oof_sigma,
        q90=float(vol_cutoffs.get("q90", 0.0)),
    )
    regime_valid = regime_dev[oof_mask]

    lambda_h = mapper_cfg.lambda_h(horizon_days)
    if _is_raw_only_pipeline(
        calibration_mode=effective_mode,
        calibration_blend=calibration_blend,
        max_calibration_shift=max_shift,
        lambda_h=lambda_h,
    ):
        valid_final = valid_raw
    else:
        valid_final = calibrate_and_constrain(
            raw_probs=valid_raw,
            calibrated_probs=valid_guard,
            base_probs=base_probs,
            mu=oof_mu[oof_mask],
            sigma=oof_sigma[oof_mask],
            lambda_h=lambda_h,
            horizon_days=horizon_days,
            regime_shift_score=0.0,
            regimes=regime_valid,
            config=mapper_cfg,
        )

    full_models: dict[float, QuantileLinearRegressor] = {}
    test_quantiles: dict[float, np.ndarray] = {}
    for level in quantile_levels:
        model = QuantileLinearRegressor(level, config.quantile).fit(x_dev, y_reg_dev)
        full_models[level] = model
        test_quantiles[level] = model.predict(x_test)
    q10_test = test_quantiles[q10_level]
    q50_test = test_quantiles[q50_level]
    q90_test = test_quantiles[q90_level]
    down_raw_test, flat_raw_test, up_raw_test, mu_test, sigma_test = quantiles_to_raw_probabilities(
        q10=q10_test,
        q50=q50_test,
        q90=q90_test,
        threshold_up=threshold_up_test,
        threshold_down=threshold_down_test,
        sigma_floor=mapper_cfg.sigma_floor(horizon_days),
        quantile_predictions=test_quantiles,
        quantile_levels=quantile_levels,
        mapping_mode=mapping_mode,
        eps=mapper_cfg.eps,
    )
    test_raw = np.column_stack([down_raw_test, flat_raw_test, up_raw_test])
    test_raw = refine_raw_probabilities(
        probs=test_raw,
        horizon_days=horizon_days,
        config=mapper_cfg,
    )
    if calibrator is not None:
        test_model_cal = calibrator.predict_proba(
            np.log(np.clip(test_raw, mapper_cfg.eps, 1.0))
        )
        if mode == "conservative":
            test_model_cal = _temperature_smooth_probs(
                test_model_cal,
                temperature=max(config.calibrator.conservative_temperature, 1.0),
                eps=mapper_cfg.eps,
            )
    else:
        test_model_cal = test_raw

    test_cal = blend_calibrated_probabilities(
        raw_probs=test_raw,
        model_calibrated_probs=test_model_cal,
        blend_weight=calibration_blend,
        max_shift=max_shift,
        eps=mapper_cfg.eps,
    )
    test_guard, test_trigger_mask = _apply_short_guard_export(
        raw_probs=test_raw,
        calibrated_probs=test_cal,
        mu=mu_test,
        sigma=sigma_test,
        horizon_days=horizon_days,
        mapper_cfg=mapper_cfg,
        eps=mapper_cfg.eps,
    )
    regime_test = _assign_regimes_from_cutoffs(
        vol_values=vol_test,
        cutoffs=vol_cutoffs,
        mu=mu_test,
        sigma=sigma_test,
    )
    if _is_raw_only_pipeline(
        calibration_mode=effective_mode,
        calibration_blend=calibration_blend,
        max_calibration_shift=max_shift,
        lambda_h=lambda_h,
    ):
        test_final = test_raw
    else:
        test_final = calibrate_and_constrain(
            raw_probs=test_raw,
            calibrated_probs=test_guard,
            base_probs=base_probs,
            mu=mu_test,
            sigma=sigma_test,
            lambda_h=lambda_h,
            horizon_days=horizon_days,
            regime_shift_score=0.0,
            regimes=regime_test,
            config=mapper_cfg,
        )

    metrics_valid = _stage_metrics(valid_raw, valid_guard, valid_final, y_cls_dev[oof_mask], regime_valid)
    metrics_test = _stage_metrics(test_raw, test_guard, test_final, y_cls_test, regime_test)
    train_dist = _distribution(y_cls_dev[train_core_mask])
    valid_dist = _distribution(y_cls_dev[oof_mask])
    test_dist = _distribution(y_cls_test)
    recent_window_rows = min(mapper_cfg.base_prob_recent_window(horizon_days), y_cls_dev.shape[0])
    recent_dist = _distribution(y_cls_dev[-recent_window_rows:])
    train_by_regime = _distribution_by_regime_export(
        labels=y_cls_dev,
        regimes=regime_dev,
        mask=train_core_mask,
    )
    valid_by_regime = _distribution_by_regime_export(
        labels=y_cls_dev,
        regimes=regime_dev,
        mask=oof_mask,
    )
    test_by_regime = _distribution_by_regime_export(
        labels=y_cls_test,
        regimes=regime_test,
        mask=np.ones(y_cls_test.shape[0], dtype=bool),
    )
    threshold_uncertain_bins = _threshold_uncertain_relation_export(
        labels=y_cls_dev,
        threshold_abs=np.abs(threshold_up_dev),
        mask=np.isfinite(threshold_up_dev),
        bins=5,
    )
    calibration_flip_summary = _calibration_flip_summary_export(
        raw_probs=valid_raw,
        calibrated_probs=valid_guard,
        labels=y_cls_dev[oof_mask],
        eps=mapper_cfg.eps,
    )
    short_guard_valid_stats = _short_guard_metrics_export(
        raw_probs=valid_raw,
        guard_probs=valid_guard,
        labels=y_cls_dev[oof_mask],
        trigger_mask=valid_trigger_mask,
        eps=mapper_cfg.eps,
    )
    short_guard_test_stats = _short_guard_metrics_export(
        raw_probs=test_raw,
        guard_probs=test_guard,
        labels=y_cls_test,
        trigger_mask=test_trigger_mask,
        eps=mapper_cfg.eps,
    )
    short_guard_stats = {
        "enabled": bool(mapper_cfg.short_guard_enabled and horizon_days <= 5),
        "alpha": float(mapper_cfg.short_guard_alpha if horizon_days <= 5 else 0.0),
        "margin_threshold": float(mapper_cfg.short_guard_margin if horizon_days <= 5 else 0.0),
        "uncertain_max": float(mapper_cfg.short_guard_uncertain_max if horizon_days <= 5 else 0.0),
        "mu_sigma_min": float(mapper_cfg.short_guard_mu_sigma_min if horizon_days <= 5 else 0.0),
        "trigger_scope": "test",
        **short_guard_test_stats,
        "valid": short_guard_valid_stats,
        "test": short_guard_test_stats,
    }
    regime_shift_score = (
        abs(train_dist["up"] - valid_dist["up"])
        + abs(train_dist["down"] - valid_dist["down"])
    )

    valid_rows = _build_rows(
        dates=[feature_matrix.dates[idx] for idx in dev_global[oof_mask]],
        horizon=horizon,
        horizon_days=horizon_days,
        split_name="valid",
        actual=y_cls_dev[oof_mask],
        raw_probs=valid_raw,
        cal_probs=valid_guard,
        final_probs=valid_final,
        regimes=regime_valid,
        mapper_cfg=mapper_cfg,
        regime_shift_score=regime_shift_score,
        q10=oof_quantiles[q10_level][oof_mask],
        q50=oof_quantiles[q50_level][oof_mask],
        q90=oof_quantiles[q90_level][oof_mask],
        sigma=oof_sigma[oof_mask],
        sigma_floor=mapper_cfg.sigma_floor(horizon_days),
        threshold_low=threshold_down_dev[oof_mask],
        threshold_high=threshold_up_dev[oof_mask],
    )
    test_rows = _build_rows(
        dates=[feature_matrix.dates[idx] for idx in test_global],
        horizon=horizon,
        horizon_days=horizon_days,
        split_name="test",
        actual=y_cls_test,
        raw_probs=test_raw,
        cal_probs=test_guard,
        final_probs=test_final,
        regimes=regime_test,
        mapper_cfg=mapper_cfg,
        regime_shift_score=regime_shift_score,
        q10=q10_test,
        q50=q50_test,
        q90=q90_test,
        sigma=sigma_test,
        sigma_floor=mapper_cfg.sigma_floor(horizon_days),
        threshold_low=threshold_down_test,
        threshold_high=threshold_up_test,
    )
    recent = test_rows[-recent_rows:]

    qgap_valid = oof_quantiles[q90_level][oof_mask] - oof_quantiles[q10_level][oof_mask]
    qgap_test = q90_test - q10_test
    sigma_floor_value = mapper_cfg.sigma_floor(horizon_days)
    sigma_floor_hits_valid = int(np.sum(oof_sigma[oof_mask] <= sigma_floor_value + 1e-12))
    sigma_floor_hits_test = int(np.sum(sigma_test <= sigma_floor_value + 1e-12))

    return {
        "horizon": horizon,
        "horizon_days": horizon_days,
        "label_distribution": {
            "train": train_dist,
            "valid": valid_dist,
            "test": test_dist,
        },
        "recent_label_distribution": {
            "window_rows": recent_window_rows,
            "distribution": recent_dist,
        },
        "label_distribution_by_regime": {
            "train": train_by_regime,
            "valid": valid_by_regime,
            "test": test_by_regime,
        },
        "threshold_uncertain_bins": threshold_uncertain_bins,
        "calibration_flip_summary": calibration_flip_summary,
        "short_guard_stats": short_guard_stats,
        "metrics_valid": metrics_valid,
        "metrics_test": metrics_test,
        "chain_overall_test": {
            "avg_raw_probs": _prob_dict(np.mean(test_raw, axis=0)),
            "avg_calibrated_probs": _prob_dict(np.mean(test_guard, axis=0)),
            "avg_final_probs": _prob_dict(np.mean(test_final, axis=0)),
            "avg_signal_strength": float(np.mean([float(item["signal_strength"]) for item in test_rows])),
            "avg_regime_shift_boost_delta": float(
                np.mean([float(item["regime_shift_boost_delta"]) for item in test_rows])
            ),
            "avg_high_vol_boost_delta": float(
                np.mean([float(item["high_vol_boost_delta"]) for item in test_rows])
            ),
            "avg_total_uncertain_boost_delta": float(
                np.mean([float(item["total_uncertain_boost_delta"]) for item in test_rows])
            ),
            "avg_raw_uncertain": float(np.mean([float(item["raw_uncertain"]) for item in test_rows])),
            "avg_final_uncertain": float(np.mean([float(item["final_uncertain"]) for item in test_rows])),
            "uncertain_ceiling_applied_ratio": float(
                np.mean([1.0 if bool(item["uncertain_ceiling_applied"]) else 0.0 for item in test_rows])
            ),
            "predicted_class_count_final": _display_class_count(test_rows, "predicted_class_final"),
        },
        "recent_test_rows": recent,
        "sample_rows": valid_rows + test_rows,
        "distribution_params": {
            "valid": {
                "q10": _summary_stats(oof_quantiles[q10_level][oof_mask]),
                "q50": _summary_stats(oof_quantiles[q50_level][oof_mask]),
                "q90": _summary_stats(oof_quantiles[q90_level][oof_mask]),
                "implied_sigma": _summary_stats(oof_sigma[oof_mask]),
                "threshold_low": _summary_stats(threshold_down_dev[oof_mask]),
                "threshold_high": _summary_stats(threshold_up_dev[oof_mask]),
                "q90_minus_q10": _summary_stats(qgap_valid),
                "sigma_floor_hits": sigma_floor_hits_valid,
                "sigma_floor": sigma_floor_value,
            },
            "test": {
                "q10": _summary_stats(q10_test),
                "q50": _summary_stats(q50_test),
                "q90": _summary_stats(q90_test),
                "implied_sigma": _summary_stats(sigma_test),
                "threshold_low": _summary_stats(threshold_down_test),
                "threshold_high": _summary_stats(threshold_up_test),
                "q90_minus_q10": _summary_stats(qgap_test),
                "sigma_floor_hits": sigma_floor_hits_test,
                "sigma_floor": sigma_floor_value,
            },
        },
        "threshold_formula": f"threshold_up = k_{horizon_days} * hist_vol_20 * sqrt({horizon_days}), threshold_down = -threshold_up",
        "threshold_params": {
            "k_h": float(_k_h(config, horizon_days)),
            "horizon_days": horizon_days,
            "calibration_mode": effective_mode,
            "oof_count": oof_count,
            "regime_shift_score": float(regime_shift_score),
        },
        "shrinkage": {
            "base_probs": _prob_dict(base_probs),
            "lambda_h": float(lambda_h),
            "calibration_blend": float(calibration_blend),
            "max_calibration_shift": float(max_shift),
            "examples_test": _top_shrinkage_examples(test_rows, limit=8),
        },
    }


def _build_rows(
    *,
    dates: list[Any],
    horizon: str,
    horizon_days: int,
    split_name: str,
    actual: np.ndarray,
    raw_probs: np.ndarray,
    cal_probs: np.ndarray,
    final_probs: np.ndarray,
    regimes: np.ndarray,
    mapper_cfg: ProbabilityMapperConfig,
    regime_shift_score: float,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    sigma: np.ndarray,
    sigma_floor: float,
    threshold_low: np.ndarray,
    threshold_high: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(raw_probs.shape[0]):
        raw = raw_probs[idx]
        cal = cal_probs[idx]
        fin = final_probs[idx]
        display = _derive_display_payload(
            raw_probs=raw,
            cal_probs=cal,
            final_probs=fin,
            horizon_days=horizon_days,
            regime=str(regimes[idx]),
            mapper_cfg=mapper_cfg,
            regime_shift_score=regime_shift_score,
        )
        rows.append(
            {
                "date": dates[idx].isoformat(),
                "horizon": horizon,
                "split": split_name,
                "actual_label": LABEL_TO_NAME[int(actual[idx])],
                "actual_label_id": int(actual[idx]),
                "raw_up": float(display["raw_up"]),
                "raw_down": float(display["raw_down"]),
                "raw_uncertain": float(display["raw_uncertain"]),
                "cal_up": float(display["cal_up"]),
                "cal_down": float(display["cal_down"]),
                "cal_uncertain": float(display["cal_uncertain"]),
                "post_shrink_up": float(display["post_shrink_up"]),
                "post_shrink_down": float(display["post_shrink_down"]),
                "post_shrink_uncertain": float(display["post_shrink_uncertain"]),
                "post_uncertainty_boost_up": float(display["post_uncertainty_boost_up"]),
                "post_uncertainty_boost_down": float(display["post_uncertainty_boost_down"]),
                "post_uncertainty_boost_uncertain": float(
                    display["post_uncertainty_boost_uncertain"]
                ),
                "final_up": float(display["final_up"]),
                "final_down": float(display["final_down"]),
                "final_uncertain": float(display["final_uncertain"]),
                "regime_shift_boost_delta": float(display["regime_shift_boost_delta"]),
                "high_vol_boost_delta": float(display["high_vol_boost_delta"]),
                "total_uncertain_boost_delta": float(display["total_uncertain_boost_delta"]),
                "uncertain_ceiling_applied": bool(display["uncertain_ceiling_applied"]),
                "q10": float(q10[idx]),
                "q50": float(q50[idx]),
                "q90": float(q90[idx]),
                "sigma": float(sigma[idx]),
                "sigma_floor": float(sigma_floor),
                "threshold_low": float(threshold_low[idx]),
                "threshold_high": float(threshold_high[idx]),
                "regime": str(regimes[idx]),
                "internal_state": str(display["internal_state"]),
                "state": str(display["internal_state"]),
                "label": str(display["label"]),
                "headline_label": str(display["headline_label"]),
                "display_label": str(display["display_label"]),
                "top1": float(display["top1"]),
                "top2": float(display["top2"]),
                "margin": float(display["margin"]),
                "signal_strength": float(display["signal_strength"]),
                "confidence": float(display["signal_strength"]),
                "warning_level": str(display["warning_level"]),
                "warning_code": str(display["warning_code"]),
                "warning_message": str(display["warning_message"]),
                "predicted_class_raw": str(display["predicted_class_raw"]),
                "predicted_class_cal": str(display["predicted_class_cal"]),
                "predicted_class_final": str(display["predicted_class_final"]),
            }
        )
    return rows


def _derive_display_payload(
    *,
    raw_probs: np.ndarray,
    cal_probs: np.ndarray,
    final_probs: np.ndarray,
    horizon_days: int,
    regime: str,
    mapper_cfg: ProbabilityMapperConfig,
    regime_shift_score: float,
) -> dict[str, Any]:
    raw_up, raw_down, raw_uncertain = _to_display_probs(raw_probs)
    cal_up, cal_down, cal_uncertain = _to_display_probs(cal_probs)
    final_up, final_down, final_uncertain = _to_display_probs(final_probs)

    stage_raw_up = raw_up
    stage_raw_down = raw_down
    stage_raw_uncertain = raw_uncertain
    stage_cal_up = cal_up
    stage_cal_down = cal_down
    stage_cal_uncertain = cal_uncertain

    warning_codes: list[str] = []
    final_direction_mass = max(final_up + final_down, mapper_cfg.eps)
    final_up_cond = float(np.clip(final_up / final_direction_mass, 0.0, 1.0))
    cap = float(np.clip(mapper_cfg.display_prob_cap, 0.50, 0.99))
    if final_up_cond >= cap + 1e-12:
        final_up_cond = cap
        warning_codes.append("direction_cap_up")
    elif final_up_cond <= (1.0 - cap) - 1e-12:
        final_up_cond = 1.0 - cap
        warning_codes.append("direction_cap_down")

    final_direction_mass = max(1.0 - final_uncertain, mapper_cfg.eps)
    final_up = final_direction_mass * final_up_cond
    final_down = final_direction_mass * (1.0 - final_up_cond)
    final_up, final_down, final_uncertain = _normalize_up_down_uncertain_export(
        up=final_up,
        down=final_down,
        uncertain=final_uncertain,
        eps=mapper_cfg.eps,
    )

    pre_guard = np.asarray([final_up, final_down, final_uncertain], dtype=float)
    pre_guard_top_idx = int(np.argmax(pre_guard))
    pre_guard_sorted = np.sort(pre_guard)
    pre_guard_margin = float(max(pre_guard_sorted[-1] - pre_guard_sorted[-2], 0.0))

    raw_boost_multiplier = _raw_uncertainty_boost_multiplier_export(
        raw_uncertain=stage_raw_uncertain,
        soft_threshold=mapper_cfg.display_raw_uncertain_soft_threshold,
        hard_threshold=mapper_cfg.display_raw_uncertain_hard_threshold,
    )
    step_cap = mapper_cfg.display_uncertain_step_cap(horizon_days)
    total_boost_cap = min(mapper_cfg.display_max_total_uncertain_boost(horizon_days), step_cap * 2.0)
    boost_remaining = max(total_boost_cap, 0.0)
    regime_shift_boost_delta = 0.0
    high_vol_boost_delta = 0.0

    post_shrink_up = final_up
    post_shrink_down = final_down
    post_shrink_uncertain = final_uncertain
    post_uncertainty_boost_up = final_up
    post_uncertainty_boost_down = final_down
    post_uncertainty_boost_uncertain = final_uncertain

    allow_uncertain_boost = (
        pre_guard_top_idx != 2
        and pre_guard_margin < mapper_cfg.display_guardrail_margin_freeze
    )

    if allow_uncertain_boost and regime_shift_score >= mapper_cfg.display_regime_shift_threshold:
        final_direction_mass = max(final_up + final_down, mapper_cfg.eps)
        final_up_cond = float(np.clip(final_up / final_direction_mass, 0.0, 1.0))
        shrink = min(0.18, 0.06 + 0.30 * (regime_shift_score - mapper_cfg.display_regime_shift_threshold))
        shrink = max(0.0, shrink)
        final_up_cond = 0.5 + (final_up_cond - 0.5) * (1.0 - shrink)
        desired_boost = mapper_cfg.display_regime_shift_uncertain_boost(horizon_days) * 0.25 * raw_boost_multiplier
        actual_boost = min(
            desired_boost,
            step_cap,
            boost_remaining,
            max(0.0, 0.95 - final_uncertain),
        )
        if actual_boost > mapper_cfg.eps:
            final_uncertain += actual_boost
            boost_remaining = max(0.0, boost_remaining - actual_boost)
            warning_codes.append("regime_shift_shrink")
            regime_shift_boost_delta = float(actual_boost)
        final_direction_mass = max(1.0 - final_uncertain, mapper_cfg.eps)
        final_up = final_direction_mass * final_up_cond
        final_down = final_direction_mass * (1.0 - final_up_cond)
        final_up, final_down, final_uncertain = _normalize_up_down_uncertain_export(
            up=final_up,
            down=final_down,
            uncertain=final_uncertain,
            eps=mapper_cfg.eps,
        )
        post_shrink_up = final_up
        post_shrink_down = final_down
        post_shrink_uncertain = final_uncertain

    if allow_uncertain_boost and regime in {"high_vol", "high_vol_extreme"}:
        final_direction_mass = max(final_up + final_down, mapper_cfg.eps)
        final_up_cond = float(np.clip(final_up / final_direction_mass, 0.0, 1.0))
        if abs(final_up_cond - 0.5) < 0.14:
            desired_boost = mapper_cfg.display_high_vol_uncertain_boost(horizon_days) * 0.25 * raw_boost_multiplier
            actual_boost = min(
                desired_boost,
                step_cap,
                boost_remaining,
                max(0.0, 0.95 - final_uncertain),
            )
            if actual_boost > mapper_cfg.eps:
                final_uncertain += actual_boost
                boost_remaining = max(0.0, boost_remaining - actual_boost)
                warning_codes.append("high_vol_uncertain_boost")
                high_vol_boost_delta = float(actual_boost)
            final_direction_mass = max(1.0 - final_uncertain, mapper_cfg.eps)
            final_up = final_direction_mass * final_up_cond
            final_down = final_direction_mass * (1.0 - final_up_cond)
            final_up, final_down, final_uncertain = _normalize_up_down_uncertain_export(
                up=final_up,
                down=final_down,
                uncertain=final_uncertain,
                eps=mapper_cfg.eps,
            )
            post_uncertainty_boost_up = final_up
            post_uncertainty_boost_down = final_down
            post_uncertainty_boost_uncertain = final_uncertain

    uncertain_ceiling_applied = False
    uncertain_ceiling = mapper_cfg.display_uncertain_ceiling(horizon_days)
    if final_uncertain > uncertain_ceiling + 1e-12:
        overflow = final_uncertain - uncertain_ceiling
        final_uncertain = uncertain_ceiling
        direction_after = max(final_up + final_down, mapper_cfg.eps)
        final_up += overflow * (final_up / direction_after)
        final_down += overflow * (final_down / direction_after)
        uncertain_ceiling_applied = True
        warning_codes.append("uncertain_ceiling_applied")

    final_up, final_down, final_uncertain = _normalize_up_down_uncertain_export(
        up=final_up,
        down=final_down,
        uncertain=final_uncertain,
        eps=mapper_cfg.eps,
    )

    if (
        (not mapper_cfg.display_allow_top1_reorder)
        and pre_guard_top_idx in {0, 1}
        and int(np.argmax(np.asarray([final_up, final_down, final_uncertain], dtype=float))) != pre_guard_top_idx
    ):
        final_up = float(pre_guard[0])
        final_down = float(pre_guard[1])
        final_uncertain = float(pre_guard[2])
        post_uncertainty_boost_up = final_up
        post_uncertainty_boost_down = final_down
        post_uncertainty_boost_uncertain = final_uncertain
        regime_shift_boost_delta = 0.0
        high_vol_boost_delta = 0.0
        warning_codes.append("display_top1_guard_revert")

    final_mass = max(final_up + final_down, mapper_cfg.eps)
    final_up_cond = final_up / final_mass
    signal_strength = float(np.clip(final_mass * abs(final_up_cond - 0.5) * 2.0, 0.0, 1.0))
    probs = np.asarray([final_up, final_down, final_uncertain], dtype=float)
    probs_sorted = np.sort(probs)
    top1 = float(probs_sorted[-1])
    top2 = float(probs_sorted[-2])
    margin = float(max(top1 - top2, 0.0))

    internal_state = "uncertain"
    if (
        final_uncertain < mapper_cfg.display_uncertain_threshold
        and signal_strength >= mapper_cfg.display_label_uncertain_confidence
    ):
        if final_up_cond >= mapper_cfg.display_high_conf_threshold:
            internal_state = "high-confidence up"
        elif final_up_cond <= (1.0 - mapper_cfg.display_high_conf_threshold):
            internal_state = "high-confidence down"

    warning_code = ";".join(dict.fromkeys(warning_codes))
    display_label = _resolve_display_label_export(
        internal_state=internal_state,
        prob_up=final_up,
        prob_down=final_down,
        signal_strength=signal_strength,
        top1=top1,
        margin=margin,
        warning_code=warning_code,
        mapper_cfg=mapper_cfg,
    )
    label = _resolve_direction_label_export(
        internal_state=internal_state,
        prob_up=final_up,
        prob_down=final_down,
    )
    return {
        "raw_up": stage_raw_up,
        "raw_down": stage_raw_down,
        "raw_uncertain": stage_raw_uncertain,
        "cal_up": stage_cal_up,
        "cal_down": stage_cal_down,
        "cal_uncertain": stage_cal_uncertain,
        "post_shrink_up": post_shrink_up,
        "post_shrink_down": post_shrink_down,
        "post_shrink_uncertain": post_shrink_uncertain,
        "post_uncertainty_boost_up": post_uncertainty_boost_up,
        "post_uncertainty_boost_down": post_uncertainty_boost_down,
        "post_uncertainty_boost_uncertain": post_uncertainty_boost_uncertain,
        "final_up": final_up,
        "final_down": final_down,
        "final_uncertain": final_uncertain,
        "regime_shift_boost_delta": regime_shift_boost_delta,
        "high_vol_boost_delta": high_vol_boost_delta,
        "total_uncertain_boost_delta": float(max(0.0, final_uncertain - stage_raw_uncertain)),
        "uncertain_ceiling_applied": bool(uncertain_ceiling_applied),
        "internal_state": internal_state,
        "label": label,
        "headline_label": DISPLAY_HEADLINE_MAP.get(display_label, "不确定"),
        "display_label": display_label,
        "top1": top1,
        "top2": top2,
        "margin": margin,
        "signal_strength": signal_strength,
        "warning_level": _warning_level_from_codes(warning_code),
        "warning_code": warning_code,
        "warning_message": _warning_message_from_codes(warning_code),
        "predicted_class_raw": _predicted_class(stage_raw_up, stage_raw_down, stage_raw_uncertain),
        "predicted_class_cal": _predicted_class(stage_cal_up, stage_cal_down, stage_cal_uncertain),
        "predicted_class_final": _predicted_class(final_up, final_down, final_uncertain),
    }


def _to_display_probs(values: np.ndarray) -> tuple[float, float, float]:
    up = float(np.clip(values[LABEL_UP], 0.0, 1.0))
    uncertain = float(np.clip(values[LABEL_FLAT], 0.0, 1.0))
    down = float(np.clip(values[LABEL_DOWN], 0.0, 1.0))
    total = up + down + uncertain
    if total <= 1e-12:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return (up / total, down / total, uncertain / total)


def _normalize_up_down_uncertain_export(
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


def _raw_uncertainty_boost_multiplier_export(
    *,
    raw_uncertain: float,
    soft_threshold: float,
    hard_threshold: float,
) -> float:
    raw = float(np.clip(raw_uncertain, 0.0, 1.0))
    soft = float(np.clip(soft_threshold, 0.0, 1.0))
    hard = float(np.clip(hard_threshold, soft + 1e-6, 1.0))
    if raw <= soft:
        return 1.0
    if raw >= hard:
        return 0.0
    span = max(hard - soft, 1e-6)
    return float(np.clip((hard - raw) / span, 0.0, 1.0))


def _predicted_class(prob_up: float, prob_down: float, prob_uncertain: float) -> str:
    if prob_up >= prob_down and prob_up >= prob_uncertain:
        return "up"
    if prob_down >= prob_up and prob_down >= prob_uncertain:
        return "down"
    return "uncertain"


def _nearest_quantile_level_export(levels: Sequence[float], target: float) -> float:
    arr = np.asarray(list(levels), dtype=float)
    if arr.shape[0] == 0:
        raise ValueError("quantile levels 不能为空")
    idx = int(np.argmin(np.abs(arr - float(target))))
    return float(arr[idx])


def _resolve_direction_label_export(*, internal_state: str, prob_up: float, prob_down: float) -> str:
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


def _resolve_display_label_export(
    *,
    internal_state: str,
    prob_up: float,
    prob_down: float,
    signal_strength: float,
    top1: float,
    margin: float,
    warning_code: str,
    mapper_cfg: ProbabilityMapperConfig,
) -> str:
    if internal_state == "uncertain":
        uncertain_prob = float(np.clip(1.0 - prob_up - prob_down, 0.0, 1.0))
        direction_up = prob_up >= prob_down
        direction_gap = abs(prob_up - prob_down)
        if (
            uncertain_prob <= mapper_cfg.display_label_uncertain_lean_max_uncertain
            and direction_gap >= mapper_cfg.display_label_uncertain_lean_gap
            and signal_strength >= mapper_cfg.display_label_uncertain_confidence
        ):
            return "mild_up" if direction_up else "mild_down"
        return "uncertain"
    if signal_strength < mapper_cfg.display_label_uncertain_confidence:
        return "uncertain"
    direction_up = prob_up >= prob_down
    if margin < mapper_cfg.display_label_margin_threshold:
        return "mild_up" if direction_up else "mild_down"
    if signal_strength < mapper_cfg.display_label_mild_confidence:
        return "mild_up" if direction_up else "mild_down"
    if (
        top1 < mapper_cfg.display_label_strong_top1_threshold
        or margin < mapper_cfg.display_label_strong_margin_threshold
    ):
        return "mild_up" if direction_up else "mild_down"
    if "regime_shift_shrink" in warning_code:
        return "mild_up" if direction_up else "mild_down"
    return "strong_up" if direction_up else "strong_down"


def _warning_level_from_codes(codes: str) -> str:
    tokens = [token.strip() for token in codes.split(";") if token.strip()]
    if not tokens:
        return "none"
    if any(
        token in {
            "direction_cap_up",
            "direction_cap_down",
            "cross_horizon_conflict_shrink",
            "display_top1_guard_revert",
        }
        for token in tokens
    ):
        return "warning"
    return "info"


def _warning_message_from_codes(codes: str) -> str:
    tokens = [token.strip() for token in codes.split(";") if token.strip()]
    if not tokens:
        return ""
    messages = [WARNING_MESSAGE_MAP.get(token, token) for token in tokens]
    return "；".join(messages)


def _stage_metrics(
    raw_probs: np.ndarray,
    cal_probs: np.ndarray,
    final_probs: np.ndarray,
    labels: np.ndarray,
    regimes: np.ndarray,
) -> dict[str, Any]:
    raw = metrics_to_dict(evaluate_probabilities(probs=raw_probs, labels=labels, regimes=regimes, bins=10))
    cal = metrics_to_dict(evaluate_probabilities(probs=cal_probs, labels=labels, regimes=regimes, bins=10))
    final = metrics_to_dict(
        evaluate_probabilities(probs=final_probs, labels=labels, regimes=regimes, bins=10)
    )
    return {"raw": raw, "calibrated": cal, "final": final}


def _empty_short_guard_stats_export() -> dict[str, Any]:
    return {
        "trigger_rate": 0.0,
        "trigger_count": 0,
        "triggered_flip_rate": 0.0,
        "triggered_raw_top1_acc": 0.0,
        "triggered_guard_top1_acc": 0.0,
        "triggered_logloss": 0.0,
        "triggered_brier": 0.0,
        "triggered_to_uncertain_rate": 0.0,
    }


def _short_guard_metrics_export(
    *,
    raw_probs: np.ndarray,
    guard_probs: np.ndarray,
    labels: np.ndarray,
    trigger_mask: np.ndarray,
    eps: float,
) -> dict[str, Any]:
    base = _empty_short_guard_stats_export()
    if trigger_mask.shape[0] == 0:
        return base
    trigger = trigger_mask.astype(bool)
    trigger_count = int(np.sum(trigger))
    base["trigger_rate"] = float(np.mean(trigger))
    base["trigger_count"] = trigger_count
    if trigger_count <= 0:
        return base

    y = labels.astype(int)
    raw_major = np.argmax(raw_probs, axis=1)
    guard_major = np.argmax(guard_probs, axis=1)
    flip = raw_major != guard_major

    idx = np.arange(trigger_count, dtype=int)
    y_trigger = y[trigger]
    raw_trigger = np.clip(raw_probs[trigger], eps, 1.0)
    guard_trigger = np.clip(guard_probs[trigger], eps, 1.0)
    onehot = np.eye(3)[y_trigger]

    base["triggered_flip_rate"] = float(np.mean(flip[trigger]))
    base["triggered_raw_top1_acc"] = float(np.mean(raw_major[trigger] == y_trigger))
    base["triggered_guard_top1_acc"] = float(np.mean(guard_major[trigger] == y_trigger))
    base["triggered_logloss"] = float(np.mean(-np.log(guard_trigger[idx, y_trigger])))
    base["triggered_brier"] = float(np.mean(np.sum((guard_trigger - onehot) ** 2, axis=1)))
    base["triggered_to_uncertain_rate"] = float(np.mean(guard_major[trigger] == LABEL_FLAT))
    return base


def _apply_short_guard_export(
    *,
    raw_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    horizon_days: int,
    mapper_cfg: ProbabilityMapperConfig,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    if (not mapper_cfg.short_guard_enabled) or horizon_days > 5:
        return calibrated_probs, np.zeros(raw_probs.shape[0], dtype=bool)
    raw = np.clip(raw_probs, eps, 1.0)
    cal = np.clip(calibrated_probs, eps, 1.0)
    margin = np.abs(raw[:, LABEL_UP] - raw[:, LABEL_DOWN])
    raw_uncertain = raw[:, LABEL_FLAT]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu_sigma = np.abs(mu / np.maximum(sigma, eps)).astype(float)
    trigger = (
        (margin > mapper_cfg.short_guard_margin)
        & (raw_uncertain < mapper_cfg.short_guard_uncertain_max)
        & (mu_sigma > mapper_cfg.short_guard_mu_sigma_min)
    )
    if not np.any(trigger):
        return cal, trigger
    alpha = float(np.clip(mapper_cfg.short_guard_alpha, 0.0, 1.0))
    guarded = cal.copy()
    guarded[trigger] = alpha * cal[trigger] + (1.0 - alpha) * raw[trigger]
    return guarded, trigger


def _build_long_focus(detail: dict[str, Any], mapper_cfg: ProbabilityMapperConfig) -> dict[str, Any]:
    test_dist = detail["distribution_params"]["test"]
    valid_dist = detail["distribution_params"]["valid"]
    return {
        "threshold_formula": detail["threshold_formula"],
        "threshold_params": detail["threshold_params"],
        "threshold_stats_test": {
            "low": test_dist["threshold_low"],
            "high": test_dist["threshold_high"],
        },
        "sigma_distribution": {
            "q90_minus_q10_valid": valid_dist["q90_minus_q10"],
            "q90_minus_q10_test": test_dist["q90_minus_q10"],
            "implied_sigma_valid": valid_dist["implied_sigma"],
            "implied_sigma_test": test_dist["implied_sigma"],
            "sigma_floor": mapper_cfg.sigma_floor_60,
            "sigma_floor_hits_valid": valid_dist["sigma_floor_hits"],
            "sigma_floor_hits_test": test_dist["sigma_floor_hits"],
        },
        "shrinkage_detail": detail["shrinkage"],
    }


def _extract_short_extremes(short_detail: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    rows = [
        row
        for row in short_detail["sample_rows"]
        if max(row["raw_up"], row["raw_down"], row["raw_uncertain"]) > 0.85
    ]
    rows.sort(
        key=lambda item: max(item["raw_up"], item["raw_down"], item["raw_uncertain"]),
        reverse=True,
    )
    return rows[:limit]


def _build_ablation(*, candles: list[Any], base_config: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results, recommended = run_ablation_suite(candles=candles, base_config=base_config)
    rows: list[dict[str, Any]] = []
    for item in results:
        rows.append(
            {
                "horizon": item.horizon,
                "group": item.group,
                "variant": item.variant,
                "ok": bool(item.ok),
                "note": item.note,
                "k_h": item.k_h,
                "lambda_h": item.lambda_h,
                "sigma_floor_h": item.sigma_floor_h,
                "calibration_mode": item.calibration_mode,
                "calibration_blend_h": item.calibration_blend_h,
                "max_calibration_shift_h": item.max_calibration_shift_h,
                "raw_log_loss": item.raw_log_loss,
                "calibrated_log_loss": item.calibrated_log_loss,
                "final_log_loss": item.final_log_loss,
                "raw_brier": item.raw_brier,
                "calibrated_brier": item.calibrated_brier,
                "final_brier": item.final_brier,
                "flip_rate_raw_to_cal": item.flip_rate_raw_to_cal,
                "label_up": item.label_up,
                "label_down": item.label_down,
                "label_uncertain": item.label_flat,
                "final_up": item.final_up,
                "final_down": item.final_down,
                "final_uncertain": item.final_flat,
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["group"]), []).append(row)

    pipeline_improvement: dict[str, dict[str, float | str | None]] = {}
    for horizon in HORIZONS:
        hz_rows = [row for row in rows if row["horizon"] == horizon and row["group"].endswith("_pipeline") and row["ok"]]
        if not hz_rows:
            continue
        by_variant = {str(item["variant"]): item for item in hz_rows}
        best_item = min(
            hz_rows,
            key=lambda item: (
                float(item["final_log_loss"]),
                float(item["final_brier"]),
            ),
        )
        baseline = by_variant.get("raw", best_item)
        pipeline_improvement[horizon] = {
            "best_variant": str(best_item["variant"]),
            "best_final_log_loss": float(best_item["final_log_loss"]),
            "best_final_brier": float(best_item["final_brier"]),
            "raw_to_best_logloss_improve": _safe_diff(
                baseline.get("final_log_loss"),
                best_item.get("final_log_loss"),
            ),
            "raw_to_best_brier_improve": _safe_diff(
                baseline.get("final_brier"),
                best_item.get("final_brier"),
            ),
        }

    recommended_payload: dict[str, Any] = {}
    if isinstance(recommended, dict):
        for horizon in HORIZONS:
            item = recommended.get(horizon)
            if item is None:
                recommended_payload[horizon] = None
                continue
            recommended_payload[horizon] = {
                "group": item.group,
                "variant": item.variant,
                "k_h": item.k_h,
                "lambda_h": item.lambda_h,
                "sigma_floor_h": item.sigma_floor_h,
                "calibration_mode": item.calibration_mode,
                "calibration_blend_h": item.calibration_blend_h,
                "max_calibration_shift_h": item.max_calibration_shift_h,
                "final_log_loss": item.final_log_loss,
                "final_brier": item.final_brier,
            }
    else:
        recommended_payload = {"short": None, "mid": None, "long": None}

    return rows, {
        "groups": grouped,
        "pipeline_improvement_by_horizon": pipeline_improvement,
        "recommended_by_horizon": recommended_payload,
    }


def _build_lambda_sensitivity(*, candles: list[Any], base_config: Any) -> list[dict[str, Any]]:
    values = sorted(
        {
            max(0.0, base_config.mapper.lambda_60 - 0.15),
            max(0.0, base_config.mapper.lambda_60 - 0.05),
            base_config.mapper.lambda_60,
            min(1.0, base_config.mapper.lambda_60 + 0.05),
            min(1.0, base_config.mapper.lambda_60 + 0.15),
        }
    )
    rows: list[dict[str, Any]] = []
    for value in values:
        cfg = replace(base_config, mapper=replace(base_config.mapper, lambda_60=value))
        model = MarketProbabilityModel(cfg).fit(candles)
        diag = model.diagnostics(candles)["long"]
        stability = max(0.0, 1.0 - float(diag.chain_shift["mean_abs_shift_cal_to_final"]))
        rows.append(
            {
                "lambda_60": float(value),
                "final_log_loss": float(diag.chain_metrics.final.log_loss),
                "final_brier": float(diag.chain_metrics.final.brier_score),
                "probability_stability": float(stability),
                "mean_abs_shift_cal_to_final": float(diag.chain_shift["mean_abs_shift_cal_to_final"]),
            }
        )
    return rows


def _to_summary_row(horizon: str, detail: dict[str, Any]) -> dict[str, Any]:
    metrics_test = detail["metrics_test"]
    dist = detail["label_distribution"]
    threshold_test = detail["distribution_params"]["test"]
    return {
        "horizon": horizon,
        "train_distribution": _dist_str(dist["train"]),
        "valid_distribution": _dist_str(dist["valid"]),
        "test_distribution": _dist_str(dist["test"]),
        "raw_log_loss": metrics_test["raw"]["log_loss"],
        "calibrated_log_loss": metrics_test["calibrated"]["log_loss"],
        "final_log_loss": metrics_test["final"]["log_loss"],
        "raw_brier": metrics_test["raw"]["brier_score"],
        "calibrated_brier": metrics_test["calibrated"]["brier_score"],
        "final_brier": metrics_test["final"]["brier_score"],
        "current_threshold_low_p50": threshold_test["threshold_low"]["median"],
        "current_threshold_high_p50": threshold_test["threshold_high"]["median"],
        "current_lambda": detail["shrinkage"]["lambda_h"],
    }


def _build_short_guard_v2_exports(
    *,
    short_detail: dict[str, Any],
    feature_matrix: Any,
    symbol: str,
    min_count: int,
    acc_thresh: float,
    logloss_thresh: float,
    flip_thresh: float,
) -> dict[str, Any]:
    short_rows = list(short_detail.get("sample_rows", []))
    if not short_rows:
        return {"sample_rows": [], "bucket_rows": [], "candidate_rows": [], "config": {}}

    date_to_index = {item.isoformat(): idx for idx, item in enumerate(feature_matrix.dates)}
    feature_name_to_idx = {name: idx for idx, name in enumerate(feature_matrix.feature_names)}
    ma_spread_idx = feature_name_to_idx.get("ma_spread_5_20")
    drawdown_idx = feature_name_to_idx.get("drawdown_20")
    if ma_spread_idx is None or drawdown_idx is None:
        raise RuntimeError("feature matrix 缺少 short v2 分桶所需字段: ma_spread_5_20/drawdown_20")

    spreads: list[float] = []
    drawdowns: list[float] = []
    for row in short_rows:
        idx = date_to_index.get(str(row["date"]))
        if idx is None:
            continue
        spread_v = float(feature_matrix.values[idx, ma_spread_idx])
        drawdown_v = float(feature_matrix.values[idx, drawdown_idx])
        if np.isfinite(spread_v):
            spreads.append(abs(spread_v))
        if np.isfinite(drawdown_v):
            drawdowns.append(drawdown_v)
    spread_q33 = float(np.quantile(np.asarray(spreads, dtype=float), 0.33)) if spreads else 0.003
    spread_q66 = float(np.quantile(np.asarray(spreads, dtype=float), 0.66)) if spreads else 0.008
    dd_q33 = float(np.quantile(np.asarray(drawdowns, dtype=float), 0.33)) if drawdowns else -0.03
    dd_q66 = float(np.quantile(np.asarray(drawdowns, dtype=float), 0.66)) if drawdowns else -0.01

    sample_rows: list[dict[str, Any]] = []
    for row in short_rows:
        sample = _to_short_v2_sample_row(
            row=row,
            symbol=symbol,
            date_to_index=date_to_index,
            feature_matrix=feature_matrix,
            ma_spread_idx=ma_spread_idx,
            drawdown_idx=drawdown_idx,
            spread_q33=spread_q33,
            spread_q66=spread_q66,
            dd_q33=dd_q33,
            dd_q66=dd_q66,
        )
        sample_rows.append(sample)

    bucket_rows = _build_short_v2_bucket_rows(sample_rows)
    candidate_rows = _build_short_v2_candidate_rows(
        bucket_rows=bucket_rows,
        min_count=min_count,
        acc_thresh=acc_thresh,
        logloss_thresh=logloss_thresh,
        flip_thresh=flip_thresh,
    )
    return {
        "sample_rows": sample_rows,
        "bucket_rows": bucket_rows,
        "candidate_rows": candidate_rows,
        "config": {
            "min_count": int(min_count),
            "acc_thresh": float(acc_thresh),
            "logloss_thresh": float(logloss_thresh),
            "flip_thresh": float(flip_thresh),
            "trend_abs_q33": spread_q33,
            "trend_abs_q66": spread_q66,
            "drawdown_q33": dd_q33,
            "drawdown_q66": dd_q66,
        },
    }


def _to_short_v2_sample_row(
    *,
    row: dict[str, Any],
    symbol: str,
    date_to_index: dict[str, int],
    feature_matrix: Any,
    ma_spread_idx: int,
    drawdown_idx: int,
    spread_q33: float,
    spread_q66: float,
    dd_q33: float,
    dd_q66: float,
) -> dict[str, Any]:
    actual_label = str(row["actual_label"]).strip().lower()
    actual_idx = {"down": 0, "flat": 1, "up": 2}.get(actual_label, 1)
    raw_probs = np.asarray([float(row["raw_down"]), float(row["raw_uncertain"]), float(row["raw_up"])], dtype=float)
    cal_probs = np.asarray([float(row["cal_down"]), float(row["cal_uncertain"]), float(row["cal_up"])], dtype=float)
    raw_probs = raw_probs / max(float(np.sum(raw_probs)), 1e-12)
    cal_probs = cal_probs / max(float(np.sum(cal_probs)), 1e-12)
    raw_top_idx = int(np.argmax(raw_probs))
    cal_top_idx = int(np.argmax(cal_probs))
    raw_top1 = LABEL_TO_NAME.get(raw_top_idx, "flat")
    cal_top1 = LABEL_TO_NAME.get(cal_top_idx, "flat")
    raw_top1_prob = float(raw_probs[raw_top_idx])
    cal_top1_prob = float(cal_probs[cal_top_idx])
    raw_margin = float(max(np.sort(raw_probs)[-1] - np.sort(raw_probs)[-2], 0.0))
    cal_margin = float(max(np.sort(cal_probs)[-1] - np.sort(cal_probs)[-2], 0.0))
    raw_correct = int(raw_top_idx == actual_idx)
    cal_correct = int(cal_top_idx == actual_idx)
    flipped = int(raw_top_idx != cal_top_idx)
    flip_improved = int((raw_correct == 0) and (cal_correct == 1))

    raw_logloss = float(-np.log(np.clip(raw_probs[actual_idx], 1e-12, 1.0)))
    cal_logloss = float(-np.log(np.clip(cal_probs[actual_idx], 1e-12, 1.0)))
    one_hot = np.zeros(3, dtype=float)
    one_hot[actual_idx] = 1.0
    raw_brier = float(np.sum((raw_probs - one_hot) ** 2))
    cal_brier = float(np.sum((cal_probs - one_hot) ** 2))

    mu = float(row["q50"])
    sigma = float(row["sigma"])
    mu_sigma_abs = float(abs(mu) / max(abs(sigma), 1e-12))

    idx = date_to_index.get(str(row["date"]))
    ma_spread = float(feature_matrix.values[idx, ma_spread_idx]) if idx is not None else float("nan")
    drawdown = float(feature_matrix.values[idx, drawdown_idx]) if idx is not None else float("nan")

    trend_bucket = _trend_bucket(ma_spread, spread_q33, spread_q66)
    drawdown_bucket = _drawdown_bucket(drawdown, dd_q33, dd_q66)
    vol_bucket = _normalize_vol_bucket(str(row.get("regime", "unknown")))

    return {
        "date": str(row["date"]),
        "symbol": symbol,
        "split": str(row["split"]),
        "actual_label": actual_label,
        "raw_up": float(row["raw_up"]),
        "raw_down": float(row["raw_down"]),
        "raw_uncertain": float(row["raw_uncertain"]),
        "cal_up": float(row["cal_up"]),
        "cal_down": float(row["cal_down"]),
        "cal_uncertain": float(row["cal_uncertain"]),
        "raw_top1": raw_top1,
        "cal_top1": cal_top1,
        "raw_top1_prob": raw_top1_prob,
        "cal_top1_prob": cal_top1_prob,
        "raw_margin": raw_margin,
        "cal_margin": cal_margin,
        "mu": mu,
        "sigma": sigma,
        "mu_sigma_abs": mu_sigma_abs,
        "regime": str(row.get("regime", "unknown")),
        "vol_bucket": vol_bucket,
        "trend_bucket": trend_bucket,
        "drawdown_bucket": drawdown_bucket,
        "regime_combo": f"{vol_bucket}|{trend_bucket}",
        "regime_combo_drawdown": f"{vol_bucket}|{drawdown_bucket}",
        "raw_correct": raw_correct,
        "cal_correct": cal_correct,
        "flipped": flipped,
        "flip_improved": flip_improved,
        "raw_logloss": raw_logloss,
        "cal_logloss": cal_logloss,
        "raw_brier": raw_brier,
        "cal_brier": cal_brier,
    }


def _normalize_vol_bucket(regime: str) -> str:
    key = regime.strip().lower()
    if key in {"low_vol", "mid_vol", "high_vol"}:
        return key
    if key == "high_vol_extreme":
        return "high_vol"
    return "unknown"


def _trend_bucket(value: float, q33: float, q66: float) -> str:
    if not np.isfinite(value):
        return "unknown_trend"
    abs_value = abs(float(value))
    if abs_value <= q33:
        return "weak_trend"
    if abs_value >= q66:
        return "strong_trend"
    return "mid_trend"


def _drawdown_bucket(value: float, q33: float, q66: float) -> str:
    if not np.isfinite(value):
        return "unknown_drawdown"
    if value <= q33:
        return "deep_drawdown"
    if value <= q66:
        return "mid_drawdown"
    return "shallow_drawdown"


def _fixed_bucket(value: float, edges: Sequence[float]) -> str:
    if not np.isfinite(value):
        return "nan"
    low = 0.0
    for edge in edges:
        if value <= edge:
            return f"{low:.2f}-{edge:.2f}"
        low = edge
    return f">{edges[-1]:.2f}"


def _build_short_v2_bucket_rows(sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not sample_rows:
        return []
    total = float(len(sample_rows))
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in sample_rows:
        keys = [
            ("margin", _fixed_bucket(float(item["raw_margin"]), [0.05, 0.10, 0.15, 0.20])),
            ("uncertain", _fixed_bucket(float(item["raw_uncertain"]), [0.15, 0.25, 0.35, 0.50])),
            ("mu_sigma", _fixed_bucket(float(item["mu_sigma_abs"]), [0.5, 0.8, 1.2, 1.8])),
            ("vol_bucket", str(item["vol_bucket"])),
            ("trend_bucket", str(item["trend_bucket"])),
            ("drawdown_bucket", str(item["drawdown_bucket"])),
            ("regime_combo", str(item["regime_combo"])),
            ("regime_combo_drawdown", str(item["regime_combo_drawdown"])),
        ]
        for key in keys:
            groups.setdefault(key, []).append(item)

    rows: list[dict[str, Any]] = []
    for (bucket_type, bucket_name), items in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        count = len(items)
        raw_top1_share = float(np.mean([1.0 if it["raw_top1"] in {"up", "down"} else 0.0 for it in items]))
        cal_top1_share = float(np.mean([1.0 if it["cal_top1"] in {"up", "down"} else 0.0 for it in items]))
        flipped = [float(it["flipped"]) for it in items]
        flip_mask = [it for it in items if int(it["flipped"]) == 1]
        rows.append(
            {
                "bucket_type": bucket_type,
                "bucket_name": bucket_name,
                "count": int(count),
                "weight": float(count / total),
                "raw_top1_acc": float(np.mean([it["raw_correct"] for it in items])),
                "cal_top1_acc": float(np.mean([it["cal_correct"] for it in items])),
                "acc_delta": float(
                    np.mean([it["raw_correct"] for it in items]) - np.mean([it["cal_correct"] for it in items])
                ),
                "raw_logloss": float(np.mean([it["raw_logloss"] for it in items])),
                "cal_logloss": float(np.mean([it["cal_logloss"] for it in items])),
                "logloss_delta": float(
                    np.mean([it["cal_logloss"] for it in items]) - np.mean([it["raw_logloss"] for it in items])
                ),
                "raw_brier": float(np.mean([it["raw_brier"] for it in items])),
                "cal_brier": float(np.mean([it["cal_brier"] for it in items])),
                "brier_delta": float(
                    np.mean([it["cal_brier"] for it in items]) - np.mean([it["raw_brier"] for it in items])
                ),
                "flip_rate": float(np.mean(flipped)),
                "flip_improve_rate": float(
                    np.mean([it["flip_improved"] for it in flip_mask]) if flip_mask else 0.0
                ),
                "raw_top1_share": raw_top1_share,
                "cal_top1_share": cal_top1_share,
            }
        )
    return rows


def _build_short_v2_candidate_rows(
    *,
    bucket_rows: list[dict[str, Any]],
    min_count: int,
    acc_thresh: float,
    logloss_thresh: float,
    flip_thresh: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in bucket_rows:
        count_ok = int(row["count"]) >= int(min_count)
        acc_ok = float(row["acc_delta"]) >= float(acc_thresh)
        ll_ok = float(row["logloss_delta"]) >= float(logloss_thresh)
        flip_ok = float(row["flip_rate"]) >= float(flip_thresh)
        suggest_guard = int(count_ok and acc_ok and ll_ok and flip_ok)
        rows.append(
            {
                **row,
                "min_count": int(min_count),
                "acc_thresh": float(acc_thresh),
                "logloss_thresh": float(logloss_thresh),
                "flip_thresh": float(flip_thresh),
                "count_ok": int(count_ok),
                "acc_ok": int(acc_ok),
                "logloss_ok": int(ll_ok),
                "flip_ok": int(flip_ok),
                "suggest_guard": suggest_guard,
                "suggest_alpha_range": "0.75~0.85" if suggest_guard else "",
            }
        )
    rows.sort(
        key=lambda item: (
            int(item["suggest_guard"]),
            float(item["acc_delta"]),
            float(item["logloss_delta"]),
            float(item["flip_rate"]),
            int(item["count"]),
        ),
        reverse=True,
    )
    return rows


def _choose_test_size(n_total: int, min_train: int, min_valid: int) -> int:
    target = max(min_valid, int(round(n_total * 0.18)))
    max_test = n_total - (min_train + min_valid)
    if max_test < min_valid:
        return min_valid
    return min(target, max_test)


def _distribution(labels: np.ndarray) -> dict[str, float]:
    counts = np.bincount(labels.astype(int), minlength=3).astype(float)
    total = float(np.sum(counts))
    if total <= 1e-12:
        return {"up": 0.0, "flat": 0.0, "down": 0.0}
    return {
        "up": float(counts[LABEL_UP] / total),
        "flat": float(counts[LABEL_FLAT] / total),
        "down": float(counts[LABEL_DOWN] / total),
    }


def _distribution_by_regime_export(
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
        out[regime] = _distribution(label_values[regime_mask])
    return out


def _threshold_uncertain_relation_export(
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
        rows.append(
            {
                "bucket": float(idx),
                "low": low,
                "high": high,
                "count": float(count),
                "uncertain_rate": float(np.mean(y[in_bin] == LABEL_FLAT)),
            }
        )
    return rows


def _calibration_flip_summary_export(
    *,
    raw_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    labels: np.ndarray,
    eps: float,
) -> dict[str, float]:
    if raw_probs.shape[0] == 0:
        return {}
    raw_major = np.argmax(raw_probs, axis=1)
    cal_major = np.argmax(calibrated_probs, axis=1)
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


def _dist_str(dist: dict[str, float]) -> str:
    return (
        f"up={dist['up'] * 100:.1f}% "
        f"down={dist['down'] * 100:.1f}% "
        f"uncertain={dist['flat'] * 100:.1f}%"
    )


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.shape[0] == 0:
        return {
            "count": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "count": float(finite.shape[0]),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p25": float(np.quantile(finite, 0.25)),
        "p75": float(np.quantile(finite, 0.75)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _k_h(config: Any, horizon_days: int) -> float:
    if horizon_days <= 5:
        return config.label.k_5
    if horizon_days <= 20:
        return config.label.k_20
    return config.label.k_60


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


def _assign_regimes_from_cutoffs(
    *,
    vol_values: np.ndarray,
    cutoffs: dict[str, float],
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    q33 = float(cutoffs.get("q33", 0.0))
    q66 = float(cutoffs.get("q66", 0.0))
    q90 = float(cutoffs.get("q90", 0.0))
    out = np.full(vol_values.shape[0], "mid_vol", dtype=object)
    out[vol_values <= q33] = "low_vol"
    out[(vol_values > q33) & (vol_values <= q66)] = "mid_vol"
    out[vol_values > q66] = "high_vol"
    signal = np.abs(mu) / np.maximum(sigma, 1e-8)
    extreme = (vol_values >= q90) & (signal >= 1.20) & np.isfinite(signal)
    out[extreme] = "high_vol_extreme"
    return out


def _prob_dict(arr: np.ndarray) -> dict[str, float]:
    up = float(arr[LABEL_UP])
    uncertain = float(arr[LABEL_FLAT])
    down = float(arr[LABEL_DOWN])
    return {
        "up": up,
        "down": down,
        "uncertain": uncertain,
        "flat": uncertain,
    }


def _class_count(pred: np.ndarray) -> dict[str, int]:
    uncertain = int(np.sum(pred == LABEL_FLAT))
    return {
        "up": int(np.sum(pred == LABEL_UP)),
        "down": int(np.sum(pred == LABEL_DOWN)),
        "uncertain": uncertain,
        "flat": uncertain,
    }


def _display_class_count(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    up = 0
    down = 0
    uncertain = 0
    for row in rows:
        label = str(row.get(key, "uncertain"))
        if label == "up":
            up += 1
        elif label == "down":
            down += 1
        else:
            uncertain += 1
    return {
        "up": up,
        "down": down,
        "uncertain": uncertain,
    }


def _top_shrinkage_examples(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    sorted_rows = sorted(
        rows,
        key=lambda item: abs(item["final_up"] - item["cal_up"])
        + abs(item["final_uncertain"] - item["cal_uncertain"])
        + abs(item["final_down"] - item["cal_down"]),
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    for item in sorted_rows[:limit]:
        out.append(
            {
                "date": item["date"],
                "actual_label": item["actual_label"],
                "cal_probs": {
                    "up": item["cal_up"],
                    "down": item["cal_down"],
                    "uncertain": item["cal_uncertain"],
                },
                "final_probs": {
                    "up": item["final_up"],
                    "down": item["final_down"],
                    "uncertain": item["final_uncertain"],
                },
                "signal_strength": item["signal_strength"],
                "display_label": item["display_label"],
            }
        )
    return out


def _safe_diff(before: Any, after: Any) -> float | None:
    if before is None or after is None:
        return None
    return float(before) - float(after)


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


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "horizon",
        "train_distribution",
        "valid_distribution",
        "test_distribution",
        "raw_log_loss",
        "calibrated_log_loss",
        "final_log_loss",
        "raw_brier",
        "calibrated_brier",
        "final_brier",
        "current_threshold_low_p50",
        "current_threshold_high_p50",
        "current_lambda",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_sample_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "date",
        "horizon",
        "split",
        "actual_label",
        "actual_label_id",
        "raw_up",
        "raw_down",
        "raw_uncertain",
        "cal_up",
        "cal_down",
        "cal_uncertain",
        "post_shrink_up",
        "post_shrink_down",
        "post_shrink_uncertain",
        "post_uncertainty_boost_up",
        "post_uncertainty_boost_down",
        "post_uncertainty_boost_uncertain",
        "final_up",
        "final_down",
        "final_uncertain",
        "regime_shift_boost_delta",
        "high_vol_boost_delta",
        "total_uncertain_boost_delta",
        "uncertain_ceiling_applied",
        "state",
        "internal_state",
        "label",
        "headline_label",
        "display_label",
        "top1",
        "top2",
        "margin",
        "signal_strength",
        "warning_level",
        "warning_code",
        "warning_message",
        "q10",
        "q50",
        "q90",
        "sigma",
        "sigma_floor",
        "threshold_low",
        "threshold_high",
        "regime",
        "confidence",
        "predicted_class_raw",
        "predicted_class_cal",
        "predicted_class_final",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_ablation_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "horizon",
        "group",
        "variant",
        "ok",
        "note",
        "k_h",
        "lambda_h",
        "sigma_floor_h",
        "calibration_mode",
        "calibration_blend_h",
        "max_calibration_shift_h",
        "raw_log_loss",
        "calibrated_log_loss",
        "final_log_loss",
        "raw_brier",
        "calibrated_brier",
        "final_brier",
        "flip_rate_raw_to_cal",
        "label_up",
        "label_down",
        "label_uncertain",
        "final_up",
        "final_down",
        "final_uncertain",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_short_v2_samples_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "date",
        "symbol",
        "split",
        "actual_label",
        "raw_up",
        "raw_down",
        "raw_uncertain",
        "cal_up",
        "cal_down",
        "cal_uncertain",
        "raw_top1",
        "cal_top1",
        "raw_top1_prob",
        "cal_top1_prob",
        "raw_margin",
        "cal_margin",
        "mu",
        "sigma",
        "mu_sigma_abs",
        "regime",
        "vol_bucket",
        "trend_bucket",
        "drawdown_bucket",
        "regime_combo",
        "regime_combo_drawdown",
        "raw_correct",
        "cal_correct",
        "flipped",
        "flip_improved",
        "raw_logloss",
        "cal_logloss",
        "raw_brier",
        "cal_brier",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_short_v2_bucket_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "bucket_type",
        "bucket_name",
        "count",
        "weight",
        "raw_top1_acc",
        "cal_top1_acc",
        "acc_delta",
        "raw_logloss",
        "cal_logloss",
        "logloss_delta",
        "raw_brier",
        "cal_brier",
        "brier_delta",
        "flip_rate",
        "flip_improve_rate",
        "raw_top1_share",
        "cal_top1_share",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_short_v2_candidates_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "bucket_type",
        "bucket_name",
        "count",
        "weight",
        "raw_top1_acc",
        "cal_top1_acc",
        "acc_delta",
        "raw_logloss",
        "cal_logloss",
        "logloss_delta",
        "raw_brier",
        "cal_brier",
        "brier_delta",
        "flip_rate",
        "flip_improve_rate",
        "raw_top1_share",
        "cal_top1_share",
        "min_count",
        "acc_thresh",
        "logloss_thresh",
        "flip_thresh",
        "count_ok",
        "acc_ok",
        "logloss_ok",
        "flip_ok",
        "suggest_guard",
        "suggest_alpha_range",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_latest_index(
    *,
    path: Path,
    output_dir: Path,
    current_bundle: dict[str, Any],
) -> None:
    snapshots = _collect_snapshots(output_dir)
    latest_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for item in snapshots:
        key = (str(item.get("symbol", "")), str(item.get("period", "")))
        previous = latest_by_key.get(key)
        if previous is None:
            latest_by_key[key] = item
            continue
        if _snapshot_sort_key(item) > _snapshot_sort_key(previous):
            latest_by_key[key] = item

    latest_items = sorted(
        latest_by_key.values(),
        key=_snapshot_sort_key,
        reverse=True,
    )

    lines: list[str] = []
    lines.append("# Quantile Diagnostics Latest Index")
    lines.append("")
    lines.append(f"- Updated at: `{datetime.now().isoformat()}`")
    lines.append("")
    lines.append("## Current Run")
    lines.append("")
    lines.append(f"- symbol: `{current_bundle.get('symbol', '')}`")
    lines.append(f"- period: `{current_bundle.get('period', '')}`")
    lines.append(f"- generated_at: `{current_bundle.get('generated_at', '')}`")
    lines.append(f"- source: `{current_bundle.get('source', '')}`")
    lines.append(f"- last_trade_date: `{current_bundle.get('last_trade_date', '')}`")
    lines.append(
        f"- cache_fallback: `{'yes' if bool(current_bundle.get('cache_fallback')) else 'no'}`"
    )
    lines.append("")
    lines.append(f"- summary: {_as_link(current_bundle.get('summary'))}")
    lines.append(f"- detailed: {_as_link(current_bundle.get('detailed'))}")
    lines.append(f"- samples: {_as_link(current_bundle.get('samples'))}")
    lines.append(f"- ablation: {_as_link(current_bundle.get('ablation'))}")
    lines.append(f"- diagnostics json: {_as_link(current_bundle.get('json'))}")
    lines.append("")
    lines.append("## Latest By Symbol/Period")
    lines.append("")
    lines.append("| symbol | period | generated_at | source | last_trade_date | cache_fallback | summary | detailed | samples | ablation | json |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for item in latest_items:
        lines.append(
            "| {symbol} | {period} | {generated_at} | {source} | {last_trade_date} | {cache} | {summary} | {detailed} | {samples} | {ablation} | {json_file} |".format(
                symbol=item.get("symbol", ""),
                period=item.get("period", ""),
                generated_at=item.get("generated_at", ""),
                source=item.get("source", ""),
                last_trade_date=item.get("last_trade_date", ""),
                cache="yes" if bool(item.get("cache_fallback")) else "no",
                summary=_as_link(item.get("summary")),
                detailed=_as_link(item.get("detailed")),
                samples=_as_link(item.get("samples")),
                ablation=_as_link(item.get("ablation")),
                json_file=_as_link(item.get("json")),
            )
        )
    lines.append("")
    lines.append("## Recent Snapshots")
    lines.append("")
    recent = sorted(snapshots, key=_snapshot_sort_key, reverse=True)[:20]
    lines.append("| symbol | period | generated_at | detailed |")
    lines.append("|---|---|---|---|")
    for item in recent:
        lines.append(
            "| {symbol} | {period} | {generated_at} | {detailed} |".format(
                symbol=item.get("symbol", ""),
                period=item.get("period", ""),
                generated_at=item.get("generated_at", ""),
                detailed=_as_link(item.get("detailed")),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _collect_snapshots(output_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for json_path in sorted(output_dir.glob("*_quantile_diagnostics_*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        symbol = str(payload.get("symbol", "")).upper()
        period = str(payload.get("period", "")).upper()
        generated_at = str(payload.get("generated_at", ""))
        data_status = payload.get("data_status") if isinstance(payload.get("data_status"), dict) else {}
        source = str(data_status.get("source", ""))
        last_trade_date = str(data_status.get("last_trade_date", ""))
        cache_fallback = bool(data_status.get("used_cache_fallback", False))

        name = json_path.name
        token = "_quantile_diagnostics_"
        if token not in name:
            continue
        prefix, suffix_with_ext = name.split(token, 1)
        suffix = suffix_with_ext[:-5] if suffix_with_ext.endswith(".json") else suffix_with_ext
        summary = output_dir / f"{prefix}_quantile_summary_{suffix}.csv"
        detailed = output_dir / f"{prefix}_quantile_detailed_report_{suffix}.md"
        samples = output_dir / f"{prefix}_quantile_sample_predictions_{suffix}.csv"
        ablation = output_dir / f"{prefix}_quantile_ablation_{suffix}.csv"
        out.append(
            {
                "symbol": symbol,
                "period": period,
                "generated_at": generated_at,
                "source": source,
                "last_trade_date": last_trade_date,
                "cache_fallback": cache_fallback,
                "summary": summary,
                "detailed": detailed,
                "samples": samples,
                "ablation": ablation,
                "json": json_path,
            }
        )
    return out


def _snapshot_sort_key(item: dict[str, Any]) -> tuple[float, str]:
    generated_at = str(item.get("generated_at", ""))
    try:
        ts = datetime.fromisoformat(generated_at).timestamp()
    except Exception:
        ts = 0.0
    return (ts, generated_at)


def _as_link(value: Any) -> str:
    if value is None:
        return "-"
    path = Path(str(value))
    name = path.name
    if not name:
        return "-"
    return f"[{name}](./{name})"


def _write_detail_markdown(
    *,
    detail_path: Path,
    symbol: str,
    period: str,
    data_status: Any,
    used_cache_fallback: bool,
    diagnostics: dict[str, Any],
    summary_path: Path,
    sample_path: Path,
    short_v2_samples_path: Path,
    short_v2_bucket_path: Path,
    short_v2_candidates_path: Path,
    json_path: Path,
    ablation_path: Path,
) -> None:
    lines: list[str] = []
    lines.append(f"# Quantile 诊断导出报告 - {symbol}")
    lines.append("")
    lines.append(f"- 周期: `{period}`")
    lines.append(f"- 数据源: `{data_status.source}`")
    lines.append(f"- 最新交易日: `{data_status.last_trade_date.isoformat()}`")
    lines.append(f"- 数据新鲜度: `{'fresh' if data_status.is_fresh else 'stale'}`")
    lines.append(f"- 缓存回退: `{'yes' if used_cache_fallback else 'no'}`")
    lines.append("")
    lines.append("## 1. Summary Table")
    lines.append("")
    lines.append("| horizon | train dist | valid dist | test dist | raw ll | cal ll | final ll | raw brier | cal brier | final brier | thr low p50 | thr high p50 | lambda |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in diagnostics["summary_rows"]:
        lines.append(
            "| {h} | {td} | {vd} | {testd} | {rll:.4f} | {cll:.4f} | {fll:.4f} | {rb:.4f} | {cb:.4f} | {fb:.4f} | {tl:.4%} | {th:.4%} | {lam:.3f} |".format(
                h=row["horizon"],
                td=row["train_distribution"],
                vd=row["valid_distribution"],
                testd=row["test_distribution"],
                rll=row["raw_log_loss"],
                cll=row["calibrated_log_loss"],
                fll=row["final_log_loss"],
                rb=row["raw_brier"],
                cb=row["calibrated_brier"],
                fb=row["final_brier"],
                tl=row["current_threshold_low_p50"],
                th=row["current_threshold_high_p50"],
                lam=row["current_lambda"],
            )
        )
    lines.append("")
    lines.append("## 2. Detailed Report")
    lines.append("")
    for horizon in HORIZONS:
        detail = diagnostics["horizon_details"][horizon]
        lines.append(f"### 2.{HORIZONS.index(horizon) + 1} {horizon}")
        lines.append("")
        lines.append("#### 标签分布")
        lines.append(
            "- train: {0}".format(_dist_str(detail["label_distribution"]["train"]))
        )
        lines.append(
            "- valid: {0}".format(_dist_str(detail["label_distribution"]["valid"]))
        )
        lines.append(
            "- test: {0}".format(_dist_str(detail["label_distribution"]["test"]))
        )
        recent = detail.get("recent_label_distribution", {})
        if isinstance(recent, dict) and recent:
            recent_dist = recent.get("distribution", {})
            lines.append(
                "- recent_window[{rows}]: {dist}".format(
                    rows=int(recent.get("window_rows", 0)),
                    dist=_dist_str(recent_dist) if isinstance(recent_dist, dict) else "-",
                )
            )
        by_regime = detail.get("label_distribution_by_regime", {})
        if isinstance(by_regime, dict) and by_regime:
            lines.append("- by_regime(train): `{}`".format(by_regime.get("train", {})))
            lines.append("- by_regime(valid): `{}`".format(by_regime.get("valid", {})))
            lines.append("- by_regime(test): `{}`".format(by_regime.get("test", {})))
        threshold_bins = detail.get("threshold_uncertain_bins", [])
        if isinstance(threshold_bins, list) and threshold_bins:
            lines.append("- threshold_vs_uncertain: `{}`".format(threshold_bins))
        lines.append("")
        lines.append("#### 测试集链路指标 (raw vs calibrated vs final)")
        lines.append("")
        lines.append("| stage | log_loss | brier | accuracy | confusion_matrix |")
        lines.append("|---|---:|---:|---:|---|")
        for stage in ("raw", "calibrated", "final"):
            m = detail["metrics_test"][stage]
            lines.append(
                f"| {stage} | {m['log_loss']:.4f} | {m['brier_score']:.4f} | {m['accuracy']:.4f} | `{m['confusion_matrix']}` |"
            )
        lines.append("")
        lines.append("#### 测试集概率链路整体均值")
        chain = detail["chain_overall_test"]
        lines.append(
            f"- avg raw probs(up/down/uncertain): {chain['avg_raw_probs']['up']:.4f} / {chain['avg_raw_probs']['down']:.4f} / {chain['avg_raw_probs']['uncertain']:.4f}"
        )
        lines.append(
            f"- avg calibrated probs(up/down/uncertain): {chain['avg_calibrated_probs']['up']:.4f} / {chain['avg_calibrated_probs']['down']:.4f} / {chain['avg_calibrated_probs']['uncertain']:.4f}"
        )
        lines.append(
            f"- avg final probs(up/down/uncertain): {chain['avg_final_probs']['up']:.4f} / {chain['avg_final_probs']['down']:.4f} / {chain['avg_final_probs']['uncertain']:.4f}"
        )
        lines.append(f"- avg signal_strength: {chain['avg_signal_strength']:.4f}")
        lines.append(
            f"- avg boost delta(regime/high_vol/total): "
            f"{chain['avg_regime_shift_boost_delta']:.4f} / "
            f"{chain['avg_high_vol_boost_delta']:.4f} / "
            f"{chain['avg_total_uncertain_boost_delta']:.4f}"
        )
        lines.append(
            f"- avg uncertain(raw/final): {chain['avg_raw_uncertain']:.4f} / {chain['avg_final_uncertain']:.4f}"
        )
        lines.append(
            f"- uncertain ceiling applied ratio: {chain['uncertain_ceiling_applied_ratio']:.4f}"
        )
        lines.append(f"- predicted class count(final): `{chain['predicted_class_count_final']}`")
        flip_summary = detail.get("calibration_flip_summary", {})
        if isinstance(flip_summary, dict) and flip_summary:
            lines.append(
                "- calibration flip summary: "
                f"flip_rate={flip_summary.get('flip_rate', 0.0):.4f}, "
                f"flip_count={int(flip_summary.get('flip_count', 0))}, "
                f"raw_top1_mean={flip_summary.get('raw_top1_prob_mean', 0.0):.4f}, "
                f"cal_top1_mean={flip_summary.get('cal_top1_prob_mean', 0.0):.4f}, "
                f"flip_logloss_improve={flip_summary.get('flip_logloss_improve_rate', 0.0):.4f}, "
                f"flip_brier_improve={flip_summary.get('flip_brier_improve_rate', 0.0):.4f}, "
                f"strong_to_uncertain={flip_summary.get('strong_direction_to_uncertain_rate', 0.0):.4f}"
            )
        short_guard_stats = detail.get("short_guard_stats", {})
        if isinstance(short_guard_stats, dict) and short_guard_stats:
            guard_view = short_guard_stats.get("test", short_guard_stats)
            lines.append(
                "- short_guard_stats(test): "
                f"enabled={bool(short_guard_stats.get('enabled', False))}, "
                f"trigger_rate={float(guard_view.get('trigger_rate', 0.0)):.4f}, "
                f"trigger_count={int(guard_view.get('trigger_count', 0))}, "
                f"triggered_flip_rate={float(guard_view.get('triggered_flip_rate', 0.0)):.4f}, "
                f"triggered_raw_top1_acc={float(guard_view.get('triggered_raw_top1_acc', 0.0)):.4f}, "
                f"triggered_guard_top1_acc={float(guard_view.get('triggered_guard_top1_acc', 0.0)):.4f}, "
                f"triggered_logloss={float(guard_view.get('triggered_logloss', 0.0)):.4f}, "
                f"triggered_brier={float(guard_view.get('triggered_brier', 0.0)):.4f}, "
                f"triggered_to_uncertain_rate={float(guard_view.get('triggered_to_uncertain_rate', 0.0)):.4f}"
            )
        lines.append("")
        lines.append("#### 最近测试样本（链路明细）")
        lines.append("")
        lines.append("| date | actual | raw(up/down/uncertain) | cal(up/down/uncertain) | post_shrink(up/down/uncertain) | post_boost(up/down/uncertain) | final(up/down/uncertain) | boost_delta(regime/high/total) | ceiling | state | label | headline | display | top1/top2/margin | signal_strength | warning | pred_final |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---:|---|---|")
        for row in detail["recent_test_rows"]:
            lines.append(
                "| {date} | {actual} | {ru:.3f}/{rd:.3f}/{runc:.3f} | {cu:.3f}/{cd:.3f}/{cunc:.3f} | {su:.3f}/{sd:.3f}/{sunc:.3f} | {bu:.3f}/{bd:.3f}/{bunc:.3f} | {fu:.3f}/{fd:.3f}/{func:.3f} | {d1:.3f}/{d2:.3f}/{d3:.3f} | {ceiling} | {state} | {label} | {headline} | {display} | {top1:.3f}/{top2:.3f}/{margin:.3f} | {signal:.3f} | {warning} | {pred} |".format(
                    date=row["date"],
                    actual=row["actual_label"],
                    ru=row["raw_up"],
                    rd=row["raw_down"],
                    runc=row["raw_uncertain"],
                    cu=row["cal_up"],
                    cd=row["cal_down"],
                    cunc=row["cal_uncertain"],
                    su=row["post_shrink_up"],
                    sd=row["post_shrink_down"],
                    sunc=row["post_shrink_uncertain"],
                    bu=row["post_uncertainty_boost_up"],
                    bd=row["post_uncertainty_boost_down"],
                    bunc=row["post_uncertainty_boost_uncertain"],
                    fu=row["final_up"],
                    fd=row["final_down"],
                    func=row["final_uncertain"],
                    d1=row["regime_shift_boost_delta"],
                    d2=row["high_vol_boost_delta"],
                    d3=row["total_uncertain_boost_delta"],
                    ceiling="yes" if row["uncertain_ceiling_applied"] else "no",
                    state=row["internal_state"],
                    label=row["label"],
                    headline=row["headline_label"],
                    display=row["display_label"],
                    top1=row["top1"],
                    top2=row["top2"],
                    margin=row["margin"],
                    signal=row["signal_strength"],
                    warning=row["warning_code"] or "-",
                    pred=row["predicted_class_final"],
                )
            )
        lines.append("")
        lines.append("#### 分布参数（test）")
        params = detail["distribution_params"]["test"]
        lines.append(f"- q10: `{params['q10']}`")
        lines.append(f"- q50: `{params['q50']}`")
        lines.append(f"- q90: `{params['q90']}`")
        lines.append(f"- implied sigma: `{params['implied_sigma']}`")
        lines.append(f"- threshold_low: `{params['threshold_low']}`")
        lines.append(f"- threshold_high: `{params['threshold_high']}`")
        lines.append(f"- q90-q10: `{params['q90_minus_q10']}`")
        lines.append(
            f"- sigma_floor: `{params['sigma_floor']}`, floor hits: `{params['sigma_floor_hits']}`"
        )
        lines.append("")
        lines.append("#### Reliability (test, 0.1 bins)")
        for stage in ("raw", "calibrated", "final"):
            m = detail["metrics_test"][stage]
            lines.append(f"- {stage} prob_up bins: `{m['calibration_up']}`")
            lines.append(f"- {stage} prob_down bins: `{m['calibration_down']}`")
        lines.append("")

    lines.append("## 3. Long Horizon 重点诊断")
    long_focus = diagnostics["horizon_details"]["long"]["long_focus"]
    lines.append("")
    lines.append(f"- threshold 公式: `{long_focus['threshold_formula']}`")
    lines.append(f"- threshold 参数: `{long_focus['threshold_params']}`")
    lines.append(f"- threshold stats(test): `{long_focus['threshold_stats_test']}`")
    lines.append(f"- sigma 分布摘要: `{long_focus['sigma_distribution']}`")
    lines.append("- shrinkage 明细（test samples）:")
    for item in long_focus["shrinkage_detail"]["examples_test"]:
        lines.append(f"  - `{item}`")
    lines.append("")

    lines.append("## 4. Short Horizon 重点诊断")
    short_extremes = diagnostics["horizon_details"]["short"]["extreme_samples"]
    lines.append("")
    lines.append(
        f"- raw 极端样本数 (max(raw_probs)>0.85): `{len(short_extremes)}`"
    )
    lines.append("| date | split | actual | raw(up/down/uncertain) | cal(up/down/uncertain) | final(up/down/uncertain) | state | display | signal_strength | q10/q50/q90 | sigma | threshold |")
    lines.append("|---|---|---|---|---|---|---|---|---:|---|---:|---|")
    for row in short_extremes:
        lines.append(
            "| {date} | {split} | {actual} | {ru:.3f}/{rd:.3f}/{runc:.3f} | {cu:.3f}/{cd:.3f}/{cunc:.3f} | {fu:.3f}/{fd:.3f}/{func:.3f} | {state} | {display} | {signal:.3f} | {q10:.3%}/{q50:.3%}/{q90:.3%} | {sigma:.3%} | [{tl:.3%},{th:.3%}] |".format(
                date=row["date"],
                split=row["split"],
                actual=row["actual_label"],
                ru=row["raw_up"],
                rd=row["raw_down"],
                runc=row["raw_uncertain"],
                cu=row["cal_up"],
                cd=row["cal_down"],
                cunc=row["cal_uncertain"],
                fu=row["final_up"],
                fd=row["final_down"],
                func=row["final_uncertain"],
                state=row["internal_state"],
                display=row["display_label"],
                signal=row["signal_strength"],
                q10=row["q10"],
                q50=row["q50"],
                q90=row["q90"],
                sigma=row["sigma"],
                tl=row["threshold_low"],
                th=row["threshold_high"],
            )
        )
    lines.append("")

    lines.append("## 5. Ablation Summary")
    abl = diagnostics["ablation"]
    lines.append("")
    lines.append(f"- pipeline 改善拆解(by horizon): `{abl['pipeline_improvement_by_horizon']}`")
    lines.append(f"- 推荐方案(by horizon): `{abl['recommended_by_horizon']}`")
    lines.append(f"- lambda 灵敏度(long): `{abl['lambda_sensitivity']}`")
    groups = abl.get("groups", {})
    for group in sorted(groups.keys()):
        lines.append(f"- {group}: `{groups[group]}`")
    lines.append("")
    lines.append("## 6. 导出文件")
    lines.append("")
    lines.append(f"- summary csv: `{summary_path}`")
    lines.append(f"- sample predictions csv: `{sample_path}`")
    lines.append(f"- short v2 samples csv: `{short_v2_samples_path}`")
    lines.append(f"- short v2 bucket report csv: `{short_v2_bucket_path}`")
    lines.append(f"- short v2 candidates csv: `{short_v2_candidates_path}`")
    lines.append(f"- diagnostics json: `{json_path}`")
    lines.append(f"- ablation csv: `{ablation_path}`")
    lines.append("")

    detail_path.write_text("\n".join(lines), encoding="utf-8")


def _period_to_trading_days(period: str) -> int:
    value = period.strip().upper()
    mapping = {
        "1M": 21,
        "3M": 63,
        "6M": 126,
        "1Y": 252,
        "3Y": 756,
        "5Y": 1260,
    }
    if value in mapping:
        return mapping[value]
    if value.endswith("D") and value[:-1].isdigit():
        return max(int(value[:-1]), 5)
    raise ValueError(f"不支持的 period: {period}")


if __name__ == "__main__":
    raise SystemExit(main())
