#!/usr/bin/env python3
"""Export structured diagnostics for quantile probability model."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any
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
    "strong_up": "上行",
    "mild_up": "偏上",
    "uncertain": "不确定",
    "mild_down": "偏下",
    "strong_down": "下行",
}
WARNING_MESSAGE_MAP = {
    "direction_cap_up": "方向概率触发上限压缩，已降低极端化输出。",
    "direction_cap_down": "方向概率触发下限压缩，已降低极端化输出。",
    "regime_shift_shrink": "检测到市场状态漂移，已自动收缩置信度。",
    "high_vol_uncertain_boost": "当前高波动环境，不确定权重已提升。",
    "cross_horizon_conflict_shrink": "中长期信号冲突，已执行跨周期降置信处理。",
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

    now_tag = datetime.now(timezone).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"{args.symbol.lower()}_quantile_summary_{now_tag}.csv"
    detail_path = output_dir / f"{args.symbol.lower()}_quantile_detailed_report_{now_tag}.md"
    sample_path = output_dir / f"{args.symbol.lower()}_quantile_sample_predictions_{now_tag}.csv"
    json_path = output_dir / f"{args.symbol.lower()}_quantile_diagnostics_{now_tag}.json"
    ablation_path = output_dir / f"{args.symbol.lower()}_quantile_ablation_{now_tag}.csv"

    _write_summary_csv(summary_path, diagnostics["summary_rows"])
    _write_sample_csv(sample_path, diagnostics["sample_rows"])
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
    oof_q10 = np.full(x_dev.shape[0], np.nan, dtype=float)
    oof_q50 = np.full(x_dev.shape[0], np.nan, dtype=float)
    oof_q90 = np.full(x_dev.shape[0], np.nan, dtype=float)
    oof_sigma = np.full(x_dev.shape[0], np.nan, dtype=float)
    oof_mu = np.full(x_dev.shape[0], np.nan, dtype=float)
    oof_seen = np.zeros(x_dev.shape[0], dtype=bool)

    for train_idx, valid_idx in expanding_time_series_splits(x_dev.shape[0], split_cfg):
        q10_model = QuantileLinearRegressor(0.10, config.quantile).fit(x_dev[train_idx], y_reg_dev[train_idx])
        q50_model = QuantileLinearRegressor(0.50, config.quantile).fit(x_dev[train_idx], y_reg_dev[train_idx])
        q90_model = QuantileLinearRegressor(0.90, config.quantile).fit(x_dev[train_idx], y_reg_dev[train_idx])

        q10_pred = q10_model.predict(x_dev[valid_idx])
        q50_pred = q50_model.predict(x_dev[valid_idx])
        q90_pred = q90_model.predict(x_dev[valid_idx])
        down_raw, flat_raw, up_raw, mu, sigma = quantiles_to_raw_probabilities(
            q10=q10_pred,
            q50=q50_pred,
            q90=q90_pred,
            threshold_up=threshold_up_dev[valid_idx],
            threshold_down=threshold_down_dev[valid_idx],
            sigma_floor=mapper_cfg.sigma_floor(horizon_days),
            eps=mapper_cfg.eps,
        )
        oof_raw[valid_idx] = np.column_stack([down_raw, flat_raw, up_raw])
        oof_q10[valid_idx] = q10_pred
        oof_q50[valid_idx] = q50_pred
        oof_q90[valid_idx] = q90_pred
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
            calibrated_probs=valid_cal,
            base_probs=base_probs,
            mu=oof_mu[oof_mask],
            sigma=oof_sigma[oof_mask],
            lambda_h=lambda_h,
            regimes=regime_valid,
            config=mapper_cfg,
        )

    q10_full = QuantileLinearRegressor(0.10, config.quantile).fit(x_dev, y_reg_dev)
    q50_full = QuantileLinearRegressor(0.50, config.quantile).fit(x_dev, y_reg_dev)
    q90_full = QuantileLinearRegressor(0.90, config.quantile).fit(x_dev, y_reg_dev)
    q10_test = q10_full.predict(x_test)
    q50_test = q50_full.predict(x_test)
    q90_test = q90_full.predict(x_test)
    down_raw_test, flat_raw_test, up_raw_test, mu_test, sigma_test = quantiles_to_raw_probabilities(
        q10=q10_test,
        q50=q50_test,
        q90=q90_test,
        threshold_up=threshold_up_test,
        threshold_down=threshold_down_test,
        sigma_floor=mapper_cfg.sigma_floor(horizon_days),
        eps=mapper_cfg.eps,
    )
    test_raw = np.column_stack([down_raw_test, flat_raw_test, up_raw_test])
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
            calibrated_probs=test_cal,
            base_probs=base_probs,
            mu=mu_test,
            sigma=sigma_test,
            lambda_h=lambda_h,
            regimes=regime_test,
            config=mapper_cfg,
        )

    metrics_valid = _stage_metrics(valid_raw, valid_cal, valid_final, y_cls_dev[oof_mask], regime_valid)
    metrics_test = _stage_metrics(test_raw, test_cal, test_final, y_cls_test, regime_test)
    train_dist = _distribution(y_cls_dev[train_core_mask])
    valid_dist = _distribution(y_cls_dev[oof_mask])
    test_dist = _distribution(y_cls_test)
    regime_shift_score = (
        abs(train_dist["up"] - valid_dist["up"])
        + abs(train_dist["down"] - valid_dist["down"])
    )

    valid_rows = _build_rows(
        dates=[feature_matrix.dates[idx] for idx in dev_global[oof_mask]],
        horizon=horizon,
        split_name="valid",
        actual=y_cls_dev[oof_mask],
        raw_probs=valid_raw,
        cal_probs=valid_cal,
        final_probs=valid_final,
        regimes=regime_valid,
        mapper_cfg=mapper_cfg,
        regime_shift_score=regime_shift_score,
        q10=oof_q10[oof_mask],
        q50=oof_q50[oof_mask],
        q90=oof_q90[oof_mask],
        sigma=oof_sigma[oof_mask],
        sigma_floor=mapper_cfg.sigma_floor(horizon_days),
        threshold_low=threshold_down_dev[oof_mask],
        threshold_high=threshold_up_dev[oof_mask],
    )
    test_rows = _build_rows(
        dates=[feature_matrix.dates[idx] for idx in test_global],
        horizon=horizon,
        split_name="test",
        actual=y_cls_test,
        raw_probs=test_raw,
        cal_probs=test_cal,
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

    qgap_valid = oof_q90[oof_mask] - oof_q10[oof_mask]
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
        "metrics_valid": metrics_valid,
        "metrics_test": metrics_test,
        "chain_overall_test": {
            "avg_raw_probs": _prob_dict(np.mean(test_raw, axis=0)),
            "avg_calibrated_probs": _prob_dict(np.mean(test_cal, axis=0)),
            "avg_final_probs": _prob_dict(np.mean(test_final, axis=0)),
            "avg_signal_strength": float(np.mean([float(item["signal_strength"]) for item in test_rows])),
            "predicted_class_count_final": _display_class_count(test_rows, "predicted_class_final"),
        },
        "recent_test_rows": recent,
        "sample_rows": valid_rows + test_rows,
        "distribution_params": {
            "valid": {
                "q10": _summary_stats(oof_q10[oof_mask]),
                "q50": _summary_stats(oof_q50[oof_mask]),
                "q90": _summary_stats(oof_q90[oof_mask]),
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
                "final_up": float(display["final_up"]),
                "final_down": float(display["final_down"]),
                "final_uncertain": float(display["final_uncertain"]),
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
    regime: str,
    mapper_cfg: ProbabilityMapperConfig,
    regime_shift_score: float,
) -> dict[str, Any]:
    raw_up, raw_down, raw_uncertain = _to_display_probs(raw_probs)
    cal_up, cal_down, cal_uncertain = _to_display_probs(cal_probs)
    final_up, final_down, final_uncertain = _to_display_probs(final_probs)

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

    warning_codes: list[str] = []
    cap = float(np.clip(mapper_cfg.display_prob_cap, 0.50, 0.99))
    if final_up_cond >= cap - 1e-6:
        warning_codes.append("direction_cap_up")
    elif final_up_cond <= (1.0 - cap) + 1e-6:
        warning_codes.append("direction_cap_down")
    if regime_shift_score >= mapper_cfg.display_regime_shift_threshold and max(final_up_cond, 1.0 - final_up_cond) >= 0.70:
        warning_codes.append("regime_shift_shrink")
    if regime in {"high_vol", "high_vol_extreme"} and abs(final_up_cond - 0.5) < 0.18:
        warning_codes.append("high_vol_uncertain_boost")

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
        "raw_up": raw_up,
        "raw_down": raw_down,
        "raw_uncertain": raw_uncertain,
        "cal_up": cal_up,
        "cal_down": cal_down,
        "cal_uncertain": cal_uncertain,
        "final_up": final_up,
        "final_down": final_down,
        "final_uncertain": final_uncertain,
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
        "predicted_class_raw": _predicted_class(raw_up, raw_down, raw_uncertain),
        "predicted_class_cal": _predicted_class(cal_up, cal_down, cal_uncertain),
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


def _predicted_class(prob_up: float, prob_down: float, prob_uncertain: float) -> str:
    if prob_up >= prob_down and prob_up >= prob_uncertain:
        return "up"
    if prob_down >= prob_up and prob_down >= prob_uncertain:
        return "down"
    return "uncertain"


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
    if any(token in {"direction_cap_up", "direction_cap_down", "cross_horizon_conflict_shrink"} for token in tokens):
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
        "final_up",
        "final_down",
        "final_uncertain",
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
        lines.append(f"- predicted class count(final): `{chain['predicted_class_count_final']}`")
        lines.append("")
        lines.append("#### 最近测试样本（链路明细）")
        lines.append("")
        lines.append("| date | actual | raw(up/down/uncertain) | cal(up/down/uncertain) | final(up/down/uncertain) | state | label | headline | display | top1/top2/margin | signal_strength | warning | pred_final |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---:|---|---|")
        for row in detail["recent_test_rows"]:
            lines.append(
                "| {date} | {actual} | {ru:.3f}/{rd:.3f}/{runc:.3f} | {cu:.3f}/{cd:.3f}/{cunc:.3f} | {fu:.3f}/{fd:.3f}/{func:.3f} | {state} | {label} | {headline} | {display} | {top1:.3f}/{top2:.3f}/{margin:.3f} | {signal:.3f} | {warning} | {pred} |".format(
                    date=row["date"],
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
