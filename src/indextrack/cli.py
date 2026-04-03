"""CLI entrypoint for IndexTrack."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Sequence
from zoneinfo import ZoneInfo

from indextrack.app.orchestrator import AnalysisOrchestrator, DataFetchOutcome, DataUnavailableError
from indextrack.domain.features import FeatureEngine, FeatureError
from indextrack.domain.report import ReportGenerator, ReportOutput
from indextrack.domain.scenario import (
    MomentumScorer,
    ScenarioProbs,
    ScenarioEngine,
    VolatilityScorer,
    build_horizon_weights,
)
from indextrack.domain.trend import TrendScorer
from indextrack.infra.config import RuntimeConfig, load_runtime_config
from indextrack.infra.freshness import FreshnessGuard
from indextrack.infra.logging import setup_logging
from indextrack.infra.providers.primary import YahooFinancePrimaryProvider
from indextrack.infra.providers.router import ProviderRouter
from indextrack.infra.providers.secondary import YahooFinanceSecondaryProvider
from indextrack.infra.repository.sqlite_repo import SQLiteRepository
from indextrack.ui.page import render_dashboard_page, run_ui_server
from indextrack.ui.view_model import build_index_card_view_model

INDEX_MAP = {
    "SP500": ["SP500"],
    "NASDAQ": ["NASDAQ"],
    "BOTH": ["SP500", "NASDAQ"],
}
UI_PERIODS = {"1M", "3M", "6M", "1Y"}


@dataclass(frozen=True)
class _ComputedAnalysis:
    """Computed scenario + report bundle for one symbol."""

    scenarios: list[ScenarioProbs]
    report: ReportOutput
    horizon_outputs: dict[str, Any] | None = None
    backtest_summary: dict[str, object] | None = None


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line workflow."""
    try:
        config = load_runtime_config()
    except Exception as exc:  # noqa: BLE001
        message, suggestions = _friendly_error(exc)
        print(f"错误: {message}")
        if suggestions:
            print("建议:")
            for item in suggestions:
                print(f"- {item}")
        return 1

    parser = _build_parser(
        default_index=config.default_index,
        default_period=config.default_period,
        default_model=config.model_mode,
    )
    args = parser.parse_args(argv)
    setup_logging(level=args.log_level, log_path=args.log_path)
    logger = logging.getLogger("indextrack.cli")

    timezone_name = args.timezone or config.timezone
    try:
        timezone = ZoneInfo(timezone_name)
    except Exception:  # noqa: BLE001
        message, suggestions = _friendly_error(ValueError(f"不支持的时区: {timezone_name}"))
        print(f"错误: {message}")
        if suggestions:
            print("建议:")
            for item in suggestions:
                print(f"- {item}")
        logger.error("invalid_timezone timezone=%s", timezone_name)
        return 1

    symbols = INDEX_MAP[(args.index or config.default_index).upper()]
    try:
        trading_days = _period_to_trading_days(args.period or config.default_period)
    except ValueError as exc:
        message, suggestions = _friendly_error(exc)
        print(f"错误: {message}")
        if suggestions:
            print("建议:")
            for item in suggestions:
                print(f"- {item}")
        logger.error("invalid_period period=%s error=%s", args.period, exc)
        return 1
    lookback_trading_days = max(trading_days, 260)
    lookback_calendar_days = int(lookback_trading_days * 1.7)
    end = datetime.now(timezone).date()
    start = end - timedelta(days=lookback_calendar_days)

    provider_router = ProviderRouter(
        primary=YahooFinancePrimaryProvider(request_timeout_sec=config.request_timeout_sec),
        secondary=YahooFinanceSecondaryProvider(request_timeout_sec=config.request_timeout_sec),
        retries=1,
    )
    repository = SQLiteRepository(db_path=args.db_path)
    freshness_guard = FreshnessGuard(timezone_name=timezone_name)
    orchestrator = AnalysisOrchestrator(
        provider_router=provider_router,
        repository=repository,
        freshness_guard=freshness_guard,
    )

    scenario_engine = ScenarioEngine(
        weights=build_horizon_weights(
            short=config.scenario_weights.short,
            mid=config.scenario_weights.mid,
            long=config.scenario_weights.long,
        )
    )
    active_model_mode = args.model
    probability_model_config: Any | None = None
    if active_model_mode == "quantile":
        try:
            probability_model_config = _build_probability_model_config(config)
        except FeatureError as exc:
            logger.warning(
                "quantile_dependency_missing_fallback_to_legacy reason=%s",
                exc,
            )
            print(f"警告: {exc}")
            print("已自动切换到 --model legacy 继续运行。")
            active_model_mode = "legacy"
    feature_engine = FeatureEngine()
    trend_scorer = TrendScorer()
    momentum_scorer = MomentumScorer()
    volatility_scorer = VolatilityScorer()
    report_generator = ReportGenerator()

    if args.ui:
        return _run_ui_mode(
            logger=logger,
            host=args.ui_host,
            port=args.ui_port,
            timezone=timezone,
            orchestrator=orchestrator,
            feature_engine=feature_engine,
            trend_scorer=trend_scorer,
            momentum_scorer=momentum_scorer,
            volatility_scorer=volatility_scorer,
            scenario_engine=scenario_engine,
            report_generator=report_generator,
            model_mode=active_model_mode,
            probability_model_config=probability_model_config,
        )

    had_failure = False
    logger.info(
        "analysis_start symbols=%s period=%s timezone=%s db_path=%s requested_model=%s active_model=%s",
        symbols,
        args.period,
        timezone_name,
        args.db_path,
        args.model,
        active_model_mode,
    )
    for symbol in symbols:
        print("=" * 72)
        print(f"指数: {symbol}")
        logger.info("symbol_analysis_start symbol=%s start=%s end=%s", symbol, start, end)
        try:
            outcome = orchestrator.fetch_market_data(
                symbol=symbol,
                start=start,
                end=end,
                lookback_days=lookback_calendar_days,
            )
            _render_symbol_report(
                outcome=outcome,
                feature_engine=feature_engine,
                trend_scorer=trend_scorer,
                momentum_scorer=momentum_scorer,
                volatility_scorer=volatility_scorer,
                scenario_engine=scenario_engine,
                report_generator=report_generator,
                model_mode=active_model_mode,
                probability_model_config=probability_model_config,
                prob_debug=args.prob_debug,
                run_prob_ablation=args.prob_ablation,
            )
            logger.info(
                "symbol_analysis_success symbol=%s source=%s is_fresh=%s cache_fallback=%s",
                symbol,
                outcome.source,
                outcome.data_status.is_fresh,
                outcome.used_cache_fallback,
            )
        except (DataUnavailableError, FeatureError, ValueError) as exc:
            had_failure = True
            message, suggestions = _friendly_error(exc)
            print(f"错误: {message}")
            if suggestions:
                print("建议:")
                for item in suggestions:
                    print(f"- {item}")
            logger.exception("symbol_analysis_failed symbol=%s error=%s", symbol, exc)

    logger.info("analysis_end had_failure=%s", had_failure)
    return 1 if had_failure else 0


def _render_symbol_report(
    *,
    outcome: DataFetchOutcome,
    feature_engine: FeatureEngine,
    trend_scorer: TrendScorer,
    momentum_scorer: MomentumScorer,
    volatility_scorer: VolatilityScorer,
    scenario_engine: ScenarioEngine,
    report_generator: ReportGenerator,
    model_mode: str,
    probability_model_config: Any | None,
    prob_debug: bool,
    run_prob_ablation: bool,
) -> None:
    computed = _compute_analysis(
        outcome=outcome,
        feature_engine=feature_engine,
        trend_scorer=trend_scorer,
        momentum_scorer=momentum_scorer,
        volatility_scorer=volatility_scorer,
        scenario_engine=scenario_engine,
        report_generator=report_generator,
        model_mode=model_mode,
        probability_model_config=probability_model_config,
    )

    print(
        "数据状态: "
        f"source={outcome.data_status.source}, "
        f"last_trade_date={outcome.data_status.last_trade_date.isoformat()}, "
        f"is_fresh={'yes' if outcome.data_status.is_fresh else 'no'}"
    )
    if outcome.warning:
        print(f"数据告警: {outcome.warning}")
    for item in computed.scenarios:
        print(
            f"{_horizon_label(item.horizon)}: "
            f"上涨={item.uptrend_pct:.2f}% "
            f"下跌={item.downtrend_pct:.2f}% "
            f"不确定={item.sideways_pct:.2f}%"
        )
    if computed.horizon_outputs:
        print("概率细节:")
        for horizon in ("short", "mid", "long"):
            detail = computed.horizon_outputs[horizon]
            print(
                f"- {_horizon_label(horizon)} "
                f"raw(上/下/不确定)="
                f"{detail.display_prob_up_raw * 100:.2f}%/"
                f"{detail.display_prob_down_raw * 100:.2f}%/"
                f"{detail.display_prob_uncertain * 100:.2f}% "
                f"cal(上/下/不确定)="
                f"{detail.display_prob_up_calibrated * 100:.2f}%/"
                f"{detail.display_prob_down_calibrated * 100:.2f}%/"
                f"{detail.display_prob_uncertain * 100:.2f}% "
                f"final(上/下/不确定)="
                f"{detail.display_prob_up * 100:.2f}%/"
                f"{detail.display_prob_down * 100:.2f}%/"
                f"{detail.display_prob_uncertain * 100:.2f}% "
                f"internal_state={detail.internal_state} "
                f"label={detail.label} "
                f"headline={detail.headline_label} "
                f"display_label={detail.display_label} "
                f"top1={detail.top1_prob * 100:.2f}% "
                f"top2={detail.top2_prob * 100:.2f}% "
                f"margin={detail.margin * 100:.2f}% "
                f"阈值=[{detail.threshold_down * 100:.2f}%, {detail.threshold_up * 100:.2f}%] "
                f"regime={detail.regime} signal_strength={detail.signal_strength * 100:.1f}%"
            )
            if detail.warning_code:
                print(
                    f"  warning_level={detail.warning_level} "
                    f"warning_code={detail.warning_code} "
                    f"warning_message={detail.warning_message}"
                )
    if prob_debug and computed.backtest_summary:
        _print_probability_debug_info(computed.backtest_summary)
    if run_prob_ablation and outcome.candles and model_mode == "quantile":
        try:
            _run_probability_ablation(candles=outcome.candles, base_config=probability_model_config)
        except FeatureError as exc:
            print(f"概率消融实验失败: {exc}")
    print("结论:")
    print(computed.report.summary_zh)
    print("风险提示:")
    for note in computed.report.risk_notes:
        print(f"- {note}")


def _run_ui_mode(
    *,
    logger: logging.Logger,
    host: str,
    port: int,
    timezone: ZoneInfo,
    orchestrator: AnalysisOrchestrator,
    feature_engine: FeatureEngine,
    trend_scorer: TrendScorer,
    momentum_scorer: MomentumScorer,
    volatility_scorer: VolatilityScorer,
    scenario_engine: ScenarioEngine,
    report_generator: ReportGenerator,
    model_mode: str,
    probability_model_config: Any | None,
) -> int:
    logger.info("ui_server_start host=%s port=%s", host, port)

    def build_dashboard(period: str, selected_model_mode: str) -> str:
        normalized_period = _normalize_ui_period(period)
        normalized_model_mode = selected_model_mode.strip().lower()
        if normalized_model_mode not in {"quantile", "legacy"}:
            normalized_model_mode = model_mode
        effective_model_mode = normalized_model_mode
        ui_notice: str | None = None
        if effective_model_mode == "quantile" and probability_model_config is None:
            effective_model_mode = "legacy"
            ui_notice = "quantile 依赖缺失，当前请求已自动切换为 legacy。"
        lookback_days = _ui_lookback_calendar_days(
            period=normalized_period,
            model_mode=effective_model_mode,
            probability_model_config=probability_model_config,
        )
        end = datetime.now(timezone).date()
        start = end - timedelta(days=lookback_days)
        logger.info(
            "ui_dashboard_build period=%s requested_model=%s active_model=%s lookback_days=%s",
            normalized_period,
            normalized_model_mode,
            effective_model_mode,
            lookback_days,
        )

        cards = []
        for symbol in INDEX_MAP["BOTH"]:
            outcome = orchestrator.fetch_market_data(
                symbol=symbol,
                start=start,
                end=end,
                lookback_days=lookback_days,
            )
            try:
                computed = _compute_analysis(
                    outcome=outcome,
                    feature_engine=feature_engine,
                    trend_scorer=trend_scorer,
                    momentum_scorer=momentum_scorer,
                    volatility_scorer=volatility_scorer,
                    scenario_engine=scenario_engine,
                    report_generator=report_generator,
                    model_mode=effective_model_mode,
                    probability_model_config=probability_model_config,
                )
            except FeatureError as exc:
                if effective_model_mode != "quantile":
                    raise
                logger.warning("ui_quantile_failed_fallback_to_legacy symbol=%s error=%s", symbol, exc)
                effective_model_mode = "legacy"
                ui_notice = f"quantile 运行失败，已自动切换为 legacy：{exc}"
                computed = _compute_analysis(
                    outcome=outcome,
                    feature_engine=feature_engine,
                    trend_scorer=trend_scorer,
                    momentum_scorer=momentum_scorer,
                    volatility_scorer=volatility_scorer,
                    scenario_engine=scenario_engine,
                    report_generator=report_generator,
                    model_mode="legacy",
                    probability_model_config=None,
                )
            card = build_index_card_view_model(
                symbol=symbol,
                period=normalized_period,
                model_mode=effective_model_mode,
                candles=outcome.candles,
                summary_zh=computed.report.summary_zh,
                scenarios=computed.scenarios,
                data_status=outcome.data_status,
                used_cache_fallback=outcome.used_cache_fallback,
                horizon_outputs=computed.horizon_outputs,
            )
            cards.append(card)

        generated_at = datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S")
        return render_dashboard_page(
            cards=cards,
            period=normalized_period,
            generated_at=generated_at,
            model_mode=effective_model_mode,
            ui_notice=ui_notice,
        )

    print(f"UI 服务已启动: http://{host}:{port}")
    print("按 Ctrl+C 停止服务。")
    try:
        run_ui_server(
            host=host,
            port=port,
            build_dashboard=build_dashboard,
            default_model=model_mode,
        )
    except KeyboardInterrupt:
        print("\nUI 服务已停止。")
        logger.info("ui_server_stopped")
    return 0


def _compute_analysis(
    *,
    outcome: DataFetchOutcome,
    feature_engine: FeatureEngine,
    trend_scorer: TrendScorer,
    momentum_scorer: MomentumScorer,
    volatility_scorer: VolatilityScorer,
    scenario_engine: ScenarioEngine,
    report_generator: ReportGenerator,
    model_mode: str,
    probability_model_config: Any | None,
) -> _ComputedAnalysis:
    if model_mode == "quantile":
        if probability_model_config is None:
            raise FeatureError("概率模型配置不可用，请安装 quantile 依赖后重试。")
        return _compute_analysis_quantile(
            outcome=outcome,
            report_generator=report_generator,
            probability_model_config=probability_model_config,
        )
    return _compute_analysis_legacy(
        outcome=outcome,
        feature_engine=feature_engine,
        trend_scorer=trend_scorer,
        momentum_scorer=momentum_scorer,
        volatility_scorer=volatility_scorer,
        scenario_engine=scenario_engine,
        report_generator=report_generator,
    )


def _compute_analysis_quantile(
    *,
    outcome: DataFetchOutcome,
    report_generator: ReportGenerator,
    probability_model_config: Any,
) -> _ComputedAnalysis:
    try:
        from indextrack.domain.probability import MarketProbabilityModel, ProbabilityModelError
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        raise FeatureError(
            f"概率模型依赖缺失: No module named '{missing}'。"
        ) from exc

    model = MarketProbabilityModel(probability_model_config)
    try:
        model.fit(outcome.candles)
        horizon_outputs = model.predict_proba(outcome.candles)
        trend_signals = model.predict(outcome.candles)
        backtest = model.backtest(outcome.candles)
    except ProbabilityModelError as exc:
        raise FeatureError(f"概率模型失败: {exc}") from exc

    scenarios = _scenario_probs_from_horizon_outputs(horizon_outputs)
    display_labels = {
        horizon: getattr(horizon_outputs[horizon], "display_label", "uncertain")
        for horizon in ("short", "mid", "long")
    }
    report = report_generator.generate(
        symbol=outcome.symbol,
        trend_signals=trend_signals,
        scenario_probs=scenarios,
        data_status=outcome.data_status,
        used_cache_fallback=outcome.used_cache_fallback,
        display_labels=display_labels,
    )
    return _ComputedAnalysis(
        scenarios=scenarios,
        report=report,
        horizon_outputs=horizon_outputs,
        backtest_summary=backtest.to_dict(),
    )


def _compute_analysis_legacy(
    *,
    outcome: DataFetchOutcome,
    feature_engine: FeatureEngine,
    trend_scorer: TrendScorer,
    momentum_scorer: MomentumScorer,
    volatility_scorer: VolatilityScorer,
    scenario_engine: ScenarioEngine,
    report_generator: ReportGenerator,
) -> _ComputedAnalysis:
    features = feature_engine.compute(outcome.candles)
    trend_result = trend_scorer.score(features)
    momentum_scores = momentum_scorer.score(features)
    volatility_scores = volatility_scorer.score(features)
    scenarios = scenario_engine.generate(
        trend_scores=trend_result,
        momentum_scores=momentum_scores,
        volatility_scores=volatility_scores,
    )
    report = report_generator.generate(
        symbol=outcome.symbol,
        trend_signals=trend_result.signals,
        scenario_probs=scenarios,
        data_status=outcome.data_status,
        used_cache_fallback=outcome.used_cache_fallback,
    )
    return _ComputedAnalysis(scenarios=scenarios, report=report)


def _scenario_probs_from_horizon_outputs(
    outputs: dict[str, Any]
) -> list[ScenarioProbs]:
    return [
        ScenarioProbs(
            horizon="short",
            uptrend_pct=outputs["short"].display_prob_up * 100.0,
            sideways_pct=outputs["short"].display_prob_uncertain * 100.0,
            downtrend_pct=outputs["short"].display_prob_down * 100.0,
        ),
        ScenarioProbs(
            horizon="mid",
            uptrend_pct=outputs["mid"].display_prob_up * 100.0,
            sideways_pct=outputs["mid"].display_prob_uncertain * 100.0,
            downtrend_pct=outputs["mid"].display_prob_down * 100.0,
        ),
        ScenarioProbs(
            horizon="long",
            uptrend_pct=outputs["long"].display_prob_up * 100.0,
            sideways_pct=outputs["long"].display_prob_uncertain * 100.0,
            downtrend_pct=outputs["long"].display_prob_down * 100.0,
        ),
    ]


def _print_probability_debug_info(backtest_summary: dict[str, object]) -> None:
    print("概率调试信息:")
    for horizon in ("short", "mid", "long"):
        item = backtest_summary.get(horizon)
        if not isinstance(item, dict):
            continue
        diagnostics = item.get("diagnostics")
        if not isinstance(diagnostics, dict):
            continue
        label_dist = diagnostics.get("label_distribution", {})
        train_label_dist = diagnostics.get("train_label_distribution", label_dist)
        oof_label_dist = diagnostics.get("oof_label_distribution", label_dist)
        train_rows = int(diagnostics.get("train_rows", 0))
        oof_rows = int(diagnostics.get("oof_rows", 0))
        chain_shift = diagnostics.get("chain_shift", {})
        chain_metrics = diagnostics.get("chain_metrics", {})
        raw_metrics = chain_metrics.get("raw", {})
        calibrated_metrics = chain_metrics.get("calibrated", {})
        final_metrics = chain_metrics.get("final", {})
        base_probs = diagnostics.get("base_probs", {})
        threshold_up_stats = diagnostics.get("threshold_up_stats", {})
        threshold_down_stats = diagnostics.get("threshold_down_stats", {})
        q10_stats = diagnostics.get("q10_stats", {})
        q50_stats = diagnostics.get("q50_stats", {})
        q90_stats = diagnostics.get("q90_stats", {})
        sigma_stats = diagnostics.get("sigma_stats", {})
        binary_chain = diagnostics.get("binary_chain_metrics", {})
        binary_raw = binary_chain.get("raw", {})
        binary_cal = binary_chain.get("calibrated", {})
        binary_final = binary_chain.get("final", {})
        binary_pipeline = diagnostics.get("binary_pipeline", {})
        print(
            f"- {_horizon_label(horizon)} 标签分布 "
            f"train[{train_rows}]="
            f"{train_label_dist.get('pct_up', 0.0) * 100:.1f}%/"
            f"{train_label_dist.get('pct_flat', 0.0) * 100:.1f}%/"
            f"{train_label_dist.get('pct_down', 0.0) * 100:.1f}% "
            f"oof_valid[{oof_rows}]="
            f"{oof_label_dist.get('pct_up', 0.0) * 100:.1f}%/"
            f"{oof_label_dist.get('pct_flat', 0.0) * 100:.1f}%/"
            f"{oof_label_dist.get('pct_down', 0.0) * 100:.1f}% "
            f"base(d/f/u)={base_probs.get('down', 0.0) * 100:.1f}%/"
            f"{base_probs.get('flat', 0.0) * 100:.1f}%/"
            f"{base_probs.get('up', 0.0) * 100:.1f}%"
        )
        print(
            f"  raw(logloss={raw_metrics.get('log_loss', 0.0):.4f}, brier={raw_metrics.get('brier_score', 0.0):.4f}) "
            f"cal(logloss={calibrated_metrics.get('log_loss', 0.0):.4f}, brier={calibrated_metrics.get('brier_score', 0.0):.4f}) "
            f"final(logloss={final_metrics.get('log_loss', 0.0):.4f}, brier={final_metrics.get('brier_score', 0.0):.4f})"
        )
        print(
            f"  shift(raw->cal flip={chain_shift.get('direction_flip_rate_raw_to_cal', 0.0):.3f}, "
            f"mean_abs={chain_shift.get('mean_abs_shift_raw_to_cal', 0.0):.4f}) "
            f"threshold=[{threshold_down_stats.get('p50', 0.0) * 100:.2f}%,"
            f"{threshold_up_stats.get('p50', 0.0) * 100:.2f}%] "
            f"q10/q50/q90(p50)="
            f"{q10_stats.get('p50', 0.0) * 100:.2f}%/"
            f"{q50_stats.get('p50', 0.0) * 100:.2f}%/"
            f"{q90_stats.get('p50', 0.0) * 100:.2f}% "
            f"sigma_p50={sigma_stats.get('p50', 0.0) * 100:.2f}%"
        )
        print(f"  confusion raw={raw_metrics.get('confusion_matrix', [])}")
        print(f"  confusion cal={calibrated_metrics.get('confusion_matrix', [])}")
        print(f"  confusion final={final_metrics.get('confusion_matrix', [])}")
        if isinstance(binary_chain, dict) and binary_chain:
            print(
                "  binary(raw/cal/final) "
                f"logloss={binary_raw.get('log_loss', 0.0):.4f}/"
                f"{binary_cal.get('log_loss', 0.0):.4f}/"
                f"{binary_final.get('log_loss', 0.0):.4f} "
                f"brier={binary_raw.get('brier_score', 0.0):.4f}/"
                f"{binary_cal.get('brier_score', 0.0):.4f}/"
                f"{binary_final.get('brier_score', 0.0):.4f} "
                f"acc={binary_raw.get('accuracy', 0.0):.4f}/"
                f"{binary_cal.get('accuracy', 0.0):.4f}/"
                f"{binary_final.get('accuracy', 0.0):.4f}"
            )
        if isinstance(binary_pipeline, dict) and binary_pipeline:
            print(
                "  binary_pipeline "
                f"mode={binary_pipeline.get('mode', '')} "
                f"enabled={binary_pipeline.get('enabled', False)} "
                f"oof_dir_rows={int(binary_pipeline.get('directional_oof_rows', 0))} "
                f"prob_cap={float(binary_pipeline.get('prob_cap', 0.0)):.2f} "
                f"regime_shift_score={float(binary_pipeline.get('regime_shift_score', 0.0)):.3f}"
            )
        _print_reliability_bins("raw prob_down", raw_metrics.get("calibration_down", []))
        _print_reliability_bins("raw prob_up", raw_metrics.get("calibration_up", []))
        _print_reliability_bins("cal prob_down", calibrated_metrics.get("calibration_down", []))
        _print_reliability_bins("cal prob_up", calibrated_metrics.get("calibration_up", []))
        _print_reliability_bins("final prob_down", final_metrics.get("calibration_down", []))
        _print_reliability_bins("final prob_up", final_metrics.get("calibration_up", []))
        _print_reliability_bins("binary raw prob_up", binary_raw.get("calibration_up", []))
        _print_reliability_bins("binary cal prob_up", binary_cal.get("calibration_up", []))
        _print_reliability_bins("binary final prob_up", binary_final.get("calibration_up", []))


def _print_reliability_bins(name: str, bins: object) -> None:
    if not isinstance(bins, list):
        return
    if not bins:
        return
    compact = []
    for item in bins:
        if not isinstance(item, dict):
            continue
        compact.append(
            f"{item.get('bucket', '')}:pred={item.get('predicted_mean', 0.0):.2f},"
            f"actual={item.get('actual_rate', 0.0):.2f},n={int(item.get('count', 0))}"
        )
    if compact:
        print(f"  reliability {name}: " + " | ".join(compact))


def _run_probability_ablation(*, candles: Sequence[Any], base_config: Any | None) -> None:
    if base_config is None:
        raise FeatureError("无法运行消融实验：概率模型配置缺失。")
    try:
        from indextrack.domain.probability.experiments import format_ablation_table, run_ablation_suite
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        raise FeatureError(f"消融实验依赖缺失: No module named '{missing}'") from exc
    results, recommended = run_ablation_suite(candles=candles, base_config=base_config)
    print("概率消融实验（horizon-specific）:")
    print(format_ablation_table(results))
    if isinstance(recommended, dict):
        print("推荐参数方案（按 horizon）:")
        for horizon in ("short", "mid", "long"):
            item = recommended.get(horizon)
            if item is None:
                print(f"- {_horizon_label(horizon)}: 无可用推荐")
                continue
            print(
                f"- {_horizon_label(horizon)}: "
                f"group={item.group}, variant={item.variant}, "
                f"k={item.k_h:.3f}, lambda={item.lambda_h:.3f}, "
                f"cal_mode={item.calibration_mode}, blend={item.calibration_blend_h:.2f}, "
                f"shift_cap={item.max_calibration_shift_h:.2f}, "
                f"final_logloss={item.final_log_loss:.4f}, final_brier={item.final_brier:.4f}"
            )


def _build_probability_model_config(runtime_config: RuntimeConfig) -> Any:
    try:
        from indextrack.domain.probability.calibrator import CalibratorConfig
        from indextrack.domain.probability.label_builder import LabelConfig
        from indextrack.domain.probability.model import MarketProbabilityModelConfig
        from indextrack.domain.probability.probability_mapper import ProbabilityMapperConfig
        from indextrack.domain.probability.quantile_model import (
            QuantileModelConfig,
            TimeSeriesSplitConfig,
        )
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        raise FeatureError(
            f"概率模型依赖缺失: No module named '{missing}'。"
            "请先安装依赖（推荐: UV_CACHE_DIR=/tmp/uv-cache python3 -m uv sync）。"
        ) from exc

    prob = runtime_config.probability_model
    return MarketProbabilityModelConfig(
        label=LabelConfig(
            k_5=prob.k_5,
            k_20=prob.k_20,
            k_60=prob.k_60,
        ),
        quantile=QuantileModelConfig(
            learning_rate=prob.quantile_lr,
            max_iter=prob.quantile_iters,
        ),
        calibrator=CalibratorConfig(
            mode=prob.calibrator_mode,
            mode_5=prob.calibrator_mode_5,
            mode_20=prob.calibrator_mode_20,
            mode_60=prob.calibrator_mode_60,
            conservative_temperature=prob.calibrator_temperature,
            learning_rate=prob.calibrator_lr,
            max_iter=prob.calibrator_iters,
        ),
        mapper=ProbabilityMapperConfig(
            sigma_floor_5=prob.sigma_floor_5,
            sigma_floor_20=prob.sigma_floor_20,
            sigma_floor_60=prob.sigma_floor_60,
            lambda_5=prob.lambda_5,
            lambda_20=prob.lambda_20,
            lambda_60=prob.lambda_60,
            calibration_blend_5=prob.calibration_blend_5,
            calibration_blend_20=prob.calibration_blend_20,
            calibration_blend_60=prob.calibration_blend_60,
            max_calibration_shift_5=prob.calibration_max_shift_5,
            max_calibration_shift_20=prob.calibration_max_shift_20,
            max_calibration_shift_60=prob.calibration_max_shift_60,
            prob_cap=prob.prob_cap,
            display_prob_cap=prob.display_prob_cap,
        ),
        split=TimeSeriesSplitConfig(
            n_splits=prob.oof_splits,
            min_train_size=prob.min_train_size,
            min_valid_size=prob.min_valid_size,
        ),
    )


def _build_parser(
    *,
    default_index: str,
    default_period: str,
    default_model: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IndexTrack CLI")
    parser.add_argument("--index", choices=["SP500", "NASDAQ", "BOTH"], default=default_index)
    parser.add_argument("--period", default=default_period)
    parser.add_argument("--model", choices=["legacy", "quantile"], default=default_model)
    parser.add_argument("--timezone", default=None)
    parser.add_argument("--db-path", default=".data/indextrack.db")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-path", default=".data/indextrack.log")
    parser.add_argument("--prob-debug", action="store_true", help="输出 quantile 训练/校准调试信息")
    parser.add_argument("--prob-ablation", action="store_true", help="输出 quantile 消融实验对比表")
    parser.add_argument("--ui", action="store_true", help="启动本地 UI 服务")
    parser.add_argument("--ui-host", default="127.0.0.1")
    parser.add_argument("--ui-port", type=int, default=8000)
    return parser


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
    raise ValueError(f"不支持的 period: {period}，可用示例: 1M/3M/6M/1Y/3Y/5Y")


def _normalize_ui_period(period: str) -> str:
    value = period.strip().upper()
    return value if value in UI_PERIODS else "1M"


def _ui_lookback_calendar_days(
    *,
    period: str,
    model_mode: str,
    probability_model_config: Any | None,
) -> int:
    display_trading_days = _period_to_trading_days(period)
    baseline = max(320, int(display_trading_days * 2.0))
    if model_mode != "quantile":
        return baseline

    min_train_size = 120
    min_valid_size = 20
    if probability_model_config is not None:
        split_config = getattr(probability_model_config, "split", None)
        if split_config is not None:
            min_train_size = int(getattr(split_config, "min_train_size", min_train_size))
            min_valid_size = int(getattr(split_config, "min_valid_size", min_valid_size))

    # 经验下限: 预留 long horizon(60) + 最大特征窗口(60) + 至少两段 valid fold。
    min_trading_days_for_quantile = max(
        260,
        min_train_size + (2 * min_valid_size) + 120,
    )
    quantile_lookback = int(min_trading_days_for_quantile * 1.7)
    return max(baseline, quantile_lookback)


def _horizon_label(horizon: str) -> str:
    return {"short": "短期", "mid": "中期", "long": "长期"}.get(horizon, horizon)


def _friendly_error(exc: Exception) -> tuple[str, list[str]]:
    raw = str(exc)
    suggestions: list[str] = []

    if "nodename nor servname" in raw or "请求失败" in raw:
        suggestions.append("请检查网络连通性，确认可访问外部行情接口后重试。")
    if "无可用缓存" in raw:
        suggestions.append("先在网络可用时至少成功运行一次，以生成本地缓存。")
    if "period" in raw and "不支持" in raw:
        suggestions.append("将 period 改为 1M/3M/6M/1Y/3Y/5Y 或 ND（如 90D）。")
    if "不支持的时区" in raw:
        suggestions.append("请使用有效时区名，例如 America/New_York 或 Asia/Shanghai。")
    if "INDEXTRACK_" in raw:
        suggestions.append("请检查相关环境变量配置格式是否正确。")
    if "概率模型失败" in raw:
        suggestions.append("可临时切换 --model legacy 验证是否为新模型样本不足。")
        suggestions.append("或扩大分析周期（如 1Y/3Y）以提供更多训练样本。")
    if "OOF 样本不足" in raw:
        suggestions.append("可先运行 --period 1Y 刷新更长历史，再回到 UI 周期查看。")
        suggestions.append("如需对照旧参数，可设置 INDEXTRACK_PROB_PROFILE=baseline。")
    if "No module named" in raw:
        suggestions.append("缺少 Python 依赖，请先运行 UV_CACHE_DIR=/tmp/uv-cache python3 -m uv sync。")
        suggestions.append("若只缺 numpy，也可先运行 python3 -m pip install numpy。")

    return raw, suggestions


if __name__ == "__main__":
    raise SystemExit(main())
