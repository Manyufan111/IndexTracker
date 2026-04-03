"""Runtime configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from os import environ
from typing import Mapping

PRIMARY_API_KEY_ENV = "INDEXTRACK_PRIMARY_API_KEY"
TIMEZONE_ENV = "INDEXTRACK_TIMEZONE"
DEFAULT_INDEX_ENV = "INDEXTRACK_DEFAULT_INDEX"
DEFAULT_PERIOD_ENV = "INDEXTRACK_DEFAULT_PERIOD"
REQUEST_TIMEOUT_ENV = "INDEXTRACK_REQUEST_TIMEOUT_SEC"
SCENARIO_SHORT_WEIGHTS_ENV = "INDEXTRACK_SCENARIO_WEIGHTS_SHORT"
SCENARIO_MID_WEIGHTS_ENV = "INDEXTRACK_SCENARIO_WEIGHTS_MID"
SCENARIO_LONG_WEIGHTS_ENV = "INDEXTRACK_SCENARIO_WEIGHTS_LONG"
MODEL_MODE_ENV = "INDEXTRACK_MODEL"
PROB_K5_ENV = "INDEXTRACK_PROB_K_5"
PROB_K20_ENV = "INDEXTRACK_PROB_K_20"
PROB_K60_ENV = "INDEXTRACK_PROB_K_60"
PROB_LAMBDA5_ENV = "INDEXTRACK_PROB_LAMBDA_5"
PROB_LAMBDA20_ENV = "INDEXTRACK_PROB_LAMBDA_20"
PROB_LAMBDA60_ENV = "INDEXTRACK_PROB_LAMBDA_60"
PROB_SIGMA_FLOOR5_ENV = "INDEXTRACK_PROB_SIGMA_FLOOR_5"
PROB_SIGMA_FLOOR20_ENV = "INDEXTRACK_PROB_SIGMA_FLOOR_20"
PROB_SIGMA_FLOOR60_ENV = "INDEXTRACK_PROB_SIGMA_FLOOR_60"
PROB_CAP_ENV = "INDEXTRACK_PROB_CAP"
DISPLAY_PROB_CAP_ENV = "INDEXTRACK_DISPLAY_PROB_CAP"
PROB_OOF_SPLITS_ENV = "INDEXTRACK_PROB_OOF_SPLITS"
PROB_MIN_TRAIN_ENV = "INDEXTRACK_PROB_MIN_TRAIN_SIZE"
PROB_MIN_VALID_ENV = "INDEXTRACK_PROB_MIN_VALID_SIZE"
PROB_QUANTILE_LR_ENV = "INDEXTRACK_PROB_QUANTILE_LR"
PROB_QUANTILE_ITERS_ENV = "INDEXTRACK_PROB_QUANTILE_ITERS"
PROB_CALIB_LR_ENV = "INDEXTRACK_PROB_CALIB_LR"
PROB_CALIB_ITERS_ENV = "INDEXTRACK_PROB_CALIB_ITERS"
PROB_CALIB_MODE_ENV = "INDEXTRACK_PROB_CALIB_MODE"
PROB_CALIB_MODE5_ENV = "INDEXTRACK_PROB_CALIB_MODE_5"
PROB_CALIB_MODE20_ENV = "INDEXTRACK_PROB_CALIB_MODE_20"
PROB_CALIB_MODE60_ENV = "INDEXTRACK_PROB_CALIB_MODE_60"
PROB_CALIB_TEMP_ENV = "INDEXTRACK_PROB_CALIB_TEMP"
PROB_CALIB_BLEND5_ENV = "INDEXTRACK_PROB_CALIB_BLEND_5"
PROB_CALIB_BLEND20_ENV = "INDEXTRACK_PROB_CALIB_BLEND_20"
PROB_CALIB_BLEND60_ENV = "INDEXTRACK_PROB_CALIB_BLEND_60"
PROB_CALIB_SHIFT5_ENV = "INDEXTRACK_PROB_CALIB_MAX_SHIFT_5"
PROB_CALIB_SHIFT20_ENV = "INDEXTRACK_PROB_CALIB_MAX_SHIFT_20"
PROB_CALIB_SHIFT60_ENV = "INDEXTRACK_PROB_CALIB_MAX_SHIFT_60"
PROB_PROFILE_ENV = "INDEXTRACK_PROB_PROFILE"

DEFAULT_TIMEZONE = "America/New_York"
DEFAULT_INDEX = "BOTH"
DEFAULT_PERIOD = "1Y"
DEFAULT_TIMEOUT_SEC = 15
DEFAULT_SCENARIO_SHORT_WEIGHTS = (1.4, 1.1, 0.5, 1.1, 0.3)
DEFAULT_SCENARIO_MID_WEIGHTS = (1.5, 1.0, 0.45, 1.0, 0.28)
DEFAULT_SCENARIO_LONG_WEIGHTS = (1.6, 0.9, 0.4, 0.9, 0.25)
DEFAULT_MODEL_MODE = "quantile"

VALID_INDEX_OPTIONS = {"SP500", "NASDAQ", "BOTH"}
VALID_MODEL_OPTIONS = {"legacy", "quantile"}
VALID_CALIBRATOR_MODES = {"softmax", "conservative", "none"}
VALID_PROB_PROFILES = {"baseline", "optimized_v1"}


class ConfigError(ValueError):
    """Raised when runtime configuration is invalid."""


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved runtime configuration for the CLI application."""

    primary_api_key: str | None
    timezone: str
    default_index: str
    default_period: str
    request_timeout_sec: int
    scenario_weights: "ScenarioWeightsConfig"
    model_mode: str
    probability_model: "ProbabilityModelRuntimeConfig"


@dataclass(frozen=True)
class ScenarioWeightsConfig:
    """Per-horizon scenario engine weights."""

    short: tuple[float, float, float, float, float]
    mid: tuple[float, float, float, float, float]
    long: tuple[float, float, float, float, float]


@dataclass(frozen=True)
class ProbabilityModelRuntimeConfig:
    """Config for the calibrated quantile probability model."""

    k_5: float
    k_20: float
    k_60: float
    lambda_5: float
    lambda_20: float
    lambda_60: float
    sigma_floor_5: float
    sigma_floor_20: float
    sigma_floor_60: float
    prob_cap: float
    display_prob_cap: float
    oof_splits: int
    min_train_size: int
    min_valid_size: int
    quantile_lr: float
    quantile_iters: int
    calibrator_lr: float
    calibrator_iters: int
    calibrator_mode: str
    calibrator_mode_5: str
    calibrator_mode_20: str
    calibrator_mode_60: str
    calibrator_temperature: float
    calibration_blend_5: float
    calibration_blend_20: float
    calibration_blend_60: float
    calibration_max_shift_5: float
    calibration_max_shift_20: float
    calibration_max_shift_60: float
    profile: str


def load_runtime_config(env: Mapping[str, str] | None = None) -> RuntimeConfig:
    """Load runtime config from environment variables and defaults."""
    source = env if env is not None else environ

    default_index = _read_str(source, DEFAULT_INDEX_ENV, DEFAULT_INDEX).upper()
    if default_index not in VALID_INDEX_OPTIONS:
        raise ConfigError(
            f"{DEFAULT_INDEX_ENV} 必须是 SP500/NASDAQ/BOTH，当前值为: {default_index}"
        )

    request_timeout_sec = _read_int(source, REQUEST_TIMEOUT_ENV, DEFAULT_TIMEOUT_SEC)
    if request_timeout_sec <= 0:
        raise ConfigError(f"{REQUEST_TIMEOUT_ENV} 必须是正整数，当前值为: {request_timeout_sec}")

    model_mode = _read_str(source, MODEL_MODE_ENV, DEFAULT_MODEL_MODE).lower()
    if model_mode not in VALID_MODEL_OPTIONS:
        raise ConfigError(
            f"{MODEL_MODE_ENV} 必须是 legacy/quantile，当前值为: {model_mode}"
        )
    prob_profile = _read_str(source, PROB_PROFILE_ENV, "optimized_v1").lower()
    if prob_profile not in VALID_PROB_PROFILES:
        raise ConfigError(
            f"{PROB_PROFILE_ENV} 必须是 baseline/optimized_v1，当前值为: {prob_profile}"
        )
    profile_defaults = _probability_profile_defaults(prob_profile)
    calibrator_mode = _read_calibrator_mode(
        source,
        PROB_CALIB_MODE_ENV,
        str(profile_defaults["calibrator_mode"]),
    )
    calibrator_mode_5 = _read_calibrator_mode(
        source,
        PROB_CALIB_MODE5_ENV,
        str(profile_defaults["calibrator_mode_5"]),
    )
    calibrator_mode_20 = _read_calibrator_mode(
        source,
        PROB_CALIB_MODE20_ENV,
        str(profile_defaults["calibrator_mode_20"]),
    )
    calibrator_mode_60 = _read_calibrator_mode(
        source,
        PROB_CALIB_MODE60_ENV,
        str(profile_defaults["calibrator_mode_60"]),
    )

    return RuntimeConfig(
        primary_api_key=_read_optional_str(source, PRIMARY_API_KEY_ENV),
        timezone=_read_str(source, TIMEZONE_ENV, DEFAULT_TIMEZONE),
        default_index=default_index,
        default_period=_read_str(source, DEFAULT_PERIOD_ENV, DEFAULT_PERIOD).upper(),
        request_timeout_sec=request_timeout_sec,
        scenario_weights=ScenarioWeightsConfig(
            short=_read_weights(
                source,
                SCENARIO_SHORT_WEIGHTS_ENV,
                DEFAULT_SCENARIO_SHORT_WEIGHTS,
            ),
            mid=_read_weights(
                source,
                SCENARIO_MID_WEIGHTS_ENV,
                DEFAULT_SCENARIO_MID_WEIGHTS,
            ),
            long=_read_weights(
                source,
                SCENARIO_LONG_WEIGHTS_ENV,
                DEFAULT_SCENARIO_LONG_WEIGHTS,
            ),
        ),
        model_mode=model_mode,
        probability_model=ProbabilityModelRuntimeConfig(
            k_5=_read_float(source, PROB_K5_ENV, profile_defaults["k_5"]),
            k_20=_read_float(source, PROB_K20_ENV, profile_defaults["k_20"]),
            k_60=_read_float(source, PROB_K60_ENV, profile_defaults["k_60"]),
            lambda_5=_read_float(source, PROB_LAMBDA5_ENV, profile_defaults["lambda_5"]),
            lambda_20=_read_float(source, PROB_LAMBDA20_ENV, profile_defaults["lambda_20"]),
            lambda_60=_read_float(source, PROB_LAMBDA60_ENV, profile_defaults["lambda_60"]),
            sigma_floor_5=_read_float(source, PROB_SIGMA_FLOOR5_ENV, profile_defaults["sigma_floor_5"]),
            sigma_floor_20=_read_float(source, PROB_SIGMA_FLOOR20_ENV, profile_defaults["sigma_floor_20"]),
            sigma_floor_60=_read_float(source, PROB_SIGMA_FLOOR60_ENV, profile_defaults["sigma_floor_60"]),
            prob_cap=_read_float(source, PROB_CAP_ENV, profile_defaults["prob_cap"]),
            display_prob_cap=_read_float(source, DISPLAY_PROB_CAP_ENV, profile_defaults["display_prob_cap"]),
            oof_splits=_read_int(source, PROB_OOF_SPLITS_ENV, 5),
            min_train_size=_read_int(source, PROB_MIN_TRAIN_ENV, 120),
            min_valid_size=_read_int(source, PROB_MIN_VALID_ENV, 20),
            quantile_lr=_read_float(source, PROB_QUANTILE_LR_ENV, 0.05),
            quantile_iters=_read_int(source, PROB_QUANTILE_ITERS_ENV, 220),
            calibrator_lr=_read_float(source, PROB_CALIB_LR_ENV, 0.08),
            calibrator_iters=_read_int(source, PROB_CALIB_ITERS_ENV, 260),
            calibrator_mode=calibrator_mode,
            calibrator_mode_5=calibrator_mode_5,
            calibrator_mode_20=calibrator_mode_20,
            calibrator_mode_60=calibrator_mode_60,
            calibrator_temperature=_read_float(source, PROB_CALIB_TEMP_ENV, profile_defaults["calibrator_temperature"]),
            calibration_blend_5=_read_float(source, PROB_CALIB_BLEND5_ENV, profile_defaults["calibration_blend_5"]),
            calibration_blend_20=_read_float(source, PROB_CALIB_BLEND20_ENV, profile_defaults["calibration_blend_20"]),
            calibration_blend_60=_read_float(source, PROB_CALIB_BLEND60_ENV, profile_defaults["calibration_blend_60"]),
            calibration_max_shift_5=_read_float(source, PROB_CALIB_SHIFT5_ENV, profile_defaults["calibration_max_shift_5"]),
            calibration_max_shift_20=_read_float(source, PROB_CALIB_SHIFT20_ENV, profile_defaults["calibration_max_shift_20"]),
            calibration_max_shift_60=_read_float(source, PROB_CALIB_SHIFT60_ENV, profile_defaults["calibration_max_shift_60"]),
            profile=prob_profile,
        ),
    )


def _read_str(source: Mapping[str, str], name: str, default: str) -> str:
    value = source.get(name, default)
    return value.strip()


def _read_optional_str(source: Mapping[str, str], name: str) -> str | None:
    value = source.get(name)
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _read_int(source: Mapping[str, str], name: str, default: int) -> int:
    raw = source.get(name)
    if raw is None:
        return default
    cleaned = raw.strip()
    if not cleaned:
        return default
    try:
        return int(cleaned)
    except ValueError as exc:
        raise ConfigError(f"{name} 必须是整数，当前值为: {cleaned}") from exc


def _read_float(source: Mapping[str, str], name: str, default: float) -> float:
    raw = source.get(name)
    if raw is None:
        return default
    cleaned = raw.strip()
    if not cleaned:
        return default
    try:
        return float(cleaned)
    except ValueError as exc:
        raise ConfigError(f"{name} 必须是浮点数，当前值为: {cleaned}") from exc


def _read_calibrator_mode(source: Mapping[str, str], name: str, default: str) -> str:
    value = _read_str(source, name, default).lower()
    if value not in VALID_CALIBRATOR_MODES:
        raise ConfigError(
            f"{name} 必须是 softmax/conservative/none，当前值为: {value}"
        )
    return value


def _read_weights(
    source: Mapping[str, str],
    name: str,
    default: tuple[float, float, float, float, float],
) -> tuple[float, float, float, float, float]:
    raw = source.get(name)
    if raw is None:
        return default
    cleaned = raw.strip()
    if not cleaned:
        return default

    parts = [item.strip() for item in cleaned.split(",")]
    if len(parts) != 5:
        raise ConfigError(f"{name} 必须提供 5 个逗号分隔数字，当前值为: {cleaned}")

    values: list[float] = []
    for part in parts:
        try:
            values.append(float(part))
        except ValueError as exc:
            raise ConfigError(f"{name} 包含非数字值: {part}") from exc
    return (values[0], values[1], values[2], values[3], values[4])


def _probability_profile_defaults(profile: str) -> dict[str, float | str]:
    if profile == "baseline":
        return {
            "k_5": 0.35,
            "k_20": 0.50,
            "k_60": 0.65,
            "lambda_5": 0.80,
            "lambda_20": 0.75,
            "lambda_60": 0.70,
            "sigma_floor_5": 0.0030,
            "sigma_floor_20": 0.0060,
            "sigma_floor_60": 0.0100,
            "prob_cap": 0.72,
            "display_prob_cap": 0.90,
            "calibrator_mode": "softmax",
            "calibrator_mode_5": "softmax",
            "calibrator_mode_20": "softmax",
            "calibrator_mode_60": "softmax",
            "calibrator_temperature": 1.00,
            "calibration_blend_5": 1.00,
            "calibration_blend_20": 1.00,
            "calibration_blend_60": 1.00,
            "calibration_max_shift_5": 1.00,
            "calibration_max_shift_20": 1.00,
            "calibration_max_shift_60": 1.00,
        }
    return {
        "k_5": 0.35,
        "k_20": 0.50,
        "k_60": 0.60,
        "lambda_5": 0.80,
        "lambda_20": 1.00,
        "lambda_60": 1.00,
        "sigma_floor_5": 0.0030,
        "sigma_floor_20": 0.0060,
        "sigma_floor_60": 0.0100,
        "prob_cap": 0.72,
        "display_prob_cap": 0.90,
        "calibrator_mode": "conservative",
        "calibrator_mode_5": "conservative",
        "calibrator_mode_20": "none",
        "calibrator_mode_60": "none",
        "calibrator_temperature": 1.35,
        "calibration_blend_5": 1.00,
        "calibration_blend_20": 0.00,
        "calibration_blend_60": 0.00,
        "calibration_max_shift_5": 0.55,
        "calibration_max_shift_20": 0.00,
        "calibration_max_shift_60": 0.00,
    }
