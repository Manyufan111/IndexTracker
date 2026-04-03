"""Probability calibration utilities (multiclass + binary display layer)."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

LOGGER = logging.getLogger("indextrack.probability.calibrator")


@dataclass(frozen=True)
class CalibratorConfig:
    """Softmax calibrator hyper-parameters."""

    mode: str = "conservative"
    mode_5: str | None = None
    mode_20: str | None = None
    mode_60: str | None = None
    conservative_temperature: float = 1.35
    learning_rate: float = 0.08
    max_iter: int = 260
    l2: float = 1e-3
    early_stop_rounds: int = 60
    early_stop_tol: float = 1e-6
    recent_oof_ratio_5: float = 1.0
    recent_oof_ratio_20: float = 1.0
    recent_oof_ratio_60: float = 1.0

    def normalized_mode(self) -> str:
        return _normalize_mode(self.mode)

    def normalized_mode_for_horizon_days(self, horizon_days: int) -> str:
        override: str | None
        if horizon_days <= 5:
            override = self.mode_5
        elif horizon_days <= 20:
            override = self.mode_20
        else:
            override = self.mode_60
        if override is None:
            return self.normalized_mode()
        cleaned = override.strip()
        if not cleaned:
            return self.normalized_mode()
        return _normalize_mode(cleaned)

    def recent_oof_ratio(self, horizon_days: int) -> float:
        if horizon_days <= 5:
            value = self.recent_oof_ratio_5
        elif horizon_days <= 20:
            value = self.recent_oof_ratio_20
        else:
            value = self.recent_oof_ratio_60
        return float(np.clip(value, 0.05, 1.0))


class SoftmaxCalibrator:
    """Multiclass logistic regression over log(raw_probs)."""

    def __init__(self, config: CalibratorConfig | None = None) -> None:
        self.config = config or CalibratorConfig()
        self._weights: np.ndarray | None = None
        self._bias: np.ndarray | None = None
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SoftmaxCalibrator":
        if x.ndim != 2:
            raise ValueError("x 必须是二维矩阵")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x 与 y 行数不一致")
        if x.shape[0] < 30:
            raise ValueError("校准样本不足，至少需要 30 条")
        x_norm = self._fit_transform_x(x)

        n_samples, n_features = x_norm.shape
        n_classes = 3
        y_onehot = np.eye(n_classes)[y]

        weights = np.zeros((n_features, n_classes), dtype=float)
        bias = np.zeros(n_classes, dtype=float)

        best_loss = float("inf")
        no_improve = 0
        for iteration in range(self.config.max_iter):
            logits = x_norm @ weights + bias
            probs = _softmax(logits)
            grad_logits = (probs - y_onehot) / n_samples
            grad_w = x_norm.T @ grad_logits + self.config.l2 * weights
            grad_b = np.sum(grad_logits, axis=0)
            lr = self.config.learning_rate / (1.0 + 0.01 * iteration)
            weights -= lr * grad_w
            bias -= lr * grad_b

            loss = _log_loss(y_onehot, probs) + 0.5 * self.config.l2 * float(np.sum(weights ** 2))
            if loss + self.config.early_stop_tol < best_loss:
                best_loss = loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.config.early_stop_rounds:
                    break

        self._weights = weights
        self._bias = bias
        LOGGER.debug(
            "calibrator_fit rows=%s cols=%s best_loss=%.6f iter=%s",
            n_samples,
            n_features,
            best_loss,
            iteration + 1,
        )
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self._weights is None or self._bias is None or self._x_mean is None or self._x_std is None:
            raise RuntimeError("calibrator 尚未训练")
        std = np.where(self._x_std > 1e-12, self._x_std, 1.0)
        x_norm = (x - self._x_mean) / std
        x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)
        logits = x_norm @ self._weights + self._bias
        return _softmax(logits)

    def _fit_transform_x(self, x: np.ndarray) -> np.ndarray:
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0, ddof=0)
        std[std < 1e-12] = 1.0
        x_norm = (x - mean) / std
        x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)
        self._x_mean = mean
        self._x_std = std
        return x_norm


class BinaryLogitCalibrator:
    """Binary logistic calibrator over directional features."""

    def __init__(self, config: CalibratorConfig | None = None) -> None:
        self.config = config or CalibratorConfig()
        self._weights: np.ndarray | None = None
        self._bias: float | None = None
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BinaryLogitCalibrator":
        if x.ndim != 2:
            raise ValueError("x 必须是二维矩阵")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x 与 y 行数不一致")
        if x.shape[0] < 30:
            raise ValueError("二分类校准样本不足，至少需要 30 条")
        y_values = y.astype(int)
        unique = set(np.unique(y_values).tolist())
        if unique != {0, 1}:
            raise ValueError("二分类校准标签必须同时包含 0 和 1")

        x_norm = self._fit_transform_x(x)
        n_samples, n_features = x_norm.shape
        weights = np.zeros(n_features, dtype=float)
        bias = 0.0

        best_loss = float("inf")
        no_improve = 0
        for iteration in range(self.config.max_iter):
            logits = x_norm @ weights + bias
            probs = _sigmoid(logits)
            grad = (probs - y_values) / n_samples
            grad_w = x_norm.T @ grad + self.config.l2 * weights
            grad_b = float(np.sum(grad))
            lr = self.config.learning_rate / (1.0 + 0.01 * iteration)
            weights -= lr * grad_w
            bias -= lr * grad_b

            loss = _binary_log_loss(y_values, probs) + 0.5 * self.config.l2 * float(np.sum(weights ** 2))
            if loss + self.config.early_stop_tol < best_loss:
                best_loss = loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.config.early_stop_rounds:
                    break

        self._weights = weights
        self._bias = bias
        LOGGER.debug(
            "binary_calibrator_fit rows=%s cols=%s best_loss=%.6f iter=%s",
            n_samples,
            n_features,
            best_loss,
            iteration + 1,
        )
        return self

    def predict_up_prob(self, x: np.ndarray) -> np.ndarray:
        if (
            self._weights is None
            or self._bias is None
            or self._x_mean is None
            or self._x_std is None
        ):
            raise RuntimeError("binary calibrator 尚未训练")
        std = np.where(self._x_std > 1e-12, self._x_std, 1.0)
        x_norm = (x - self._x_mean) / std
        x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)
        logits = x_norm @ self._weights + self._bias
        return _sigmoid(logits)

    def _fit_transform_x(self, x: np.ndarray) -> np.ndarray:
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0, ddof=0)
        std[std < 1e-12] = 1.0
        x_norm = (x - mean) / std
        x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)
        self._x_mean = mean
        self._x_std = std
        return x_norm


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    denom = np.sum(exp_values, axis=1, keepdims=True)
    denom = np.where(denom <= 1e-12, 1.0, denom)
    return exp_values / denom


def _log_loss(y_onehot: np.ndarray, probs: np.ndarray) -> float:
    safe_probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.mean(np.sum(y_onehot * np.log(safe_probs), axis=1)))


def _binary_log_loss(y_true: np.ndarray, prob_up: np.ndarray) -> float:
    safe = np.clip(prob_up, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y_true * np.log(safe) + (1 - y_true) * np.log(1.0 - safe)))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _normalize_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"softmax", "conservative", "none"}:
        return normalized
    return "softmax"
