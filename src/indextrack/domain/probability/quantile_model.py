"""Quantile regression and time-series OOF helpers."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterator

import numpy as np

LOGGER = logging.getLogger("indextrack.probability.quantile_model")


@dataclass(frozen=True)
class QuantileModelConfig:
    """Training config for linear quantile regressors."""

    learning_rate: float = 0.05
    max_iter: int = 220
    l2: float = 1e-3
    clip_gradient: float = 5.0
    early_stop_rounds: int = 40
    early_stop_tol: float = 1e-5


class QuantileLinearRegressor:
    """Simple linear quantile regression via gradient descent on pinball loss."""

    def __init__(self, quantile: float, config: QuantileModelConfig | None = None) -> None:
        if not (0 < quantile < 1):
            raise ValueError("quantile 必须位于 (0,1)")
        self.quantile = float(quantile)
        self.config = config or QuantileModelConfig()
        self._weights: np.ndarray | None = None
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "QuantileLinearRegressor":
        if x.ndim != 2:
            raise ValueError("x 必须是二维矩阵")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x 与 y 行数不一致")
        if x.shape[0] < 20:
            raise ValueError("样本量不足，至少需要 20 行")

        x_norm = self._fit_transform_x(x)
        x_bias = np.column_stack([np.ones(x_norm.shape[0]), x_norm])
        weights = np.zeros(x_bias.shape[1], dtype=float)

        best_loss = float("inf")
        no_improve = 0
        for iteration in range(self.config.max_iter):
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                pred = x_bias @ weights
            pred = np.nan_to_num(pred, nan=0.0, posinf=1.5, neginf=-1.5)
            pred = np.clip(pred, -1.5, 1.5)
            residual = y - pred
            grad_factor = np.where(
                residual > 0,
                -self.quantile,
                (1.0 - self.quantile),
            )
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                grad = (x_bias.T @ grad_factor) / x_bias.shape[0]
            grad = np.nan_to_num(grad, nan=0.0, posinf=self.config.clip_gradient, neginf=-self.config.clip_gradient)
            grad[1:] += self.config.l2 * weights[1:]
            grad = np.clip(grad, -self.config.clip_gradient, self.config.clip_gradient)
            lr = self.config.learning_rate / (1.0 + 0.01 * iteration)
            weights -= lr * grad
            weights = np.clip(weights, -8.0, 8.0)

            loss = _pinball_loss(y, pred, self.quantile) + 0.5 * self.config.l2 * float(
                np.sum(weights[1:] ** 2)
            )
            if loss + self.config.early_stop_tol < best_loss:
                best_loss = loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.config.early_stop_rounds:
                    break

        self._weights = weights
        LOGGER.debug(
            "quantile_model_fit quantile=%.2f rows=%s cols=%s best_loss=%.6f iter=%s",
            self.quantile,
            x.shape[0],
            x.shape[1],
            best_loss,
            iteration + 1,
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._weights is None or self._feature_mean is None or self._feature_std is None:
            raise RuntimeError("模型未训练，无法预测")
        if x.ndim != 2:
            raise ValueError("x 必须是二维矩阵")

        std = np.where(self._feature_std > 1e-12, self._feature_std, 1.0)
        x_norm = (x - self._feature_mean) / std
        x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)
        x_bias = np.column_stack([np.ones(x_norm.shape[0]), x_norm])
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            pred = x_bias @ self._weights
        pred = np.nan_to_num(pred, nan=0.0, posinf=1.5, neginf=-1.5)
        return np.clip(pred, -1.5, 1.5)

    def _fit_transform_x(self, x: np.ndarray) -> np.ndarray:
        feature_mean = np.nanmean(x, axis=0)
        feature_std = np.nanstd(x, axis=0, ddof=0)
        feature_std[feature_std < 1e-12] = 1.0
        x_norm = (x - feature_mean) / feature_std
        x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)
        self._feature_mean = feature_mean
        self._feature_std = feature_std
        return x_norm


@dataclass(frozen=True)
class TimeSeriesSplitConfig:
    """Expanding-window split configuration."""

    n_splits: int = 5
    min_train_size: int = 120
    min_valid_size: int = 20


def expanding_time_series_splits(
    n_samples: int,
    config: TimeSeriesSplitConfig,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield train/valid index arrays with strict chronological order."""
    if n_samples < config.min_train_size + config.min_valid_size:
        midpoint = max(config.min_train_size, int(n_samples * 0.7))
        if midpoint >= n_samples:
            return
        train_idx = np.arange(0, midpoint, dtype=int)
        valid_idx = np.arange(midpoint, n_samples, dtype=int)
        if valid_idx.shape[0] >= config.min_valid_size:
            yield train_idx, valid_idx
        return

    available = n_samples - config.min_train_size
    fold_size = max(config.min_valid_size, available // max(config.n_splits, 1))
    train_end = config.min_train_size
    while train_end + config.min_valid_size <= n_samples:
        valid_end = min(train_end + fold_size, n_samples)
        train_idx = np.arange(0, train_end, dtype=int)
        valid_idx = np.arange(train_end, valid_end, dtype=int)
        if valid_idx.shape[0] >= config.min_valid_size:
            yield train_idx, valid_idx
        if valid_end == n_samples:
            break
        train_end = valid_end


def _pinball_loss(y: np.ndarray, pred: np.ndarray, quantile: float) -> float:
    residual = y - pred
    positive = np.maximum(residual, 0.0) * quantile
    negative = np.maximum(-residual, 0.0) * (1.0 - quantile)
    return float(np.mean(positive + negative))
