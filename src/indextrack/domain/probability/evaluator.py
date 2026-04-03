"""Evaluation utilities for multiclass probability forecasts."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

import numpy as np

LOGGER = logging.getLogger("indextrack.probability.evaluator")


@dataclass(frozen=True)
class CalibrationBucket:
    """One probability calibration bucket summary."""

    bucket: str
    count: int
    predicted_mean: float
    actual_rate: float


@dataclass(frozen=True)
class EvaluationMetrics:
    """Core multiclass metrics."""

    log_loss: float
    brier_score: float
    accuracy: float
    confusion_matrix: list[list[int]]
    calibration_down: list[CalibrationBucket] = field(default_factory=list)
    calibration_up: list[CalibrationBucket] = field(default_factory=list)
    regime_metrics: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class BinaryEvaluationMetrics:
    """Core binary directional metrics."""

    log_loss: float
    brier_score: float
    accuracy: float
    calibration_up: list[CalibrationBucket] = field(default_factory=list)
    calibration_down: list[CalibrationBucket] = field(default_factory=list)


def evaluate_probabilities(
    *,
    probs: np.ndarray,
    labels: np.ndarray,
    regimes: np.ndarray,
    bins: int = 8,
) -> EvaluationMetrics:
    valid = np.isfinite(probs).all(axis=1) & (labels >= 0)
    probs = probs[valid]
    labels = labels[valid]
    regimes = regimes[valid]
    if probs.shape[0] == 0:
        return EvaluationMetrics(
            log_loss=0.0,
            brier_score=0.0,
            accuracy=0.0,
            confusion_matrix=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        )

    log_loss = _multiclass_log_loss(probs, labels)
    brier = _multiclass_brier_score(probs, labels)
    pred = np.argmax(probs, axis=1)
    accuracy = float(np.mean(pred == labels))
    confusion = _confusion_matrix(labels, pred)

    calib_down = _calibration_table(probs[:, 0], labels == 0, bins=bins)
    calib_up = _calibration_table(probs[:, 2], labels == 2, bins=bins)
    regime_report = _regime_metrics(probs, labels, regimes)

    LOGGER.debug(
        "evaluation_done rows=%s log_loss=%.6f brier=%.6f accuracy=%.4f",
        probs.shape[0],
        log_loss,
        brier,
        accuracy,
    )
    return EvaluationMetrics(
        log_loss=log_loss,
        brier_score=brier,
        accuracy=accuracy,
        confusion_matrix=confusion,
        calibration_down=calib_down,
        calibration_up=calib_up,
        regime_metrics=regime_report,
    )


def evaluate_binary_probabilities(
    *,
    prob_up: np.ndarray,
    labels_binary_up: np.ndarray,
    bins: int = 10,
) -> BinaryEvaluationMetrics:
    valid = np.isfinite(prob_up) & np.isfinite(labels_binary_up)
    probs = np.clip(prob_up[valid], 0.0, 1.0)
    labels = labels_binary_up[valid].astype(int)
    if probs.shape[0] == 0:
        return BinaryEvaluationMetrics(
            log_loss=0.0,
            brier_score=0.0,
            accuracy=0.0,
            calibration_up=[],
            calibration_down=[],
        )
    safe = np.clip(probs, 1e-12, 1.0 - 1e-12)
    log_loss = float(-np.mean(labels * np.log(safe) + (1 - labels) * np.log(1.0 - safe)))
    brier = float(np.mean((probs - labels) ** 2))
    pred = (probs >= 0.5).astype(int)
    accuracy = float(np.mean(pred == labels))
    calib_up = _calibration_table(probs, labels == 1, bins=bins)
    calib_down = _calibration_table(1.0 - probs, labels == 0, bins=bins)
    return BinaryEvaluationMetrics(
        log_loss=log_loss,
        brier_score=brier,
        accuracy=accuracy,
        calibration_up=calib_up,
        calibration_down=calib_down,
    )


def metrics_to_dict(metrics: EvaluationMetrics) -> dict[str, Any]:
    """Serialize metric dataclass into plain dictionary."""
    return {
        "log_loss": metrics.log_loss,
        "brier_score": metrics.brier_score,
        "accuracy": metrics.accuracy,
        "confusion_matrix": metrics.confusion_matrix,
        "calibration_down": [
            {
                "bucket": item.bucket,
                "count": item.count,
                "predicted_mean": item.predicted_mean,
                "actual_rate": item.actual_rate,
            }
            for item in metrics.calibration_down
        ],
        "calibration_up": [
            {
                "bucket": item.bucket,
                "count": item.count,
                "predicted_mean": item.predicted_mean,
                "actual_rate": item.actual_rate,
            }
            for item in metrics.calibration_up
        ],
        "regime_metrics": metrics.regime_metrics,
    }


def binary_metrics_to_dict(metrics: BinaryEvaluationMetrics) -> dict[str, Any]:
    """Serialize binary metric dataclass into plain dictionary."""
    return {
        "log_loss": metrics.log_loss,
        "brier_score": metrics.brier_score,
        "accuracy": metrics.accuracy,
        "calibration_up": [
            {
                "bucket": item.bucket,
                "count": item.count,
                "predicted_mean": item.predicted_mean,
                "actual_rate": item.actual_rate,
            }
            for item in metrics.calibration_up
        ],
        "calibration_down": [
            {
                "bucket": item.bucket,
                "count": item.count,
                "predicted_mean": item.predicted_mean,
                "actual_rate": item.actual_rate,
            }
            for item in metrics.calibration_down
        ],
    }


def _multiclass_log_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    safe = np.clip(probs, 1e-12, 1.0)
    selected = safe[np.arange(labels.shape[0]), labels]
    return float(-np.mean(np.log(selected)))


def _multiclass_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.eye(3)[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for true, pred in zip(y_true, y_pred):
        matrix[int(true)][int(pred)] += 1
    return matrix


def _calibration_table(pred_prob: np.ndarray, event: np.ndarray, bins: int) -> list[CalibrationBucket]:
    out: list[CalibrationBucket] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for left, right in zip(edges[:-1], edges[1:]):
        if right >= 1.0:
            mask = (pred_prob >= left) & (pred_prob <= right)
        else:
            mask = (pred_prob >= left) & (pred_prob < right)
        count = int(np.sum(mask))
        if count == 0:
            continue
        predicted_mean = float(np.mean(pred_prob[mask]))
        actual_rate = float(np.mean(event[mask]))
        out.append(
            CalibrationBucket(
                bucket=f"[{left:.2f},{right:.2f}]",
                count=count,
                predicted_mean=predicted_mean,
                actual_rate=actual_rate,
            )
        )
    return out


def _regime_metrics(probs: np.ndarray, labels: np.ndarray, regimes: np.ndarray) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for regime in ("low_vol", "mid_vol", "high_vol"):
        mask = regimes == regime
        if int(np.sum(mask)) == 0:
            continue
        sub_probs = probs[mask]
        sub_labels = labels[mask]
        pred = np.argmax(sub_probs, axis=1)
        report[regime] = {
            "count": float(sub_probs.shape[0]),
            "log_loss": _multiclass_log_loss(sub_probs, sub_labels),
            "brier_score": _multiclass_brier_score(sub_probs, sub_labels),
            "accuracy": float(np.mean(pred == sub_labels)),
        }
    return report
