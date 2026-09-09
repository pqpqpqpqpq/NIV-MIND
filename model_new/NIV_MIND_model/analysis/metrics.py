"""Metrics used in the manuscript tables and figures."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)


def _arrays(
    labels: Iterable[int],
    scores: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    labels_array = np.asarray(list(labels), dtype=np.int64)
    scores_array = np.asarray(list(scores), dtype=np.float64)
    if labels_array.ndim != 1 or scores_array.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional")
    if labels_array.size != scores_array.size or labels_array.size == 0:
        raise ValueError("labels and scores must have the same non-zero length")
    if not np.all(np.isin(labels_array, (0, 1))):
        raise ValueError("labels must contain only 0 and 1")
    if np.unique(labels_array).size != 2:
        raise ValueError("both classes are required")
    if not np.isfinite(scores_array).all():
        raise ValueError("scores contain NaN or infinity")
    return labels_array, scores_array


def youden_threshold(labels: Iterable[int], scores: Iterable[float]) -> float:
    labels_array, scores_array = _arrays(labels, scores)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels_array,
        scores_array,
    )
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    indices = np.flatnonzero(finite)
    best = indices[
        np.argmax((true_positive_rate - false_positive_rate)[finite])
    ]
    return float(thresholds[best])


def binary_metrics(
    labels: Iterable[int],
    scores: Iterable[float],
    *,
    threshold: float | None = None,
) -> dict[str, float | int]:
    labels_array, scores_array = _arrays(labels, scores)
    selected_threshold = (
        youden_threshold(labels_array, scores_array)
        if threshold is None
        else float(threshold)
    )
    predictions = (scores_array >= selected_threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(
        labels_array,
        predictions,
        labels=[0, 1],
    ).ravel()
    return {
        "samples": int(labels_array.size),
        "failures": int(labels_array.sum()),
        "successes": int((labels_array == 0).sum()),
        "auc": float(roc_auc_score(labels_array, scores_array)),
        "auprc": float(average_precision_score(labels_array, scores_array)),
        "threshold": selected_threshold,
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "accuracy": float(accuracy_score(labels_array, predictions)),
        "precision": float(
            precision_score(labels_array, predictions, zero_division=0)
        ),
        "f1": float(f1_score(labels_array, predictions, zero_division=0)),
    }


def _bootstrap_auc_values(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> np.ndarray:
    if samples < 1:
        raise ValueError("samples must be positive")
    generator = np.random.default_rng(seed)
    values: list[float] = []
    while len(values) < samples:
        indices = generator.integers(0, labels.size, labels.size)
        selected_labels = labels[indices]
        if np.unique(selected_labels).size != 2:
            continue
        values.append(roc_auc_score(selected_labels, scores[indices]))
    return np.asarray(values, dtype=np.float64)


def bootstrap_auc_interval(
    labels: Iterable[int],
    scores: Iterable[float],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    labels_array, scores_array = _arrays(labels, scores)
    values = _bootstrap_auc_values(
        labels_array,
        scores_array,
        samples=samples,
        seed=seed,
    )
    alpha = (1.0 - confidence) / 2.0
    return {
        "auc": float(roc_auc_score(labels_array, scores_array)),
        "ci_low": float(np.quantile(values, alpha)),
        "ci_high": float(np.quantile(values, 1.0 - alpha)),
    }


def paired_auc_difference_interval(
    labels: Iterable[int],
    reference_scores: Iterable[float],
    comparison_scores: Iterable[float],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    labels_array, reference_array = _arrays(labels, reference_scores)
    comparison_array = np.asarray(list(comparison_scores), dtype=np.float64)
    if comparison_array.shape != reference_array.shape:
        raise ValueError("comparison_scores must match reference_scores")
    if not np.isfinite(comparison_array).all():
        raise ValueError("comparison_scores contain NaN or infinity")

    generator = np.random.default_rng(seed)
    differences: list[float] = []
    while len(differences) < samples:
        indices = generator.integers(0, labels_array.size, labels_array.size)
        selected = labels_array[indices]
        if np.unique(selected).size != 2:
            continue
        differences.append(
            roc_auc_score(selected, reference_array[indices])
            - roc_auc_score(selected, comparison_array[indices])
        )
    alpha = (1.0 - confidence) / 2.0
    return {
        "delta_auc": float(
            roc_auc_score(labels_array, reference_array)
            - roc_auc_score(labels_array, comparison_array)
        ),
        "ci_low": float(np.quantile(differences, alpha)),
        "ci_high": float(np.quantile(differences, 1.0 - alpha)),
    }


def decision_curve(
    labels: Iterable[int],
    scores: Iterable[float],
    *,
    thresholds: Iterable[float] | None = None,
) -> list[dict[str, float]]:
    labels_array, scores_array = _arrays(labels, scores)
    threshold_array = (
        np.linspace(0.01, 0.95, 95)
        if thresholds is None
        else np.asarray(list(thresholds), dtype=np.float64)
    )
    if np.any((threshold_array <= 0) | (threshold_array >= 1)):
        raise ValueError("decision thresholds must be between 0 and 1")
    prevalence = float(labels_array.mean())
    rows: list[dict[str, float]] = []
    for threshold in threshold_array:
        predictions = scores_array >= threshold
        true_positive = int(np.sum(predictions & (labels_array == 1)))
        false_positive = int(np.sum(predictions & (labels_array == 0)))
        odds = threshold / (1.0 - threshold)
        model_benefit = (
            true_positive / labels_array.size
            - false_positive / labels_array.size * odds
        )
        treat_all = prevalence - (1.0 - prevalence) * odds
        rows.append(
            {
                "threshold": float(threshold),
                "model": float(model_benefit),
                "treat_all": float(treat_all),
                "treat_none": 0.0,
            }
        )
    return rows


__all__ = [
    "binary_metrics",
    "bootstrap_auc_interval",
    "decision_curve",
    "paired_auc_difference_interval",
    "youden_threshold",
]
