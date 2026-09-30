from __future__ import annotations

import math

import numpy as np


def _arrays(y_true: object, y_pred: object) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} != {predicted.shape}")
    mask = np.isfinite(actual) & np.isfinite(predicted)
    return actual[mask], predicted[mask]


def regression_metrics(y_true: object, y_pred: object) -> dict[str, float]:
    actual, predicted = _arrays(y_true, y_pred)
    if actual.size == 0:
        raise ValueError("No finite observations")
    error = predicted - actual
    denominator = np.abs(actual).sum()
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(math.sqrt(np.square(error).mean())),
        "wape": float(np.abs(error).sum() / denominator) if denominator else float("nan"),
        "bias": float(error.mean()),
        "observations": int(actual.size),
    }


def precision_recall_at_k(y_true: object, scores: object, k: int) -> dict[str, float]:
    actual, predicted = _arrays(y_true, scores)
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, actual.size)
    selected = np.argsort(-predicted)[:k]
    true_positives = float(actual[selected].sum())
    positives = float(actual.sum())
    return {
        "precision_at_k": true_positives / k,
        "recall_at_k": true_positives / positives if positives else float("nan"),
        "k": k,
    }
