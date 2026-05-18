from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def absolute_count_error(y_true: float, y_pred: float) -> float:
    return abs(float(y_true) - float(y_pred))


def relative_count_error(y_true: float, y_pred: float) -> float:
    truth = float(y_true)
    if truth == 0.0:
        return 0.0 if float(y_pred) == 0.0 else math.inf
    return absolute_count_error(truth, y_pred) / truth


def compute_count_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    truth = np.asarray(list(y_true), dtype=np.float64)
    pred = np.asarray(list(y_pred), dtype=np.float64)
    if truth.shape != pred.shape:
        raise ValueError("Ground truth and prediction arrays must have the same shape.")
    if truth.size == 0:
        raise ValueError("At least one value is required to compute metrics.")

    residual = pred - truth
    absolute = np.abs(residual)
    relative = np.divide(absolute, np.maximum(truth, 1.0))

    return {
        "n": float(truth.size),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mape": float(relative.mean()),
        "bias": float(residual.mean()),
        "median_ae": float(np.median(absolute)),
        "max_ae": float(absolute.max()),
    }

