from __future__ import annotations

import math
import numpy as np


def compute_sem(values: list[float] | np.ndarray) -> float:
    """Standard error of the mean."""
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(arr.std(ddof=1) / math.sqrt(len(arr)))


def divergence_rate(train_loss: float, test_loss: float) -> float:
    """test_loss - train_loss — generalization gap (proxy for overfitting)."""
    return test_loss - train_loss


def aggregate_seeds(
    per_seed_metrics: list[dict[str, float]],
) -> dict[str, float]:
    """Aggregate a list of per-seed metric dicts into mean ± SEM.

    For each key k in the dicts, the output contains:
        k_mean, k_sem
    """
    if not per_seed_metrics:
        return {}
    keys = per_seed_metrics[0].keys()
    result: dict[str, float] = {}
    for k in keys:
        vals = [m[k] for m in per_seed_metrics]
        result[f"{k}_mean"] = float(np.mean(vals))
        result[f"{k}_sem"] = compute_sem(vals)
    return result
