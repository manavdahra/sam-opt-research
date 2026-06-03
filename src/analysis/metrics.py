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
    """Aggregate a list of per-seed metric dicts into mean +/- SEM.

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


def epochs_to_threshold(acc_curve: list[float], tau: float) -> int | None:
    """First epoch (1-indexed) where validation accuracy >= tau.

    Args:
        acc_curve: Per-epoch validation accuracy values (index 0 = epoch 1).
        tau: Accuracy threshold in [0, 1].

    Returns:
        1-indexed epoch number, or None if the threshold was never reached.
    """
    for epoch, acc in enumerate(acc_curve, start=1):
        if acc >= tau:
            return epoch
    return None


def flops_to_threshold(
    flops_per_epoch: float, threshold_epoch: int | None
) -> float | None:
    """Total FLOPs accumulated up to and including threshold_epoch.

    SAM-family optimizers run two forward-backward passes per batch, so
    their flops_per_epoch should already encode the 2× multiplier relative
    to SGD before calling this function.

    Args:
        flops_per_epoch: FLOPs consumed per training epoch for this optimizer.
        threshold_epoch: 1-indexed epoch returned by epochs_to_threshold.

    Returns:
        Total FLOPs as a float, or None if threshold was never reached.
    """
    if threshold_epoch is None:
        return None
    return flops_per_epoch * threshold_epoch


def wallclock_to_threshold(
    time_curve: list[float], threshold_epoch: int | None
) -> float | None:
    """Cumulative wall-clock seconds from epoch 1 up to threshold_epoch.

    Args:
        time_curve: Per-epoch elapsed_sec values (index 0 = epoch 1).
        threshold_epoch: 1-indexed epoch returned by epochs_to_threshold.

    Returns:
        Cumulative seconds, or None if threshold was never reached.
    """
    if threshold_epoch is None:
        return None
    return float(sum(time_curve[:threshold_epoch]))
