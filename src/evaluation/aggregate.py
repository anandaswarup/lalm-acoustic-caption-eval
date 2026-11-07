"""Paired percentile bootstrap over recordings (E10, used in E11-E15).

Every system is resampled with the same recording indices, so per-system CIs and the CIs of differences between systems
come from the same resamples. The statistic is the mean of per-item scores; for BLEU and METEOR, whose corpus value is
not that mean, the CI is on the per-item mean and the corpus value is reported alongside it.
"""

from __future__ import annotations

import numpy as np


def bootstrap_indices(num_items: int, num_resamples: int, seed: int) -> np.ndarray:
    """(num_resamples, num_items) recording indices drawn with replacement."""
    if num_items < 1 or num_resamples < 1:
        raise ValueError(f"Need at least one item and one resample (got {num_items}, {num_resamples})")
    return np.random.default_rng(seed).integers(0, num_items, size=(num_resamples, num_items))


def percentile_interval(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = 1.0 - confidence
    low, high = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def bootstrap_mean(values: np.ndarray, indices: np.ndarray, confidence: float) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if indices.max() >= len(values):
        raise ValueError(f"Resample indices go up to {indices.max()} but there are {len(values)} items")
    low, high = percentile_interval(values[indices].mean(axis=1), confidence)
    return {"mean": float(values.mean()), "ci_low": low, "ci_high": high}


def bootstrap_difference(a: np.ndarray, b: np.ndarray, indices: np.ndarray, confidence: float) -> dict[str, float]:
    """Paired difference mean(a) - mean(b): a and b must score the same recordings in the same order."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Paired scores differ in shape: {a.shape} vs {b.shape}")
    return bootstrap_mean(a - b, indices, confidence)
