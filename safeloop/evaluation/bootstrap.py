from __future__ import annotations

import random
from statistics import mean


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    q = min(max(q, 0.0), 1.0)
    idx = q * (len(values) - 1)
    low = int(idx)
    high = min(low + 1, len(values) - 1)
    weight = idx - low
    return values[low] * (1.0 - weight) + values[high] * weight


def bootstrap_mean_ci(
    values: list[float],
    repeats: int = 1000,
    alpha: float = 0.05,
    seed: int = 7,
) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "samples": 0.0}
    rng = random.Random(seed)
    estimates = []
    for _ in range(repeats):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(mean(sample))
    return {
        "mean": mean(values),
        "lower": percentile(estimates, alpha / 2.0),
        "upper": percentile(estimates, 1.0 - alpha / 2.0),
        "samples": float(len(values)),
    }
