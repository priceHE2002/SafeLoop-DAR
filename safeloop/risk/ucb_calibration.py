from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist

from safeloop.types import FeatureRow


def wilson_upper_bound(errors: int, total: int, delta: float) -> float:
    """One-sided Wilson upper confidence bound for a Bernoulli risk."""

    if total <= 0:
        return 1.0
    delta = min(max(delta, 1e-12), 1.0 - 1e-12)
    z = NormalDist().inv_cdf(1.0 - delta)
    phat = errors / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2.0 * total)) / denom
    radius = z * math.sqrt(phat * (1.0 - phat) / total + z * z / (4.0 * total * total)) / denom
    return min(1.0, max(0.0, center + radius))


@dataclass
class UCBGroupCalibrator:
    """Group-wise threshold selection with an upper confidence bound on risk."""

    target_risk: float = 0.02
    delta: float = 0.05
    min_coverage: int = 1
    default_threshold: float = 0.0
    thresholds: dict[str, float] = field(default_factory=dict)
    stats: dict[str, dict[str, float]] = field(default_factory=dict)

    def fit(self, rows: list[FeatureRow], scores: list[float]) -> "UCBGroupCalibrator":
        grouped: dict[str, list[tuple[float, int]]] = {}
        for row, score in zip(rows, scores):
            grouped.setdefault(row.group, []).append((float(score), int(row.label)))
        group_delta = self.delta / max(len(grouped), 1)
        for group, values in grouped.items():
            threshold, stat = self._select_threshold(values, group_delta)
            self.thresholds[group] = threshold
            self.stats[group] = stat
        return self

    def threshold_for(self, group: str) -> float:
        return self.thresholds.get(group, self.default_threshold)

    def should_exit(self, group: str, score: float) -> bool:
        return score <= self.threshold_for(group)

    def _select_threshold(self, values: list[tuple[float, int]], group_delta: float) -> tuple[float, dict[str, float]]:
        if not values:
            return self.default_threshold, {
                "accepted": 0.0,
                "errors": 0.0,
                "empirical_risk": 0.0,
                "risk_upper_bound": 1.0,
                "delta": group_delta,
            }
        values = sorted(values, key=lambda pair: pair[0])
        best_threshold = self.default_threshold
        best_stat = {
            "accepted": 0.0,
            "errors": 0.0,
            "empirical_risk": 0.0,
            "risk_upper_bound": 1.0,
            "delta": group_delta,
        }
        errors = 0
        accepted = 0
        for score, label in values:
            accepted += 1
            errors += int(label)
            empirical_risk = errors / max(accepted, 1)
            upper_bound = wilson_upper_bound(errors, accepted, group_delta)
            if accepted >= self.min_coverage and upper_bound <= self.target_risk:
                best_threshold = score
                best_stat = {
                    "accepted": float(accepted),
                    "errors": float(errors),
                    "empirical_risk": empirical_risk,
                    "risk_upper_bound": upper_bound,
                    "delta": group_delta,
                }
        return best_threshold, best_stat

    def to_dict(self) -> dict:
        return {
            "method": "ucb",
            "target_risk": self.target_risk,
            "delta": self.delta,
            "min_coverage": self.min_coverage,
            "default_threshold": self.default_threshold,
            "thresholds": self.thresholds,
            "stats": self.stats,
        }
