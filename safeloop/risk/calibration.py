from __future__ import annotations

from dataclasses import dataclass, field

from safeloop.types import FeatureRow


@dataclass
class GroupCalibrator:
    target_risk: float = 0.02
    min_coverage: int = 1
    default_threshold: float = 0.0
    thresholds: dict[str, float] = field(default_factory=dict)

    def fit(self, rows: list[FeatureRow], scores: list[float]) -> "GroupCalibrator":
        grouped: dict[str, list[tuple[float, int]]] = {}
        for row, score in zip(rows, scores):
            grouped.setdefault(row.group, []).append((float(score), int(row.label)))
        for group, values in grouped.items():
            self.thresholds[group] = self._select_threshold(values)
        return self

    def threshold_for(self, group: str) -> float:
        return self.thresholds.get(group, self.default_threshold)

    def should_exit(self, group: str, score: float) -> bool:
        return score <= self.threshold_for(group)

    def _select_threshold(self, values: list[tuple[float, int]]) -> float:
        if not values:
            return self.default_threshold
        values = sorted(values, key=lambda pair: pair[0])
        best = self.default_threshold
        errors = 0
        accepted = 0
        for score, label in values:
            accepted += 1
            errors += int(label)
            empirical_risk = errors / max(accepted, 1)
            if accepted >= self.min_coverage and empirical_risk <= self.target_risk:
                best = score
        return best

    def to_dict(self) -> dict:
        return {
            "target_risk": self.target_risk,
            "min_coverage": self.min_coverage,
            "default_threshold": self.default_threshold,
            "thresholds": self.thresholds,
        }

