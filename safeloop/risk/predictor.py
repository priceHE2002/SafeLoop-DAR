from __future__ import annotations

import math
from dataclasses import dataclass, field

from safeloop.types import FeatureRow


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class OnlineLogisticRiskPredictor:
    """Small stdlib-only logistic predictor for risk scores."""

    learning_rate: float = 0.05
    epochs: int = 30
    l2: float = 1e-4
    weights: dict[str, float] = field(default_factory=dict)
    bias: float = 0.0

    def fit(self, rows: list[FeatureRow]) -> "OnlineLogisticRiskPredictor":
        if not rows:
            return self
        keys = sorted({key for row in rows for key in row.features})
        for key in keys:
            self.weights.setdefault(key, 0.0)
        for _ in range(self.epochs):
            for row in rows:
                pred = self.predict_features(row.features)
                error = pred - row.label
                self.bias -= self.learning_rate * error
                for key in keys:
                    value = row.features.get(key, 0.0)
                    grad = error * value + self.l2 * self.weights.get(key, 0.0)
                    self.weights[key] = self.weights.get(key, 0.0) - self.learning_rate * grad
        return self

    def predict_features(self, features: dict[str, float]) -> float:
        score = self.bias
        for key, value in features.items():
            score += self.weights.get(key, 0.0) * value
        return _sigmoid(score)

    def predict_rows(self, rows: list[FeatureRow]) -> list[float]:
        return [self.predict_features(row.features) for row in rows]

    def to_dict(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "l2": self.l2,
            "weights": self.weights,
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OnlineLogisticRiskPredictor":
        return cls(
            learning_rate=float(data.get("learning_rate", 0.05)),
            epochs=int(data.get("epochs", 30)),
            l2=float(data.get("l2", 1e-4)),
            weights={str(k): float(v) for k, v in data.get("weights", {}).items()},
            bias=float(data.get("bias", 0.0)),
        )

