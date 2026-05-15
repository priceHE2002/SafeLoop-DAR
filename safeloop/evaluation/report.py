from __future__ import annotations

from safeloop.evaluation.metrics import brier_score, expected_calibration_error, roc_auc
from safeloop.types import FeatureRow


def prediction_report(rows: list[FeatureRow], scores: list[float]) -> dict[str, float]:
    labels = [row.label for row in rows]
    return {
        "examples": float(len(rows)),
        "label_rate": sum(labels) / max(len(labels), 1),
        "auroc": roc_auc(labels, scores),
        "brier": brier_score(labels, scores),
        "ece": expected_calibration_error(labels, scores),
        "avg_score": sum(scores) / max(len(scores), 1),
    }
