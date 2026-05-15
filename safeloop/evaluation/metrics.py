from __future__ import annotations

from safeloop.types import FeatureRow


def binary_metrics(labels: list[int], scores: list[float], threshold: float = 0.5) -> dict[str, float]:
    preds = [1 if score >= threshold else 0 for score in scores]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    total = max(len(labels), 1)
    return {
        "accuracy": (tp + tn) / total,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "positive_rate": sum(preds) / total,
        "label_rate": sum(labels) / total,
    }


def brier_score(labels: list[int], scores: list[float]) -> float:
    if not labels:
        return 0.0
    return sum((score - label) ** 2 for label, score in zip(labels, scores)) / len(labels)


def roc_auc(labels: list[int], scores: list[float]) -> float:
    positives = [(s, y) for y, s in zip(labels, scores) if y == 1]
    negatives = [(s, y) for y, s in zip(labels, scores) if y == 0]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for ps, _ in positives:
        for ns, _ in negatives:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def expected_calibration_error(labels: list[int], scores: list[float], bins: int = 10) -> float:
    if not labels:
        return 0.0
    ece = 0.0
    n = len(labels)
    for idx in range(bins):
        lo = idx / bins
        hi = (idx + 1) / bins
        bucket = [(y, s) for y, s in zip(labels, scores) if lo <= s < hi or (idx == bins - 1 and s == hi)]
        if not bucket:
            continue
        avg_conf = sum(s for _, s in bucket) / len(bucket)
        avg_label = sum(y for y, _ in bucket) / len(bucket)
        ece += len(bucket) / n * abs(avg_conf - avg_label)
    return ece


def risk_by_group(rows: list[FeatureRow], accepted: list[bool]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[tuple[int, bool]]] = {}
    for row, keep in zip(rows, accepted):
        groups.setdefault(row.group, []).append((row.label, keep))
    out: dict[str, dict[str, float]] = {}
    for group, values in groups.items():
        chosen = [label for label, keep in values if keep]
        out[group] = {
            "coverage": len(chosen) / max(len(values), 1),
            "risk": sum(chosen) / max(len(chosen), 1),
            "count": float(len(values)),
        }
    return out
