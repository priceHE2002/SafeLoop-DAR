from __future__ import annotations

import math


def fit_exponential_decay(points: list[tuple[float, float]]) -> dict[str, float]:
    """Fit y = a * exp(-b * d) + c with a small stdlib grid search."""

    if len(points) < 3:
        return {"a": 0.0, "b": 0.0, "c": points[-1][1] if points else 0.0, "mse": 0.0}
    depths = [x for x, _y in points]
    values = [y for _x, y in points]
    min_y = min(values)
    max_y = max(values)
    best = {"a": 0.0, "b": 0.0, "c": min_y, "mse": float("inf")}
    c_candidates = [min_y * ratio for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]]
    c_candidates += [min_y, max(0.0, min_y - 0.05), min(max_y, min_y + 0.05)]
    for c in c_candidates:
        shifted = [max(y - c, 1e-9) for y in values]
        log_y = [math.log(y) for y in shifted]
        mean_x = sum(depths) / len(depths)
        mean_y = sum(log_y) / len(log_y)
        denom = sum((x - mean_x) ** 2 for x in depths) or 1.0
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(depths, log_y)) / denom
        intercept = mean_y - slope * mean_x
        b = max(0.0, -slope)
        a = math.exp(intercept)
        preds = [a * math.exp(-b * x) + c for x in depths]
        mse = sum((pred - y) ** 2 for pred, y in zip(preds, values)) / len(values)
        if mse < best["mse"]:
            best = {"a": a, "b": b, "c": c, "mse": mse}
    return best


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)
