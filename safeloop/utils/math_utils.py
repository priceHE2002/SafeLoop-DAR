from __future__ import annotations

import math


EPS = 1e-12


def softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []
    max_logit = max(logits)
    exps = [math.exp(x - max_logit) for x in logits]
    denom = sum(exps) + EPS
    return [x / denom for x in exps]


def entropy_from_probs(probs: list[float]) -> float:
    return -sum(p * math.log(max(p, EPS)) for p in probs)


def l2_norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def l2_distance(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(n)))


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    denom = l2_norm(a) * l2_norm(b) + EPS
    return dot(a, b) / denom


def js_divergence(p: list[float], q: list[float]) -> float:
    n = max(len(p), len(q))
    pp = p + [0.0] * (n - len(p))
    qq = q + [0.0] * (n - len(q))
    total_p = sum(pp) + EPS
    total_q = sum(qq) + EPS
    pp = [x / total_p for x in pp]
    qq = [x / total_q for x in qq]
    m = [(x + y) / 2.0 for x, y in zip(pp, qq)]
    return 0.5 * _kl(pp, m) + 0.5 * _kl(qq, m)


def _kl(p: list[float], q: list[float]) -> float:
    return sum(pi * math.log(max(pi, EPS) / max(qi, EPS)) for pi, qi in zip(p, q) if pi > 0)

