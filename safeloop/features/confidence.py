from __future__ import annotations

from safeloop.types import DepthStep
from safeloop.utils.math_utils import entropy_from_probs, l2_distance, softmax


def confidence_features(step: DepthStep, prev_step: DepthStep | None = None) -> dict[str, float]:
    probs = softmax(step.logits)
    if not probs:
        return {
            "entropy": 0.0,
            "top1_prob": 0.0,
            "top1_top2_margin": 0.0,
            "logit_delta": 0.0,
            "hidden_delta": 0.0,
        }
    sorted_probs = sorted(probs, reverse=True)
    top1 = sorted_probs[0]
    top2 = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    return {
        "entropy": entropy_from_probs(probs),
        "top1_prob": top1,
        "top1_top2_margin": top1 - top2,
        "logit_delta": l2_distance(step.logits, prev_step.logits) if prev_step else 0.0,
        "hidden_delta": l2_distance(step.hidden, prev_step.hidden) if prev_step else 0.0,
    }

