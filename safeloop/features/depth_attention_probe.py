from __future__ import annotations

from dataclasses import dataclass

from safeloop.types import DepthStep
from safeloop.utils.math_utils import cosine_similarity, entropy_from_probs, js_divergence, softmax


@dataclass(slots=True)
class DepthAttentionResult:
    weights: list[float]
    readout: list[float]
    entropy: float


class CosineDepthAttentionProbe:
    """A lightweight AttnRes-inspired depth aggregation probe.

    This is a feature extractor for halting. It is not a claim about the base
    model's internal residual mechanism.
    """

    def attend(self, steps: list[DepthStep]) -> DepthAttentionResult:
        if not steps:
            return DepthAttentionResult([], [], 0.0)
        query = steps[-1].hidden
        scores = [cosine_similarity(query, step.hidden) for step in steps]
        weights = softmax(scores)
        width = len(query)
        readout = [0.0] * width
        for weight, step in zip(weights, steps):
            for idx in range(min(width, len(step.hidden))):
                readout[idx] += weight * step.hidden[idx]
        return DepthAttentionResult(weights, readout, entropy_from_probs(weights))


def depth_attention_features(
    steps_up_to_t: list[DepthStep],
    probe: CosineDepthAttentionProbe | None = None,
) -> dict[str, float]:
    probe = probe or CosineDepthAttentionProbe()
    current = probe.attend(steps_up_to_t)
    previous = probe.attend(steps_up_to_t[:-1]) if len(steps_up_to_t) > 1 else None
    if previous is None:
        stability = 0.0
        shift = 0.0
    else:
        shift = js_divergence(current.weights, previous.weights)
        stability = 1.0 / (1.0 + shift)
    recent_mass = current.weights[-1] if current.weights else 0.0
    return {
        "depth_attention_entropy": current.entropy,
        "depth_attention_stability": stability,
        "depth_attention_shift": shift,
        "depth_attention_recent_mass": recent_mass,
    }

