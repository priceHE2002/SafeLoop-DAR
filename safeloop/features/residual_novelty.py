from __future__ import annotations

from safeloop.features.depth_attention_probe import CosineDepthAttentionProbe
from safeloop.types import DepthStep
from safeloop.utils.math_utils import EPS, l2_distance, l2_norm


def residual_novelty(
    steps_up_to_t: list[DepthStep],
    probe: CosineDepthAttentionProbe | None = None,
) -> float:
    if len(steps_up_to_t) <= 1:
        return 1.0
    probe = probe or CosineDepthAttentionProbe()
    current = steps_up_to_t[-1]
    previous_readout = probe.attend(steps_up_to_t[:-1]).readout
    return l2_distance(current.hidden, previous_readout) / (l2_norm(current.hidden) + EPS)

