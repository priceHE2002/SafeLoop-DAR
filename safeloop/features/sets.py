from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from safeloop.types import FeatureRow


FEATURE_SETS: dict[str, set[str]] = {
    "entropy_only": {"entropy"},
    "margin_only": {"top1_top2_margin"},
    "top1_probability": {"top1_prob"},
    "confidence_only": {"entropy", "top1_prob", "top1_top2_margin", "logit_delta"},
    "hidden_delta_only": {"hidden_delta", "relative_depth"},
    "hidden_lite": {"hidden_delta", "residual_novelty", "relative_depth"},
    "dynamics_only": {
        "hidden_delta",
        "logit_delta",
        "depth_attention_stability",
        "depth_attention_shift",
        "residual_novelty",
        "relative_depth",
    },
    "dynamics_confidence": {
        "entropy",
        "top1_prob",
        "top1_top2_margin",
        "logit_delta",
        "hidden_delta",
        "depth_attention_stability",
        "depth_attention_shift",
        "residual_novelty",
        "relative_depth",
    },
    "full_with_groups_aux": set(),
}


PRIMARY_FEATURE_SETS = [
    "confidence_only",
    "hidden_delta_only",
    "hidden_lite",
    "dynamics_only",
    "dynamics_confidence",
]


AUXILIARY_FEATURE_SETS = ["full_with_groups_aux"]


def mask_feature_rows(rows: Iterable[FeatureRow], feature_set: str) -> list[FeatureRow]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    allowed = FEATURE_SETS[feature_set]
    if feature_set == "full_with_groups_aux":
        return [replace(row, features=dict(row.features)) for row in rows]
    return [
        replace(row, features={key: value for key, value in row.features.items() if key in allowed})
        for row in rows
    ]


def parse_feature_sets(value: str | list[str] | None) -> list[str]:
    if value is None:
        return PRIMARY_FEATURE_SETS + AUXILIARY_FEATURE_SETS
    if isinstance(value, list):
        requested = [str(item) for item in value]
    else:
        requested = [item.strip() for item in str(value).split(",") if item.strip()]
    unknown = [item for item in requested if item not in FEATURE_SETS]
    if unknown:
        raise ValueError(f"Unknown feature sets: {unknown}")
    return requested
