from __future__ import annotations

from safeloop.features.confidence import confidence_features
from safeloop.features.depth_attention_probe import CosineDepthAttentionProbe, depth_attention_features
from safeloop.features.residual_novelty import residual_novelty
from safeloop.features.token_stage import risk_group, token_stage_features
from safeloop.risk.labels import (
    high_risk_teacher_error,
    task_degradation_error,
    teacher_consistency_error,
)
from safeloop.types import FeatureRow, TokenTrace


class FeatureBuilder:
    def __init__(self) -> None:
        self.probe = CosineDepthAttentionProbe()

    def build_rows(self, trace: TokenTrace, label_type: str = "teacher_consistency") -> list[FeatureRow]:
        steps = trace.sorted_steps()
        rows: list[FeatureRow] = []
        for idx, step in enumerate(steps):
            prev = steps[idx - 1] if idx > 0 else None
            prefix_steps = steps[: idx + 1]
            features: dict[str, float] = {}
            features.update(confidence_features(step, prev))
            features.update(depth_attention_features(prefix_steps, self.probe))
            features["residual_novelty"] = residual_novelty(prefix_steps, self.probe)
            features["relative_depth"] = step.depth / max(trace.full_depth or len(steps), 1)
            features.update(token_stage_features(trace.token_type, trace.stage))
            if label_type == "teacher_consistency":
                label = teacher_consistency_error(trace, step.depth)
            elif label_type == "high_risk_teacher":
                label = high_risk_teacher_error(trace, step.depth)
            elif label_type == "task_degradation":
                label = task_degradation_error(trace, step.depth)
            else:
                raise ValueError(f"Unsupported label_type for FeatureBuilder: {label_type}")
            rows.append(
                FeatureRow(
                    request_id=trace.request_id,
                    position=trace.position,
                    depth=step.depth,
                    task=trace.task,
                    stage=trace.stage,
                    token_type=trace.token_type,
                    features=features,
                    label=label,
                    group=risk_group(trace.token_type, trace.stage),
                )
            )
        return rows

    def build_many(self, traces: list[TokenTrace], label_type: str = "teacher_consistency") -> list[FeatureRow]:
        rows: list[FeatureRow] = []
        for trace in traces:
            rows.extend(self.build_rows(trace, label_type=label_type))
        return rows
