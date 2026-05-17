from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any

from safeloop.adapters.base import BenchmarkSample
from safeloop.features.token_stage import risk_group
from safeloop.tracing.tokenization import classify_token, simple_tokenize
from safeloop.utils.config import load_json
from safeloop.utils.io import write_json


def base_required_depth(token_type: str, stage: str, position: int, max_depth: int, seed: int) -> int:
    base = 2
    if token_type in {"punctuation"}:
        base = 1
    elif token_type in {"number", "math_final_number"}:
        base = 5
    elif token_type in {"code_identifier", "json_value", "retrieved_entity"}:
        base = 4
    elif stage in {"tool_call_json", "code_generation", "final_answer"}:
        base = 3
    jitter = (position + seed) % 2
    return min(max_depth, max(1, base + jitter))


def semantic_depth_floor(
    token_type: str,
    stage: str,
    metadata: dict[str, Any],
    max_depth: int,
) -> int:
    """Minimum depth used by controlled halt-aware training.

    This is a controlled-study proxy for semantic difficulty. The paper-level
    experiment should replace it with real task degradation labels.
    """

    group = risk_group(token_type, stage)
    floor_by_group = {
        "low_risk_text": 1,
        "general": 2,
        "entity": 3,
        "math_final": 3,
        "code": 3,
        "tool_json": 3,
    }
    floor = floor_by_group.get(group, 2)
    hops = metadata.get("hops")
    if hops is not None:
        try:
            floor = max(floor, min(max_depth, 1 + int(hops) // 4))
        except (TypeError, ValueError):
            pass
    return min(max_depth, max(1, floor))


@dataclass
class HaltAwareTrainingConfig:
    max_depth: int = 8
    seed: int = 19
    epochs: int = 12
    learning_rate: float = 0.35
    compute_penalty: float = 0.08
    risk_penalty: float = 1.0
    halt_margin: float = 0.25
    mode: str = "halt_aware"


@dataclass
class HaltAwareTrainingState:
    max_depth: int
    seed: int
    mode: str
    depth_adjustment_by_group: dict[str, float] = field(default_factory=dict)
    confidence_bonus_by_group: dict[str, float] = field(default_factory=dict)
    training_curve: list[dict[str, float]] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HaltAwareTrainingState":
        return cls(
            max_depth=int(data.get("max_depth", 8)),
            seed=int(data.get("seed", 19)),
            mode=str(data.get("mode", "halt_aware")),
            depth_adjustment_by_group={
                str(k): float(v) for k, v in data.get("depth_adjustment_by_group", {}).items()
            },
            confidence_bonus_by_group={
                str(k): float(v) for k, v in data.get("confidence_bonus_by_group", {}).items()
            },
            training_curve=[dict(item) for item in data.get("training_curve", [])],
            notes=str(data.get("notes", "")),
        )

    @classmethod
    def load(cls, path: str) -> "HaltAwareTrainingState":
        return cls.from_dict(load_json(path))

    def save(self, path: str) -> None:
        write_json(path, self.to_dict())

    def depth_adjustment(self, group: str) -> float:
        return float(self.depth_adjustment_by_group.get(group, 0.0))

    def confidence_bonus(self, group: str) -> float:
        return float(self.confidence_bonus_by_group.get(group, 0.0))


def _iter_training_tokens(samples: list[BenchmarkSample]):
    for sample in samples:
        tokens = simple_tokenize(sample.expected) or simple_tokenize(sample.prompt)[-1:]
        metadata = dict(sample.metadata or {})
        for position, token in enumerate(tokens):
            token_type = classify_token(token, sample.stage)
            group = risk_group(token_type, sample.stage)
            yield sample, position, token, token_type, group, metadata


def train_halt_aware_state(
    samples: list[BenchmarkSample],
    cfg: HaltAwareTrainingConfig,
) -> HaltAwareTrainingState:
    """Controlled continued-pretraining simulation for LoopTiny.

    Standard mode mainly improves confidence calibration. Halt-aware mode also
    learns a group-wise depth adjustment that pushes recurrent depth down until
    it reaches a semantic floor.
    """

    groups = sorted({group for *_prefix, group, _metadata in _iter_training_tokens(samples)})
    state = HaltAwareTrainingState(
        max_depth=cfg.max_depth,
        seed=cfg.seed,
        mode=cfg.mode,
        depth_adjustment_by_group={group: 0.0 for group in groups},
        confidence_bonus_by_group={group: 0.0 for group in groups},
        notes=(
            "Controlled LoopTiny continued-pretraining state. Use this for "
            "mechanism validation, not as a replacement for real-model training."
        ),
    )
    tokens = list(_iter_training_tokens(samples))
    if not tokens:
        return state

    for epoch in range(1, cfg.epochs + 1):
        losses = []
        depths = []
        violations = []
        for _sample, position, _token, token_type, group, metadata in tokens:
            base_depth = base_required_depth(token_type, _sample.stage, position, cfg.max_depth, cfg.seed)
            floor = semantic_depth_floor(token_type, _sample.stage, metadata, cfg.max_depth)
            current_depth = max(
                1.0,
                min(float(cfg.max_depth), base_depth + state.depth_adjustment(group)),
            )
            risk_violation = max(0.0, floor + cfg.halt_margin - current_depth)
            excess_depth = max(0.0, current_depth - floor)
            if cfg.mode == "halt_aware":
                update = cfg.learning_rate * (
                    cfg.risk_penalty * risk_violation - cfg.compute_penalty * excess_depth
                )
                state.depth_adjustment_by_group[group] += update
            state.confidence_bonus_by_group[group] += cfg.learning_rate * 0.02
            loss = risk_violation * cfg.risk_penalty + excess_depth * cfg.compute_penalty
            losses.append(loss)
            depths.append(current_depth)
            violations.append(1.0 if risk_violation > 0 else 0.0)

        state.training_curve.append(
            {
                "epoch": float(epoch),
                "mean_proxy_loss": mean(losses),
                "mean_predicted_depth": mean(depths),
                "risk_violation_rate": mean(violations),
            }
        )

    return state
