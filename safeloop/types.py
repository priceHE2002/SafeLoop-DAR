from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


JsonDict = dict[str, Any]


@dataclass(slots=True)
class DepthStep:
    """State observed at one depth / recurrent step for one token position."""

    depth: int
    token_id: int
    token: str
    logits: list[float]
    hidden: list[float]
    latency_ms: float = 0.0
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "DepthStep":
        return cls(
            depth=int(data["depth"]),
            token_id=int(data.get("token_id", -1)),
            token=str(data.get("token", "")),
            logits=[float(x) for x in data.get("logits", [])],
            hidden=[float(x) for x in data.get("hidden", [])],
            latency_ms=float(data.get("latency_ms", 0.0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class TokenTrace:
    """All depth states for one generated token position."""

    request_id: str
    model_name: str
    task: str
    position: int
    prefix: str
    target_text: str = ""
    full_depth: int = 0
    stage: str = "unknown"
    token_type: str = "unknown"
    steps: list[DepthStep] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)

    def sorted_steps(self) -> list[DepthStep]:
        return sorted(self.steps, key=lambda step: step.depth)

    def get_step(self, depth: int) -> DepthStep | None:
        for step in self.steps:
            if step.depth == depth:
                return step
        return None

    def full_step(self) -> DepthStep:
        if self.full_depth:
            step = self.get_step(self.full_depth)
            if step is not None:
                return step
        if not self.steps:
            raise ValueError("TokenTrace has no steps")
        return max(self.steps, key=lambda step: step.depth)

    def to_dict(self) -> JsonDict:
        data = asdict(self)
        data["steps"] = [step.to_dict() for step in self.sorted_steps()]
        return data

    @classmethod
    def from_dict(cls, data: JsonDict) -> "TokenTrace":
        return cls(
            request_id=str(data["request_id"]),
            model_name=str(data.get("model_name", "")),
            task=str(data.get("task", "")),
            position=int(data.get("position", 0)),
            prefix=str(data.get("prefix", "")),
            target_text=str(data.get("target_text", "")),
            full_depth=int(data.get("full_depth", 0)),
            stage=str(data.get("stage", "unknown")),
            token_type=str(data.get("token_type", "unknown")),
            steps=[DepthStep.from_dict(step) for step in data.get("steps", [])],
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class FeatureRow:
    request_id: str
    position: int
    depth: int
    task: str
    stage: str
    token_type: str
    features: dict[str, float]
    label: int
    group: str

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class HaltingDecision:
    request_id: str
    position: int
    depth: int
    should_exit: bool
    risk_score: float
    threshold: float
    group: str

    def to_dict(self) -> JsonDict:
        return asdict(self)

