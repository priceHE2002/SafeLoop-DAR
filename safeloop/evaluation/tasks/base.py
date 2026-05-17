from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class TaskEvaluation:
    request_id: str
    depth: int
    task: str
    output: str
    expected: str
    correct: bool
    score: float
    high_cost_error: int = 0
    metric: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TaskEvaluator(Protocol):
    def evaluate(
        self,
        request_id: str,
        task: str,
        stage: str,
        output: str,
        expected: str,
        depth: int,
        metadata: dict[str, Any] | None = None,
    ) -> TaskEvaluation: ...


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def exact_or_contains(output: str, expected: str) -> bool:
    out = normalize_text(output)
    exp = normalize_text(expected)
    return bool(exp) and (out == exp or exp in out)


def detokenize(tokens: list[str]) -> str:
    text = " ".join(token for token in tokens if token is not None)
    text = re.sub(r"\s+([,.;:!?)}\]])", r"\1", text)
    text = re.sub(r"([({\[])\s+", r"\1", text)
    text = re.sub(r"\s*([*/=+\-])\s*", r"\1", text)
    text = text.replace('" ', '"').replace(' "', '"')
    return text.strip()
