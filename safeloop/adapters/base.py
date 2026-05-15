from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from safeloop.types import TokenTrace


@dataclass(slots=True)
class BenchmarkSample:
    request_id: str
    prompt: str
    task: str
    stage: str
    expected: str = ""
    metadata: dict[str, Any] | None = None


class DepthModelAdapter(ABC):
    """Interface for models exposing intermediate depth states."""

    model_name: str
    max_depth: int

    @abstractmethod
    def collect_teacher_forced_trace(
        self,
        sample: BenchmarkSample,
        max_new_tokens: int,
    ) -> list[TokenTrace]:
        raise NotImplementedError

    @abstractmethod
    def collect_free_generation_trace(
        self,
        sample: BenchmarkSample,
        max_new_tokens: int,
    ) -> list[TokenTrace]:
        raise NotImplementedError

