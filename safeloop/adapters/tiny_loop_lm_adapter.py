from __future__ import annotations

from dataclasses import dataclass

from safeloop.adapters.base import BenchmarkSample, DepthModelAdapter
from safeloop.controlled.tiny_loop_lm import TinyLoopLMCheckpoint, collect_tiny_loop_trace
from safeloop.types import TokenTrace


@dataclass
class TinyLoopLMAdapter(DepthModelAdapter):
    """Adapter for a locally trained tiny looped language model."""

    checkpoint: TinyLoopLMCheckpoint

    @property
    def model_name(self) -> str:
        return self.checkpoint.model_name

    @property
    def max_depth(self) -> int:
        return self.checkpoint.max_depth

    @classmethod
    def from_config(cls, cfg: dict) -> "TinyLoopLMAdapter":
        checkpoint_path = cfg.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError("tiny_loop_lm adapter requires checkpoint_path")
        return cls(checkpoint=TinyLoopLMCheckpoint.load(str(checkpoint_path)))

    def collect_teacher_forced_trace(
        self,
        sample: BenchmarkSample,
        max_new_tokens: int,
    ) -> list[TokenTrace]:
        return collect_tiny_loop_trace(
            self.checkpoint,
            sample,
            max_new_tokens=max_new_tokens,
            free_generation=False,
        )

    def collect_free_generation_trace(
        self,
        sample: BenchmarkSample,
        max_new_tokens: int,
    ) -> list[TokenTrace]:
        return collect_tiny_loop_trace(
            self.checkpoint,
            sample,
            max_new_tokens=max_new_tokens,
            free_generation=True,
        )
