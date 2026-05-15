from __future__ import annotations

from safeloop.adapters.base import BenchmarkSample, DepthModelAdapter
from safeloop.types import TokenTrace


class TraceCollector:
    def __init__(self, adapter: DepthModelAdapter) -> None:
        self.adapter = adapter

    def collect(
        self,
        samples: list[BenchmarkSample],
        mode: str,
        max_new_tokens: int,
    ) -> list[TokenTrace]:
        traces: list[TokenTrace] = []
        for sample in samples:
            if mode == "teacher_forced":
                traces.extend(self.adapter.collect_teacher_forced_trace(sample, max_new_tokens))
            elif mode == "free_generation":
                traces.extend(self.adapter.collect_free_generation_trace(sample, max_new_tokens))
            else:
                raise ValueError(f"Unsupported trace mode: {mode}")
        return traces

