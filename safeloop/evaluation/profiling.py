from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from statistics import mean
from typing import Iterator


@dataclass
class LatencyRecorder:
    values: dict[str, list[float]] = field(default_factory=dict)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        yield
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.values.setdefault(name, []).append(elapsed_ms)

    def add(self, name: str, value_ms: float) -> None:
        self.values.setdefault(name, []).append(float(value_ms))

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            key: {
                "count": float(len(values)),
                "mean_ms": mean(values) if values else 0.0,
                "min_ms": min(values) if values else 0.0,
                "max_ms": max(values) if values else 0.0,
            }
            for key, values in sorted(self.values.items())
        }
