from __future__ import annotations

from pathlib import Path

from safeloop.adapters.base import BenchmarkSample
from safeloop.utils.config import load_json


def load_benchmark_config(path: str | Path) -> list[BenchmarkSample]:
    cfg = load_json(path)
    samples = []
    for row in cfg.get("samples", []):
        samples.append(
            BenchmarkSample(
                request_id=str(row["request_id"]),
                prompt=str(row["prompt"]),
                task=str(row.get("task", cfg.get("name", "unknown"))),
                stage=str(row.get("stage", "unknown")),
                expected=str(row.get("expected", "")),
                metadata=dict(row.get("metadata", {})),
            )
        )
    return samples

