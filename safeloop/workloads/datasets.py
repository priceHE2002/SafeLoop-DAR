from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from safeloop.adapters.base import BenchmarkSample
from safeloop.utils.config import load_json


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def _read_source_rows(path: Path, fmt: str, data_key: str = "data") -> list[dict[str, Any]]:
    if fmt == "jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    if fmt == "json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [dict(item) for item in data]
        return [dict(item) for item in data.get(data_key, [])]
    if fmt == "csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError(f"Unsupported benchmark source format: {fmt}")


def _field(row: dict[str, Any], field_map: dict[str, str], name: str, default: str = "") -> str:
    key = field_map.get(name, name)
    return str(row.get(key, default))


def _sample_from_source_row(row: dict[str, Any], cfg: dict[str, Any], idx: int) -> BenchmarkSample:
    field_map = dict(cfg.get("field_map", {}))
    template = str(cfg.get("prompt_template", "{prompt}"))
    prompt = template.format_map(_SafeFormatDict({key: str(value) for key, value in row.items()}))
    request_id = _field(row, field_map, "request_id", f"{cfg.get('name', 'sample')}_{idx:06d}")
    expected = _field(row, field_map, "expected", "")
    task = _field(row, field_map, "task", str(cfg.get("task", cfg.get("name", "unknown"))))
    stage = _field(row, field_map, "stage", str(cfg.get("stage", "unknown")))
    metadata = dict(cfg.get("metadata", {}))
    metadata_fields = cfg.get("metadata_fields")
    if metadata_fields:
        for field in metadata_fields:
            if field in row:
                metadata[str(field)] = row[field]
    else:
        metadata["source_row"] = row
    return BenchmarkSample(
        request_id=request_id,
        prompt=prompt,
        task=task,
        stage=stage,
        expected=expected,
        metadata=metadata,
    )


def load_benchmark_config(path: str | Path) -> list[BenchmarkSample]:
    path = Path(path)
    cfg = load_json(path)
    if cfg.get("generator") == "synthetic_khop":
        from safeloop.workloads.synthetic import make_khop_samples

        return make_khop_samples(
            num_samples=int(cfg.get("num_samples", 32)),
            max_hops=int(cfg.get("max_hops", 8)),
        )
    if cfg.get("generator") == "synthetic_mixed_control":
        from safeloop.workloads.synthetic import make_mixed_control_samples

        return make_mixed_control_samples(num_repeats=int(cfg.get("num_repeats", 12)))

    if cfg.get("source_path"):
        source_path = Path(str(cfg["source_path"]))
        if not source_path.is_absolute():
            source_path = path.parent / source_path
        fmt = str(cfg.get("format", source_path.suffix.lstrip(".") or "jsonl")).lower()
        rows = _read_source_rows(source_path, fmt, data_key=str(cfg.get("data_key", "data")))
        limit = cfg.get("limit")
        if limit is not None:
            rows = rows[: int(limit)]
        return [_sample_from_source_row(row, cfg, idx) for idx, row in enumerate(rows)]

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
