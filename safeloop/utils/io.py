from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from safeloop.types import TokenTrace


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, data: object) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def read_traces(path: str | Path) -> list[TokenTrace]:
    return [TokenTrace.from_dict(row) for row in read_jsonl(path)]


def write_traces(path: str | Path, traces: Iterable[TokenTrace]) -> None:
    write_jsonl(path, (trace.to_dict() for trace in traces))

