from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    if base is None:
        return raw
    return Path(base) / raw


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    cfg = load_json(config_path)
    root = config_path.parent.parent.parent if "configs" in config_path.parts else Path.cwd()
    cfg["_config_path"] = str(config_path)
    cfg["_project_root"] = str(root)
    return cfg

