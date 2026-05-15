from __future__ import annotations

from safeloop.adapters.base import DepthModelAdapter
from safeloop.adapters.layerskip_adapter import LayerSkipAdapter
from safeloop.adapters.loopformer_adapter import LoopFormerAdapter
from safeloop.adapters.looptiny_adapter import LoopTinyAdapter
from safeloop.adapters.mock_adapter import MockLoopAdapter
from safeloop.adapters.ouro_adapter import OuroAdapter


def build_adapter(cfg: dict) -> DepthModelAdapter:
    name = str(cfg.get("adapter", "mock")).lower()
    if name == "mock":
        return MockLoopAdapter.from_config(cfg)
    if name == "ouro":
        return OuroAdapter.from_config(cfg)
    if name == "loopformer":
        return LoopFormerAdapter.from_config(cfg)
    if name == "layerskip":
        return LayerSkipAdapter.from_config(cfg)
    if name == "looptiny":
        return LoopTinyAdapter.from_config(cfg)
    raise ValueError(f"Unknown adapter: {name}")

