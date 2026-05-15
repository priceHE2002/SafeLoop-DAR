from __future__ import annotations

from safeloop.adapters.hf_depth_adapter import HFCausalDepthAdapter


class LoopFormerAdapter(HFCausalDepthAdapter):
    """LoopFormer adapter placeholder built on the generic HF probing path."""

    def _set_depth(self, depth: int) -> None:
        super()._set_depth(depth)
        if self._model is None:
            return
        cfg = getattr(self._model, "config", None)
        if cfg is not None:
            for attr in ("num_iterations", "n_iterations", "loop_iterations", "max_iterations"):
                if hasattr(cfg, attr):
                    setattr(cfg, attr, depth)

