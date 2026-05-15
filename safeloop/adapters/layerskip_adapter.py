from __future__ import annotations

from safeloop.types import DepthStep
from safeloop.adapters.hf_depth_adapter import HFCausalDepthAdapter


class LayerSkipAdapter(HFCausalDepthAdapter):
    """Layer-wise depth adapter for secondary transfer experiments."""

    def _set_depth(self, depth: int) -> None:
        if self._model is None:
            return
        cfg = getattr(self._model, "config", None)
        if cfg is not None:
            for attr in ("exit_layer", "early_exit_layer"):
                if hasattr(cfg, attr):
                    setattr(cfg, attr, depth)

    def _probe_next_token(self, prefix: str, depth: int) -> DepthStep:
        """Probe a layer-wise state when hidden states and lm_head are available."""
        self.load()
        try:
            import torch  # type: ignore
        except Exception as exc:
            raise RuntimeError("Torch is required for LayerSkip probing") from exc
        assert self._model is not None and self._tokenizer is not None
        inputs = self._tokenizer(prefix, return_tensors="pt")
        device = self._infer_input_device()
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states
        layer_idx = min(max(depth, 1), len(hidden_states) - 1)
        hidden_tensor = hidden_states[layer_idx][0, -1]
        if hasattr(self._model, "lm_head"):
            logits_tensor = self._model.lm_head(hidden_tensor).detach().float().cpu()
        else:
            logits_tensor = outputs.logits[0, -1].detach().float().cpu()
        top_id = int(torch.argmax(logits_tensor).item())
        token = self._tokenizer.decode([top_id])
        logits = torch.topk(logits_tensor, k=min(256, logits_tensor.numel())).values.tolist()
        hidden = hidden_tensor.detach().float().cpu()[: min(256, hidden_tensor.numel())].tolist()
        return DepthStep(depth=depth, token_id=top_id, token=token, logits=logits, hidden=hidden)
