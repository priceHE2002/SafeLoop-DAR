from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from safeloop.adapters.base import BenchmarkSample, DepthModelAdapter
from safeloop.tracing.tokenization import classify_token, simple_tokenize
from safeloop.types import DepthStep, TokenTrace


@dataclass
class HFCausalDepthAdapter(DepthModelAdapter):
    """Best-effort adapter for Hugging Face causal LMs with intermediate states.

    The adapter is intentionally conservative. Different looped models expose
    depth controls differently, so this class provides a shared loading path and
    a simple fixed-depth probing strategy. Model-specific adapters can override
    `_set_depth`.
    """

    model_name: str
    max_depth: int
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    revision: str | None = None
    _model: Any = None
    _tokenizer: Any = None

    @classmethod
    def from_config(cls, cfg: dict) -> "HFCausalDepthAdapter":
        return cls(
            model_name=str(cfg["model_name"]),
            max_depth=int(cfg.get("max_depth", 4)),
            trust_remote_code=bool(cfg.get("trust_remote_code", True)),
            torch_dtype=str(cfg.get("torch_dtype", "bfloat16")),
            device_map=str(cfg.get("device_map", "auto")),
            revision=cfg.get("revision"),
        )

    def load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as exc:
            raise RuntimeError("Install the 'model' extras to use HF adapters") from exc
        dtype = getattr(torch, self.torch_dtype, torch.bfloat16)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=dtype,
            device_map=self.device_map,
            revision=self.revision,
        )
        self._model.eval()

    def collect_teacher_forced_trace(
        self,
        sample: BenchmarkSample,
        max_new_tokens: int,
    ) -> list[TokenTrace]:
        self.load()
        target_tokens = simple_tokenize(sample.expected)[:max_new_tokens]
        if not target_tokens:
            target_tokens = [""] * max_new_tokens
        traces: list[TokenTrace] = []
        prefix = sample.prompt
        for pos, target_token in enumerate(target_tokens):
            token_type = classify_token(target_token, sample.stage)
            steps = [self._probe_next_token(prefix, depth) for depth in range(1, self.max_depth + 1)]
            traces.append(
                TokenTrace(
                    request_id=sample.request_id,
                    model_name=self.model_name,
                    task=sample.task,
                    position=pos,
                    prefix=prefix,
                    target_text=target_token,
                    full_depth=self.max_depth,
                    stage=sample.stage,
                    token_type=token_type,
                    steps=steps,
                    metadata={"adapter": self.__class__.__name__},
                )
            )
            prefix = f"{prefix} {target_token}".strip()
        return traces

    def collect_free_generation_trace(
        self,
        sample: BenchmarkSample,
        max_new_tokens: int,
    ) -> list[TokenTrace]:
        self.load()
        traces: list[TokenTrace] = []
        prefix = sample.prompt
        for pos in range(max_new_tokens):
            steps = [self._probe_next_token(prefix, depth) for depth in range(1, self.max_depth + 1)]
            full_step = steps[-1]
            token_type = classify_token(full_step.token, sample.stage)
            traces.append(
                TokenTrace(
                    request_id=sample.request_id,
                    model_name=self.model_name,
                    task=sample.task,
                    position=pos,
                    prefix=prefix,
                    target_text=full_step.token,
                    full_depth=self.max_depth,
                    stage=sample.stage,
                    token_type=token_type,
                    steps=steps,
                    metadata={"adapter": self.__class__.__name__, "free_generation": True},
                )
            )
            prefix = f"{prefix}{full_step.token}"
            eos_id = getattr(self._tokenizer, "eos_token_id", None)
            if eos_id is not None and full_step.token_id == eos_id:
                break
        return traces

    def _set_depth(self, depth: int) -> None:
        if self._model is None:
            return
        cfg = getattr(self._model, "config", None)
        if cfg is not None:
            if hasattr(cfg, "total_ut_steps"):
                setattr(cfg, "total_ut_steps", depth)
            if hasattr(cfg, "early_exit_threshold"):
                setattr(cfg, "early_exit_threshold", 1.0)

    def _probe_next_token(self, prefix: str, depth: int) -> DepthStep:
        self._set_depth(depth)
        try:
            import torch  # type: ignore
        except Exception as exc:
            raise RuntimeError("Torch is required for HF probing") from exc
        assert self._model is not None and self._tokenizer is not None
        inputs = self._tokenizer(prefix, return_tensors="pt")
        device = self._infer_input_device()
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True, use_cache=False)
        logits_tensor = outputs.logits[0, -1].detach().float().cpu()
        top_id = int(torch.argmax(logits_tensor).item())
        token = self._tokenizer.decode([top_id])
        logits = torch.topk(logits_tensor, k=min(256, logits_tensor.numel())).values.tolist()
        hidden_tensor = outputs.hidden_states[-1][0, -1].detach().float().cpu()
        hidden = hidden_tensor[: min(256, hidden_tensor.numel())].tolist()
        return DepthStep(depth=depth, token_id=top_id, token=token, logits=logits, hidden=hidden)

    def _infer_input_device(self):
        if self._model is None:
            return None
        device = getattr(self._model, "device", None)
        if device is not None:
            return device
        try:
            return next(self._model.parameters()).device
        except Exception:
            return None
