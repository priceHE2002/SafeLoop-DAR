from __future__ import annotations

from dataclasses import dataclass

from safeloop.adapters.base import BenchmarkSample
from safeloop.adapters.mock_adapter import MockLoopAdapter
from safeloop.controlled.halt_aware import HaltAwareTrainingState
from safeloop.features.token_stage import risk_group
from safeloop.tracing.tokenization import classify_token, simple_tokenize
from safeloop.types import DepthStep, TokenTrace


@dataclass
class LoopTinyAdapter(MockLoopAdapter):
    """Controlled LoopTiny-style adapter with optional halt-aware training state.

    Without `training_state_path`, this behaves like the deterministic mock
    looped model. With a state produced by `run_halt_aware_pretraining.py`, it
    emits earlier stable tokens for groups that learned lower recurrent depth.
    """

    training_state: HaltAwareTrainingState | None = None

    @classmethod
    def from_config(cls, cfg: dict) -> "LoopTinyAdapter":
        state = None
        state_path = cfg.get("training_state_path")
        if state_path:
            state = HaltAwareTrainingState.load(str(state_path))
        return cls(
            model_name=str(cfg.get("model_name", "LoopTiny-Controlled")),
            max_depth=int(cfg.get("max_depth", 8)),
            hidden_size=int(cfg.get("hidden_size", 16)),
            vocab_size=int(cfg.get("vocab_size", 64)),
            seed=int(cfg.get("seed", 7)),
            training_state=state,
        )

    def collect_teacher_forced_trace(
        self,
        sample: BenchmarkSample,
        max_new_tokens: int,
    ) -> list[TokenTrace]:
        return self._build_trace(sample, max_new_tokens=max_new_tokens, free_generation=False)

    def collect_free_generation_trace(
        self,
        sample: BenchmarkSample,
        max_new_tokens: int,
    ) -> list[TokenTrace]:
        return self._build_trace(sample, max_new_tokens=max_new_tokens, free_generation=True)

    def _build_trace(
        self,
        sample: BenchmarkSample,
        max_new_tokens: int,
        free_generation: bool,
    ) -> list[TokenTrace]:
        target_tokens = simple_tokenize(sample.expected) or simple_tokenize(sample.prompt)[-max_new_tokens:]
        target_tokens = target_tokens[:max_new_tokens]
        traces: list[TokenTrace] = []
        prefix = sample.prompt
        for pos, final_token in enumerate(target_tokens):
            token_type = classify_token(final_token, stage=sample.stage)
            group = risk_group(token_type, sample.stage)
            required_depth = self._required_depth_for_sample(token_type, sample.stage, pos, group)
            steps = [
                self._step_for_token(final_token, token_type, sample.stage, pos, depth, required_depth)
                for depth in range(1, self.max_depth + 1)
            ]
            trace = TokenTrace(
                request_id=sample.request_id,
                model_name=self.model_name,
                task=sample.task,
                position=pos,
                prefix=prefix,
                target_text=final_token,
                full_depth=self.max_depth,
                stage=sample.stage,
                token_type=token_type,
                steps=steps,
                metadata={
                    "free_generation": free_generation,
                    "required_depth": required_depth,
                    "risk_group": group,
                    "training_mode": self.training_state.mode if self.training_state else "baseline",
                    "sample_metadata": dict(sample.metadata or {}) | {"expected": sample.expected},
                },
            )
            traces.append(trace)
            emitted = self._first_stable_token(steps) if free_generation else final_token
            prefix = f"{prefix} {emitted}".strip()
        return traces

    def _required_depth_for_sample(
        self,
        token_type: str,
        stage: str,
        position: int,
        group: str,
    ) -> int:
        base_depth = super()._required_depth(token_type, stage, position)
        if self.training_state is None:
            return base_depth
        adjusted = round(base_depth + self.training_state.depth_adjustment(group))
        return min(self.max_depth, max(1, adjusted))

    def _logits(self, token_id: int, stable: bool, depth: int, required_depth: int) -> list[float]:
        logits = super()._logits(token_id, stable, depth, required_depth)
        if self.training_state is None:
            return logits
        bonus = 0.0
        # The controlled state stores group bonuses, but `_logits` only sees the
        # selected token. Apply a small global bonus equal to the learned average.
        if self.training_state.confidence_bonus_by_group:
            bonus = sum(self.training_state.confidence_bonus_by_group.values()) / len(
                self.training_state.confidence_bonus_by_group
            )
        logits[token_id % self.vocab_size] += bonus
        return logits
