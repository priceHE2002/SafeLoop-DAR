from __future__ import annotations

import math
import random
import hashlib
from dataclasses import dataclass

from safeloop.adapters.base import BenchmarkSample, DepthModelAdapter
from safeloop.tracing.tokenization import classify_token, simple_tokenize
from safeloop.types import DepthStep, TokenTrace


@dataclass
class MockLoopAdapter(DepthModelAdapter):
    """Deterministic looped-LM simulator for pipeline development.

    The simulator assigns high-risk tokens larger required depths and produces
    hidden states that converge as depth approaches the required depth.
    """

    model_name: str = "MockLoop-Deterministic"
    max_depth: int = 8
    hidden_size: int = 16
    vocab_size: int = 64
    seed: int = 7

    @classmethod
    def from_config(cls, cfg: dict) -> "MockLoopAdapter":
        return cls(
            model_name=str(cfg.get("model_name", "MockLoop-Deterministic")),
            max_depth=int(cfg.get("max_depth", 8)),
            hidden_size=int(cfg.get("hidden_size", 16)),
            vocab_size=int(cfg.get("vocab_size", 64)),
            seed=int(cfg.get("seed", 7)),
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
            required_depth = self._required_depth(token_type, sample.stage, pos)
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
                    "sample_metadata": dict(sample.metadata or {}) | {"expected": sample.expected},
                },
            )
            traces.append(trace)
            emitted = self._first_stable_token(steps) if free_generation else final_token
            prefix = f"{prefix} {emitted}".strip()
        return traces

    def _required_depth(self, token_type: str, stage: str, position: int) -> int:
        base = 2
        if token_type in {"punctuation"}:
            base = 1
        elif token_type in {"number", "math_final_number"}:
            base = 5
        elif token_type in {"code_identifier", "json_value", "retrieved_entity"}:
            base = 4
        elif stage in {"tool_call_json", "code_generation", "final_answer"}:
            base = 3
        jitter = (position + self.seed) % 2
        return min(self.max_depth, max(1, base + jitter))

    def _step_for_token(
        self,
        final_token: str,
        token_type: str,
        stage: str,
        position: int,
        depth: int,
        required_depth: int,
    ) -> DepthStep:
        stable = depth >= required_depth
        emitted = final_token if stable else self._wrong_token(final_token, token_type, depth)
        token_id = self._token_id(emitted)
        logits = self._logits(token_id=token_id, stable=stable, depth=depth, required_depth=required_depth)
        hidden = self._hidden(final_token, stage, position, depth, required_depth)
        return DepthStep(
            depth=depth,
            token_id=token_id,
            token=emitted,
            logits=logits,
            hidden=hidden,
            latency_ms=0.2 + 0.03 * depth,
            metadata={"stable": stable, "required_depth": required_depth},
        )

    def _wrong_token(self, final_token: str, token_type: str, depth: int) -> str:
        if token_type in {"number", "math_final_number"}:
            try:
                return str(int(final_token) + (1 if depth % 2 else -1))
            except ValueError:
                return "0"
        if token_type == "punctuation":
            return final_token
        if final_token.lower() == final_token:
            return final_token[::-1] or "x"
        return "UNK"

    def _logits(self, token_id: int, stable: bool, depth: int, required_depth: int) -> list[float]:
        logits = [-4.0] * self.vocab_size
        confidence = 2.0 + 0.6 * depth
        if stable:
            confidence += 2.0
        else:
            confidence -= 0.3 * max(required_depth - depth, 0)
        logits[token_id % self.vocab_size] = confidence
        logits[(token_id + 1) % self.vocab_size] = confidence - (1.2 if stable else 0.2)
        return logits

    def _hidden(
        self,
        final_token: str,
        stage: str,
        position: int,
        depth: int,
        required_depth: int,
    ) -> list[float]:
        rng = random.Random(f"{self.seed}:{final_token}:{stage}:{position}")
        target = [rng.uniform(-1.0, 1.0) for _ in range(self.hidden_size)]
        progress = min(depth / max(required_depth, 1), 1.0)
        oscillation = max(required_depth - depth, 0) / max(required_depth, 1)
        return [
            progress * value + oscillation * math.sin((idx + 1) * (depth + 1))
            for idx, value in enumerate(target)
        ]

    def _token_id(self, token: str) -> int:
        digest = hashlib.sha256(f"{self.seed}:{token}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % self.vocab_size

    def _first_stable_token(self, steps: list[DepthStep]) -> str:
        for step in steps:
            if step.metadata.get("stable"):
                return step.token
        return steps[-1].token
