from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any

from safeloop.adapters.base import BenchmarkSample
from safeloop.controlled.halt_aware import base_required_depth, semantic_depth_floor
from safeloop.features.token_stage import risk_group
from safeloop.tracing.tokenization import classify_token, simple_tokenize
from safeloop.types import DepthStep, TokenTrace
from safeloop.utils.config import load_json
from safeloop.utils.io import write_json


def _stable_float(key: str, index: int) -> float:
    digest = hashlib.sha256(f"{key}:{index}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return 2.0 * value - 1.0


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _mix(left: list[float], right: list[float], weight: float) -> list[float]:
    return [(1.0 - weight) * a + weight * b for a, b in zip(left, right)]


def _token_vector(token: str, hidden_size: int, seed: int) -> list[float]:
    return _normalize([_stable_float(f"{seed}:token:{token}", idx) for idx in range(hidden_size)])


def _context_vector(sample: BenchmarkSample, position: int, hidden_size: int, seed: int) -> list[float]:
    key = f"{seed}:context:{sample.task}:{sample.stage}:{sample.prompt}:{position}"
    return _normalize([_stable_float(key, idx) for idx in range(hidden_size)])


def _wrong_token(final_token: str, vocab: list[str], depth: int) -> str:
    if final_token.lstrip("-").isdigit():
        return str(int(final_token) + (1 if depth % 2 else -1))
    if len(vocab) <= 1:
        return final_token[::-1] or "UNK"
    digest = hashlib.sha256(f"{final_token}:{depth}".encode("utf-8")).hexdigest()
    index = (int(digest[:8], 16) % (len(vocab) - 1)) + 1
    candidate = vocab[index % len(vocab)]
    if candidate == final_token:
        candidate = vocab[(index + 1) % len(vocab)]
    return candidate


@dataclass
class TinyLoopLMConfig:
    model_name: str = "TinyLoopLM"
    max_depth: int = 8
    hidden_size: int = 24
    seed: int = 23
    epochs: int = 20
    learning_rate: float = 0.25
    compute_penalty: float = 0.08
    risk_penalty: float = 1.0
    halt_margin: float = 0.25
    halt_aware: bool = False


@dataclass
class TinyLoopLMCheckpoint:
    model_name: str
    max_depth: int
    hidden_size: int
    seed: int
    halt_aware: bool
    vocab: list[str]
    token_vectors: dict[str, list[float]]
    group_depth_offset: dict[str, float] = field(default_factory=dict)
    group_halt_bias: dict[str, float] = field(default_factory=dict)
    training_curve: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TinyLoopLMCheckpoint":
        return cls(
            model_name=str(data.get("model_name", "TinyLoopLM")),
            max_depth=int(data.get("max_depth", 8)),
            hidden_size=int(data.get("hidden_size", 24)),
            seed=int(data.get("seed", 23)),
            halt_aware=bool(data.get("halt_aware", False)),
            vocab=[str(item) for item in data.get("vocab", [])],
            token_vectors={
                str(token): [float(value) for value in vector]
                for token, vector in data.get("token_vectors", {}).items()
            },
            group_depth_offset={
                str(group): float(value) for group, value in data.get("group_depth_offset", {}).items()
            },
            group_halt_bias={
                str(group): float(value) for group, value in data.get("group_halt_bias", {}).items()
            },
            training_curve=[dict(item) for item in data.get("training_curve", [])],
        )

    @classmethod
    def load(cls, path: str) -> "TinyLoopLMCheckpoint":
        return cls.from_dict(load_json(path))

    def save(self, path: str) -> None:
        write_json(path, self.to_dict())


def _iter_tokens(samples: list[BenchmarkSample]):
    for sample in samples:
        tokens = simple_tokenize(sample.expected) or simple_tokenize(sample.prompt)[-1:]
        for position, token in enumerate(tokens):
            token_type = classify_token(token, sample.stage)
            group = risk_group(token_type, sample.stage)
            yield sample, position, token, token_type, group


def build_tiny_loop_vocab(samples: list[BenchmarkSample]) -> list[str]:
    vocab = {"UNK"}
    for _sample, _position, token, _token_type, _group in _iter_tokens(samples):
        vocab.add(token)
        if token.lstrip("-").isdigit():
            vocab.add(str(int(token) + 1))
            vocab.add(str(int(token) - 1))
        else:
            vocab.add(token[::-1] or "UNK")
    return sorted(vocab)


def train_tiny_loop_lm(
    samples: list[BenchmarkSample],
    cfg: TinyLoopLMConfig,
) -> TinyLoopLMCheckpoint:
    """Train a tiny recurrent-depth LM for controlled experiments.

    The model is intentionally small and stdlib-only. It learns token prototype
    vectors and, when `halt_aware=True`, group-wise recurrent-depth offsets that
    trade off compute against semantic-risk floors.
    """

    vocab = build_tiny_loop_vocab(samples)
    token_vectors = {token: _token_vector(token, cfg.hidden_size, cfg.seed) for token in vocab}
    groups = sorted({group for *_prefix, group in _iter_tokens(samples)})
    checkpoint = TinyLoopLMCheckpoint(
        model_name=cfg.model_name,
        max_depth=cfg.max_depth,
        hidden_size=cfg.hidden_size,
        seed=cfg.seed,
        halt_aware=cfg.halt_aware,
        vocab=vocab,
        token_vectors=token_vectors,
        group_depth_offset={group: 0.0 for group in groups},
        group_halt_bias={group: 0.0 for group in groups},
    )
    tokens = list(_iter_tokens(samples))
    if not tokens:
        return checkpoint

    for epoch in range(1, cfg.epochs + 1):
        token_losses = []
        compute_losses = []
        risk_violations = []
        depths = []
        for sample, position, token, token_type, group in tokens:
            metadata = dict(sample.metadata or {})
            floor = semantic_depth_floor(token_type, sample.stage, metadata, cfg.max_depth)
            base_depth = base_required_depth(token_type, sample.stage, position, cfg.max_depth, cfg.seed)
            predicted_depth = max(
                1.0,
                min(float(cfg.max_depth), base_depth + checkpoint.group_depth_offset.get(group, 0.0)),
            )
            excess_depth = max(0.0, predicted_depth - floor)
            risk_violation = max(0.0, floor + cfg.halt_margin - predicted_depth)
            if cfg.halt_aware:
                checkpoint.group_depth_offset[group] += cfg.learning_rate * (
                    cfg.risk_penalty * risk_violation - cfg.compute_penalty * excess_depth
                )
                checkpoint.group_halt_bias[group] += cfg.learning_rate * (
                    0.05 * excess_depth - 0.15 * risk_violation
                )
            else:
                checkpoint.group_halt_bias[group] += cfg.learning_rate * 0.01

            context = _context_vector(sample, position, cfg.hidden_size, cfg.seed)
            target = token_vectors[token]
            trained = _normalize(_mix(target, context, 0.03 if cfg.halt_aware else 0.05))
            token_vectors[token] = trained
            token_loss = 1.0 - _dot(trained, target)
            token_losses.append(token_loss)
            compute_losses.append(excess_depth)
            risk_violations.append(1.0 if risk_violation > 0 else 0.0)
            depths.append(predicted_depth)

        checkpoint.training_curve.append(
            {
                "epoch": float(epoch),
                "token_proxy_loss": mean(token_losses),
                "mean_compute_excess": mean(compute_losses),
                "risk_violation_rate": mean(risk_violations),
                "mean_predicted_depth": mean(depths),
            }
        )
    return checkpoint


def required_depth_for_token(
    checkpoint: TinyLoopLMCheckpoint,
    token_type: str,
    stage: str,
    position: int,
    group: str,
    metadata: dict[str, Any],
) -> int:
    base = base_required_depth(token_type, stage, position, checkpoint.max_depth, checkpoint.seed)
    adjusted = round(base + checkpoint.group_depth_offset.get(group, 0.0))
    floor = semantic_depth_floor(token_type, stage, metadata, checkpoint.max_depth)
    if checkpoint.halt_aware:
        adjusted = max(adjusted, floor)
    return min(checkpoint.max_depth, max(1, adjusted))


def depth_step_for_token(
    checkpoint: TinyLoopLMCheckpoint,
    sample: BenchmarkSample,
    position: int,
    final_token: str,
    depth: int,
    required_depth: int,
) -> DepthStep:
    token_type = classify_token(final_token, sample.stage)
    group = risk_group(token_type, sample.stage)
    stable = depth >= required_depth
    emitted = final_token if stable else _wrong_token(final_token, checkpoint.vocab, depth)
    target_vector = checkpoint.token_vectors.get(final_token) or _token_vector(
        final_token, checkpoint.hidden_size, checkpoint.seed
    )
    context = _context_vector(sample, position, checkpoint.hidden_size, checkpoint.seed)
    progress = min(depth / max(required_depth, 1), 1.0)
    hidden = _normalize(_mix(context, target_vector, progress))
    scores = []
    for token in checkpoint.vocab:
        prototype = checkpoint.token_vectors.get(token) or _token_vector(
            token, checkpoint.hidden_size, checkpoint.seed
        )
        score = 4.0 * _dot(hidden, prototype)
        if token == emitted:
            score += 1.0 + 0.4 * depth
        if stable and token == final_token:
            score += 2.0 + checkpoint.group_halt_bias.get(group, 0.0)
        scores.append(score)
    token_id = checkpoint.vocab.index(emitted) if emitted in checkpoint.vocab else 0
    return DepthStep(
        depth=depth,
        token_id=token_id,
        token=emitted,
        logits=scores,
        hidden=hidden,
        latency_ms=0.18 + 0.025 * depth,
        metadata={
            "stable": stable,
            "required_depth": required_depth,
            "tiny_loop_group": group,
            "halt_aware": checkpoint.halt_aware,
        },
    )


def collect_tiny_loop_trace(
    checkpoint: TinyLoopLMCheckpoint,
    sample: BenchmarkSample,
    max_new_tokens: int,
    free_generation: bool = False,
) -> list[TokenTrace]:
    tokens = simple_tokenize(sample.expected) or simple_tokenize(sample.prompt)[-max_new_tokens:]
    tokens = tokens[:max_new_tokens]
    traces: list[TokenTrace] = []
    prefix = sample.prompt
    metadata = dict(sample.metadata or {})
    for position, final_token in enumerate(tokens):
        token_type = classify_token(final_token, sample.stage)
        group = risk_group(token_type, sample.stage)
        required_depth = required_depth_for_token(
            checkpoint, token_type, sample.stage, position, group, metadata
        )
        steps = [
            depth_step_for_token(checkpoint, sample, position, final_token, depth, required_depth)
            for depth in range(1, checkpoint.max_depth + 1)
        ]
        trace = TokenTrace(
            request_id=sample.request_id,
            model_name=checkpoint.model_name,
            task=sample.task,
            position=position,
            prefix=prefix,
            target_text=final_token,
            full_depth=checkpoint.max_depth,
            stage=sample.stage,
            token_type=token_type,
            steps=steps,
            metadata={
                "free_generation": free_generation,
                "required_depth": required_depth,
                "risk_group": group,
                "training_mode": "halt_aware" if checkpoint.halt_aware else "standard",
                "sample_metadata": metadata | {"expected": sample.expected},
            },
        )
        traces.append(trace)
        emitted = next((step.token for step in steps if step.metadata.get("stable")), steps[-1].token)
        prefix = f"{prefix} {emitted if free_generation else final_token}".strip()
    return traces
