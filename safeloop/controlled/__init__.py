"""Controlled recurrent-depth experiments."""

from safeloop.controlled.halt_aware import (
    HaltAwareTrainingConfig,
    HaltAwareTrainingState,
    train_halt_aware_state,
)
from safeloop.controlled.tiny_loop_lm import (
    TinyLoopLMCheckpoint,
    TinyLoopLMConfig,
    train_tiny_loop_lm,
)

__all__ = [
    "HaltAwareTrainingConfig",
    "HaltAwareTrainingState",
    "TinyLoopLMCheckpoint",
    "TinyLoopLMConfig",
    "train_halt_aware_state",
    "train_tiny_loop_lm",
]
