from __future__ import annotations

from safeloop.types import TokenTrace


def teacher_consistency_error(trace: TokenTrace, depth: int) -> int:
    step = trace.get_step(depth)
    if step is None:
        raise ValueError(f"Trace {trace.request_id}:{trace.position} has no depth {depth}")
    full = trace.full_step()
    return 0 if step.token == full.token else 1


def high_risk_teacher_error(trace: TokenTrace, depth: int) -> int:
    high_risk = trace.token_type in {
        "math_final_number",
        "number",
        "code_identifier",
        "json_value",
        "tool_argument",
        "retrieved_entity",
        "citation_entity",
    }
    return teacher_consistency_error(trace, depth) if high_risk else 0


def task_degradation_error(trace: TokenTrace, depth: int) -> int:
    """Task-level degradation label when available.

    Real free-generation evaluators can attach a mapping such as
    `metadata["task_degradation_by_depth"] = {"1": 1, "2": 0}`. If it is absent,
    this falls back to teacher consistency so downstream scripts remain usable.
    """

    mapping = trace.metadata.get("task_degradation_by_depth", {})
    if str(depth) in mapping:
        return int(mapping[str(depth)])
    if depth in mapping:
        return int(mapping[depth])
    return teacher_consistency_error(trace, depth)
