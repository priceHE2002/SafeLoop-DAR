from __future__ import annotations

from typing import Any

from safeloop.evaluation.tasks.base import TaskEvaluation, exact_or_contains


class HumanEvalEvaluator:
    """Conservative code evaluator.

    The full HumanEval pass@1 protocol should run in a sandboxed process. This
    evaluator provides a safe default for pipeline integration: syntax validity
    plus optional expected substring matching.
    """

    def evaluate(
        self,
        request_id: str,
        task: str,
        stage: str,
        output: str,
        expected: str,
        depth: int,
        metadata: dict[str, Any] | None = None,
    ) -> TaskEvaluation:
        try:
            compile(output, f"<{request_id}:{depth}>", "exec")
            syntax_ok = True
        except SyntaxError:
            syntax_ok = False
        expected_ok = exact_or_contains(output, expected) if expected else True
        correct = syntax_ok and expected_ok
        return TaskEvaluation(
            request_id=request_id,
            depth=depth,
            task=task,
            output=output,
            expected=expected,
            correct=correct,
            score=1.0 if correct else 0.0,
            high_cost_error=0 if correct else 1,
            metric={"syntax_ok": syntax_ok, "expected_substring_ok": expected_ok, "pass_at_1": float(correct)},
        )
