from __future__ import annotations

import json
from typing import Any

from safeloop.evaluation.tasks.base import TaskEvaluation


def _execute_mock_tool(payload: dict[str, Any]) -> Any:
    tool = payload.get("tool") or payload.get("name")
    arguments = payload.get("arguments", {})
    if tool == "calculator":
        expression = str(arguments.get("expression", ""))
        allowed = set("0123456789+-*/(). ")
        if not expression or any(char not in allowed for char in expression):
            raise ValueError("unsafe calculator expression")
        return eval(expression, {"__builtins__": {}}, {})  # noqa: S307 - restricted arithmetic only
    return payload


class ToolCallEvaluator:
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
            payload = json.loads(output)
            parsed = True
            result = _execute_mock_tool(payload)
            executed = True
        except Exception:
            payload = None
            parsed = False
            result = None
            executed = False

        expected_result = (metadata or {}).get("expected_result")
        if expected_result is None:
            try:
                expected_payload = json.loads(expected)
                expected_result = _execute_mock_tool(expected_payload)
            except Exception:
                expected_result = None

        result_ok = executed if expected_result is None else str(result) == str(expected_result)
        correct = parsed and executed and result_ok
        return TaskEvaluation(
            request_id=request_id,
            depth=depth,
            task=task,
            output=output,
            expected=expected,
            correct=correct,
            score=1.0 if correct else 0.0,
            high_cost_error=0 if correct else 1,
            metric={"json_valid": parsed, "tool_executed": executed, "result": result, "result_ok": result_ok},
        )
