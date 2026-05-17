from __future__ import annotations

import json
from typing import Any

from safeloop.evaluation.tasks.base import TaskEvaluation


def _loads_json(text: str) -> Any:
    return json.loads(text)


class JSONSchemaEvaluator:
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
        metadata = metadata or {}
        try:
            parsed = _loads_json(output)
            schema_valid = True
        except Exception:
            parsed = None
            schema_valid = False

        required_keys = metadata.get("required_keys", [])
        keys_ok = True
        if required_keys and isinstance(parsed, dict):
            keys_ok = all(key in parsed for key in required_keys)
        elif required_keys:
            keys_ok = False

        expected_ok = True
        try:
            expected_parsed = _loads_json(expected)
            if isinstance(expected_parsed, dict) and isinstance(parsed, dict):
                expected_ok = all(parsed.get(key) == value for key, value in expected_parsed.items())
        except Exception:
            expected_ok = expected.strip() in output if expected.strip() else schema_valid

        correct = schema_valid and keys_ok and expected_ok
        return TaskEvaluation(
            request_id=request_id,
            depth=depth,
            task=task,
            output=output,
            expected=expected,
            correct=correct,
            score=1.0 if correct else 0.0,
            high_cost_error=0 if correct else 1,
            metric={
                "schema_valid": schema_valid,
                "required_keys_ok": keys_ok,
                "expected_fields_ok": expected_ok,
            },
        )
