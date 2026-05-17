from __future__ import annotations

import re
from typing import Any

from safeloop.evaluation.tasks.base import TaskEvaluation


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def extract_final_number(text: str) -> str:
    matches = _NUMBER.findall(text.replace(",", ""))
    return matches[-1] if matches else ""


class GSM8KEvaluator:
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
        pred = extract_final_number(output)
        gold = extract_final_number(expected)
        correct = bool(gold) and pred == gold
        return TaskEvaluation(
            request_id=request_id,
            depth=depth,
            task=task,
            output=output,
            expected=expected,
            correct=correct,
            score=1.0 if correct else 0.0,
            high_cost_error=0 if correct else 1,
            metric={"final_number": pred, "gold_number": gold, "exact_match": float(correct)},
        )
