from __future__ import annotations

import re
from typing import Any

from safeloop.evaluation.tasks.base import TaskEvaluation, normalize_text
from safeloop.evaluation.tasks.gsm8k import extract_final_number


_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")


def extract_boxed_or_final(text: str) -> str:
    boxed = _BOXED.findall(text)
    if boxed:
        return normalize_text(boxed[-1])
    number = extract_final_number(text)
    return number if number else normalize_text(text)


class MATH500Evaluator:
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
        pred = extract_boxed_or_final(output)
        gold = extract_boxed_or_final(expected)
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
            metric={"answer": pred, "gold": gold, "exact_match": float(correct)},
        )
