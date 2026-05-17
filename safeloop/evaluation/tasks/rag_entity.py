from __future__ import annotations

from typing import Any

from safeloop.evaluation.tasks.base import TaskEvaluation, exact_or_contains


class RAGEntityEvaluator:
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
        gold_entities = metadata.get("gold_entities") or ([expected] if expected else [])
        entity_hits = [entity for entity in gold_entities if exact_or_contains(output, str(entity))]
        correct = bool(gold_entities) and len(entity_hits) == len(gold_entities)
        score = len(entity_hits) / max(len(gold_entities), 1)
        return TaskEvaluation(
            request_id=request_id,
            depth=depth,
            task=task,
            output=output,
            expected=expected,
            correct=correct,
            score=score,
            high_cost_error=0 if correct else 1,
            metric={
                "entity_correct": correct,
                "entity_recall": score,
                "gold_entities": gold_entities,
                "matched_entities": entity_hits,
            },
        )
