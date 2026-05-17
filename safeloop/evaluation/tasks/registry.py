from __future__ import annotations

from safeloop.evaluation.tasks.base import TaskEvaluator
from safeloop.evaluation.tasks.gsm8k import GSM8KEvaluator
from safeloop.evaluation.tasks.humaneval import HumanEvalEvaluator
from safeloop.evaluation.tasks.json_schema import JSONSchemaEvaluator
from safeloop.evaluation.tasks.math500 import MATH500Evaluator
from safeloop.evaluation.tasks.rag_entity import RAGEntityEvaluator
from safeloop.evaluation.tasks.tool_call import ToolCallEvaluator


def get_task_evaluator(task: str, stage: str = "") -> TaskEvaluator:
    key = task.lower()
    stage_key = stage.lower()
    if key in {"gsm8k", "math"}:
        return GSM8KEvaluator()
    if key in {"math500", "math_500", "competition_math"}:
        return MATH500Evaluator()
    if key in {"humaneval", "humaneval_lite", "code"} or stage_key == "code_generation":
        return HumanEvalEvaluator()
    if key in {"tool", "tool_call", "agentic_rag"} or stage_key == "tool_call_json":
        return ToolCallEvaluator()
    if key in {"json", "structured", "structured_output"}:
        return JSONSchemaEvaluator()
    if key in {"rag", "retrieval", "entity", "hotpotqa", "2wikimultihopqa"}:
        return RAGEntityEvaluator()
    return RAGEntityEvaluator() if stage_key in {"retrieval", "final_answer"} else JSONSchemaEvaluator()
