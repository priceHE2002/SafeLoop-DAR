from __future__ import annotations

from collections import defaultdict
from typing import Any

from safeloop.evaluation.tasks.base import TaskEvaluation, detokenize
from safeloop.evaluation.tasks.registry import get_task_evaluator
from safeloop.types import TokenTrace


def group_traces_by_request(traces: list[TokenTrace]) -> dict[str, list[TokenTrace]]:
    grouped: dict[str, list[TokenTrace]] = defaultdict(list)
    for trace in traces:
        grouped[trace.request_id].append(trace)
    return grouped


def render_depth_output(traces: list[TokenTrace], depth: int) -> str:
    tokens = []
    for trace in sorted(traces, key=lambda item: item.position):
        step = trace.get_step(depth)
        if step is None:
            step = trace.full_step()
        tokens.append(step.token)
    return detokenize(tokens)


def expected_output(traces: list[TokenTrace]) -> str:
    return detokenize([trace.target_text for trace in sorted(traces, key=lambda item: item.position)])


def available_depths(traces: list[TokenTrace]) -> list[int]:
    depths = {step.depth for trace in traces for step in trace.steps}
    return sorted(depths)


def annotate_task_degradation(traces: list[TokenTrace]) -> tuple[list[TokenTrace], dict[str, Any]]:
    grouped = group_traces_by_request(traces)
    all_evaluations = []
    request_summaries = []

    for request_id, request_traces in grouped.items():
        request_traces = sorted(request_traces, key=lambda item: item.position)
        first = request_traces[0]
        metadata = dict(first.metadata.get("sample_metadata", {}))
        evaluator = get_task_evaluator(first.task, first.stage)
        expected = metadata.get("expected", "") or expected_output(request_traces)
        depths = available_depths(request_traces)
        if not depths:
            continue
        full_depth = max(depths)
        full_eval = evaluator.evaluate(
            request_id=request_id,
            task=first.task,
            stage=first.stage,
            output=render_depth_output(request_traces, full_depth),
            expected=expected,
            depth=full_depth,
            metadata=metadata,
        )
        degradation_by_depth = {}
        high_cost_by_depth = {}
        score_by_depth = {}
        correct_by_depth = {}
        evals_by_depth: dict[str, TaskEvaluation] = {}
        for depth in depths:
            evaluation = evaluator.evaluate(
                request_id=request_id,
                task=first.task,
                stage=first.stage,
                output=render_depth_output(request_traces, depth),
                expected=expected,
                depth=depth,
                metadata=metadata,
            )
            degradation = 1 if evaluation.score + 1e-12 < full_eval.score else 0
            degradation_by_depth[str(depth)] = degradation
            high_cost_by_depth[str(depth)] = int(evaluation.high_cost_error)
            score_by_depth[str(depth)] = evaluation.score
            correct_by_depth[str(depth)] = int(evaluation.correct)
            evals_by_depth[str(depth)] = evaluation
            all_evaluations.append(
                evaluation.to_dict()
                | {
                    "full_depth": full_depth,
                    "full_depth_correct": full_eval.correct,
                    "full_depth_score": full_eval.score,
                    "task_degradation": degradation,
                }
            )

        for trace in request_traces:
            trace.metadata["task_degradation_by_depth"] = degradation_by_depth
            trace.metadata["high_cost_error_by_depth"] = high_cost_by_depth
            trace.metadata["task_score_by_depth"] = score_by_depth
            trace.metadata["task_correct_by_depth"] = correct_by_depth
            trace.metadata["task_evaluator"] = evaluator.__class__.__name__
            trace.metadata["full_depth_correct"] = full_eval.correct
            trace.metadata["full_depth_score"] = full_eval.score

        request_summaries.append(
            {
                "request_id": request_id,
                "task": first.task,
                "stage": first.stage,
                "depths": depths,
                "full_depth": full_depth,
                "full_depth_correct": full_eval.correct,
                "full_depth_score": full_eval.score,
                "best_depth_score": max(score_by_depth.values()) if score_by_depth else 0.0,
                "evaluations": {depth: evals_by_depth[depth].to_dict() for depth in evals_by_depth},
            }
        )

    summary = {
        "requests": len(grouped),
        "evaluations": len(all_evaluations),
        "request_summaries": request_summaries,
        "evaluations_by_depth": all_evaluations,
    }
    return traces, summary
