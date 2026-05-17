from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from safeloop.evaluation.task_degradation import annotate_task_degradation
from safeloop.utils.config import load_experiment_config
from safeloop.utils.io import ensure_dir, read_traces, write_json, write_traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach task-degradation labels to traces.")
    parser.add_argument("--config")
    parser.add_argument("--trace")
    parser.add_argument("--output-dir")
    parser.add_argument("--output-name")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config) if args.config else {}
    root = Path(cfg.get("_project_root", "."))
    trace = args.trace or cfg.get("trace_path")
    if not trace:
        raise SystemExit("--trace or config.trace_path is required")
    trace = str(Path(trace) if Path(trace).is_absolute() else root / trace)
    output_dir = args.output_dir or cfg.get("output_dir", "runs/task_evaluation")
    output_name = args.output_name or cfg.get("output_name", "teacher_forced.evaluated.jsonl")

    traces = read_traces(trace)
    evaluated_traces, summary = annotate_task_degradation(traces)
    out_dir = ensure_dir(root / output_dir)
    write_traces(Path(out_dir) / output_name, evaluated_traces)
    write_json(Path(out_dir) / "task_metrics.json", summary)


if __name__ == "__main__":
    main()
