from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from safeloop.adapters.factory import build_adapter
from safeloop.tracing.collector import TraceCollector
from safeloop.utils.config import load_experiment_config, load_json
from safeloop.utils.io import ensure_dir, write_json
from safeloop.utils.math_utils import l2_distance, l2_norm, softmax
from safeloop.workloads.datasets import load_benchmark_config


def top_probability(logits: list[float]) -> float:
    probs = softmax(logits)
    return max(probs) if probs else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate recurrent-depth control for an adapter.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--max-new-tokens", type=int)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    root = Path(cfg["_project_root"])
    model_cfg = load_json(root / cfg["model_config"])
    samples = load_benchmark_config(root / cfg["benchmark_config"])
    if not samples:
        raise SystemExit("benchmark config has no samples")
    max_new_tokens = args.max_new_tokens or int(cfg.get("max_new_tokens", 1))
    output_dir = args.output_dir or f"runs/{model_cfg.get('adapter', 'model')}_depth_control"

    traces = TraceCollector(build_adapter(model_cfg)).collect(
        samples=[samples[0]],
        mode="teacher_forced",
        max_new_tokens=max_new_tokens,
    )
    rows = []
    for trace in traces:
        previous = None
        for step in trace.sorted_steps():
            rows.append(
                {
                    "request_id": trace.request_id,
                    "position": trace.position,
                    "depth": step.depth,
                    "top_token": step.token,
                    "top_probability": top_probability(step.logits),
                    "hidden_norm": l2_norm(step.hidden),
                    "hidden_delta_from_previous_depth": l2_distance(step.hidden, previous.hidden) if previous else 0.0,
                    "logit_delta_from_previous_depth": l2_distance(step.logits, previous.logits) if previous else 0.0,
                    "latency_ms": step.latency_ms,
                }
            )
            previous = step

    out_dir = ensure_dir(root / output_dir)
    write_json(
        out_dir / "depth_control_validation.json",
        {
            "model_config": model_cfg,
            "experiment_config": cfg,
            "checks": {
                "depths_observed": sorted({row["depth"] for row in rows}),
                "hidden_changes_observed": any(row["hidden_delta_from_previous_depth"] > 0 for row in rows),
                "latency_recorded": any(row["latency_ms"] > 0 for row in rows),
            },
            "rows": rows,
        },
    )


if __name__ == "__main__":
    main()
