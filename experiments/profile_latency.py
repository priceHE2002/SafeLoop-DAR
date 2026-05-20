from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from safeloop.evaluation.profiling import LatencyRecorder
from safeloop.features.builder import FeatureBuilder
from safeloop.risk.predictor import OnlineLogisticRiskPredictor
from safeloop.utils.io import ensure_dir, read_traces, write_json


def profile_from_trace(trace_path: str, output_dir: str) -> None:
    traces = read_traces(trace_path)
    recorder = LatencyRecorder()
    for trace in traces:
        for step in trace.steps:
            if step.latency_ms:
                recorder.add("recorded_depth_step_latency", step.latency_ms)

    builder = FeatureBuilder()
    with recorder.measure("feature_extraction_total"):
        rows = builder.build_many(traces, label_type="task_degradation")
    predictor = OnlineLogisticRiskPredictor(epochs=3)
    with recorder.measure("risk_predictor_fit"):
        predictor.fit(rows)
    with recorder.measure("risk_predictor_inference"):
        predictor.predict_rows(rows)

    depth_latencies = recorder.values.get("recorded_depth_step_latency", [])
    loop_step_ms = sum(depth_latencies) / max(len(depth_latencies), 1)
    out_dir = ensure_dir(output_dir)
    write_json(
        Path(out_dir) / "profile.json",
        {
            "source": trace_path,
            "tokens": len(traces),
            "feature_rows": len(rows),
            "loop_step_ms": loop_step_ms,
            "summary": recorder.summary(),
            "notes": "Trace-based profiling uses recorded DepthStep latency when available; real GPU profiling should replace this profile.",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile SafeLoop latency components.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output-dir", default="runs/latency_profile")
    args = parser.parse_args()
    profile_from_trace(args.trace, args.output_dir)


if __name__ == "__main__":
    main()
