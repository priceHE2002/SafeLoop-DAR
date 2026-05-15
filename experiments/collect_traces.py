from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from safeloop.adapters.factory import build_adapter
from safeloop.tracing.collector import TraceCollector
from safeloop.utils.config import load_experiment_config, load_json
from safeloop.utils.io import ensure_dir, write_json, write_traces
from safeloop.utils.seed import set_seed
from safeloop.workloads.datasets import load_benchmark_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect loop-depth traces.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    root = Path(cfg["_project_root"])
    set_seed(int(cfg.get("seed", 7)))

    model_cfg = load_json(root / cfg["model_config"])
    benchmark_path = root / cfg["benchmark_config"]
    samples = load_benchmark_config(benchmark_path)
    adapter = build_adapter(model_cfg)
    collector = TraceCollector(adapter)
    traces = collector.collect(
        samples=samples,
        mode=str(cfg.get("mode", "teacher_forced")),
        max_new_tokens=int(cfg.get("max_new_tokens", 16)),
    )

    out_dir = ensure_dir(root / cfg.get("output_dir", "runs/traces"))
    trace_name = "free_generation.jsonl" if cfg.get("mode") == "free_generation" else "teacher_forced.jsonl"
    write_traces(out_dir / trace_name, traces)
    write_json(out_dir / "run_config.json", cfg)
    write_json(
        out_dir / "summary.json",
        {
            "model": model_cfg.get("model_name"),
            "adapter": model_cfg.get("adapter"),
            "samples": len(samples),
            "token_traces": len(traces),
            "mode": cfg.get("mode", "teacher_forced"),
        },
    )


if __name__ == "__main__":
    main()

