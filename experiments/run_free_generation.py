from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from safeloop.adapters.factory import build_adapter
from safeloop.tracing.collector import TraceCollector
from safeloop.utils.config import load_experiment_config, load_json
from safeloop.utils.io import ensure_dir, write_json, write_traces
from safeloop.workloads.datasets import load_benchmark_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect free-generation traces.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    root = Path(cfg["_project_root"])
    model_cfg = load_json(root / cfg["model_config"])
    samples = load_benchmark_config(root / cfg["benchmark_config"])
    traces = TraceCollector(build_adapter(model_cfg)).collect(
        samples=samples,
        mode="free_generation",
        max_new_tokens=int(cfg.get("max_new_tokens", 16)),
    )
    out_dir = ensure_dir(root / cfg.get("output_dir", "runs/free_generation"))
    write_traces(out_dir / "free_generation.jsonl", traces)
    write_json(out_dir / "summary.json", {"token_traces": len(traces)})


if __name__ == "__main__":
    main()

