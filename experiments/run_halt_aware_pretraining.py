from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from safeloop.controlled.halt_aware import HaltAwareTrainingConfig, train_halt_aware_state
from safeloop.utils.config import load_experiment_config, load_json
from safeloop.utils.io import ensure_dir, write_json
from safeloop.workloads.datasets import load_benchmark_config


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled halt-aware LoopTiny training.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    root = Path(cfg["_project_root"])
    out_dir = ensure_dir(_resolve(root, cfg.get("output_dir", "runs/halt_aware_pretraining")))
    model_cfg_path = _resolve(root, cfg["model_config"])
    benchmark_path = _resolve(root, cfg["benchmark_config"])
    model_cfg = load_json(model_cfg_path)
    samples = load_benchmark_config(benchmark_path)

    train_cfg = HaltAwareTrainingConfig(
        max_depth=int(model_cfg.get("max_depth", cfg.get("max_depth", 8))),
        seed=int(model_cfg.get("seed", cfg.get("seed", 19))),
        epochs=int(cfg.get("epochs", 12)),
        learning_rate=float(cfg.get("learning_rate", 0.35)),
        compute_penalty=float(cfg.get("compute_penalty", 0.08)),
        risk_penalty=float(cfg.get("risk_penalty", 1.0)),
        halt_margin=float(cfg.get("halt_margin", 0.25)),
        mode="standard",
    )
    standard_state = train_halt_aware_state(samples, train_cfg)
    halt_cfg = HaltAwareTrainingConfig(**(train_cfg.__dict__ | {"mode": "halt_aware"}))
    halt_state = train_halt_aware_state(samples, halt_cfg)

    standard_state_path = out_dir / "standard_state.json"
    halt_state_path = out_dir / "halt_aware_state.json"
    standard_state.save(str(standard_state_path))
    halt_state.save(str(halt_state_path))

    baseline_model_cfg = dict(model_cfg)
    baseline_model_cfg["model_name"] = str(model_cfg.get("model_name", "LoopTiny")) + "-Baseline"
    standard_model_cfg = dict(model_cfg) | {
        "model_name": str(model_cfg.get("model_name", "LoopTiny")) + "-StandardPT",
        "training_state_path": str(standard_state_path),
    }
    halt_model_cfg = dict(model_cfg) | {
        "model_name": str(model_cfg.get("model_name", "LoopTiny")) + "-HaltAwarePT",
        "training_state_path": str(halt_state_path),
    }
    write_json(out_dir / "baseline_model_config.json", baseline_model_cfg)
    write_json(out_dir / "standard_model_config.json", standard_model_cfg)
    write_json(out_dir / "halt_aware_model_config.json", halt_model_cfg)
    write_json(
        out_dir / "training_summary.json",
        {
            "samples": len(samples),
            "benchmark_config": str(benchmark_path),
            "base_model_config": str(model_cfg_path),
            "standard_state": standard_state.to_dict(),
            "halt_aware_state": halt_state.to_dict(),
            "outputs": {
                "baseline_model_config": str(out_dir / "baseline_model_config.json"),
                "standard_model_config": str(out_dir / "standard_model_config.json"),
                "halt_aware_model_config": str(out_dir / "halt_aware_model_config.json"),
            },
        },
    )


if __name__ == "__main__":
    main()
