from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from safeloop.controlled.tiny_loop_lm import TinyLoopLMConfig, train_tiny_loop_lm
from safeloop.utils.config import load_experiment_config
from safeloop.utils.io import ensure_dir, write_json
from safeloop.workloads.datasets import load_benchmark_config


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _config_from_dict(data: dict, halt_aware: bool) -> TinyLoopLMConfig:
    suffix = "HaltAware" if halt_aware else "Standard"
    return TinyLoopLMConfig(
        model_name=str(data.get("model_name", f"TinyLoopLM-{suffix}")),
        max_depth=int(data.get("max_depth", 8)),
        hidden_size=int(data.get("hidden_size", 24)),
        seed=int(data.get("seed", 23)),
        epochs=int(data.get("epochs", 20)),
        learning_rate=float(data.get("learning_rate", 0.25)),
        compute_penalty=float(data.get("compute_penalty", 0.08)),
        risk_penalty=float(data.get("risk_penalty", 1.0)),
        halt_margin=float(data.get("halt_margin", 0.25)),
        halt_aware=halt_aware,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small stdlib TinyLoopLM.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    root = Path(cfg["_project_root"])
    out_dir = ensure_dir(_resolve(root, cfg.get("output_dir", "runs/tiny_loop_lm_training")))
    samples = load_benchmark_config(_resolve(root, cfg["benchmark_config"]))
    standard_cfg = _config_from_dict(cfg, halt_aware=False)
    standard_cfg.model_name = str(cfg.get("standard_model_name", "TinyLoopLM-Standard"))
    halt_cfg = _config_from_dict(cfg, halt_aware=True)
    halt_cfg.model_name = str(cfg.get("halt_aware_model_name", "TinyLoopLM-HaltAware"))
    init_cfg = _config_from_dict(cfg | {"epochs": 0}, halt_aware=False)
    init_cfg.model_name = str(cfg.get("untrained_model_name", "TinyLoopLM-Untrained"))

    untrained = train_tiny_loop_lm(samples, init_cfg)
    standard = train_tiny_loop_lm(samples, standard_cfg)
    halt_aware = train_tiny_loop_lm(samples, halt_cfg)
    untrained_path = out_dir / "untrained_checkpoint.json"
    standard_path = out_dir / "standard_checkpoint.json"
    halt_path = out_dir / "halt_aware_checkpoint.json"
    untrained.save(str(untrained_path))
    standard.save(str(standard_path))
    halt_aware.save(str(halt_path))

    untrained_model_cfg = {
        "adapter": "tiny_loop_lm",
        "model_name": untrained.model_name,
        "checkpoint_path": str(untrained_path),
    }
    standard_model_cfg = {
        "adapter": "tiny_loop_lm",
        "model_name": standard.model_name,
        "checkpoint_path": str(standard_path),
    }
    halt_model_cfg = {
        "adapter": "tiny_loop_lm",
        "model_name": halt_aware.model_name,
        "checkpoint_path": str(halt_path),
    }
    write_json(out_dir / "untrained_model_config.json", untrained_model_cfg)
    write_json(out_dir / "standard_model_config.json", standard_model_cfg)
    write_json(out_dir / "halt_aware_model_config.json", halt_model_cfg)
    write_json(
        out_dir / "training_summary.json",
        {
            "benchmark_config": str(_resolve(root, cfg["benchmark_config"])),
            "samples": len(samples),
            "untrained_checkpoint": str(untrained_path),
            "standard_checkpoint": str(standard_path),
            "halt_aware_checkpoint": str(halt_path),
            "untrained_last_epoch": untrained.training_curve[-1] if untrained.training_curve else {},
            "standard_last_epoch": standard.training_curve[-1] if standard.training_curve else {},
            "halt_aware_last_epoch": halt_aware.training_curve[-1] if halt_aware.training_curve else {},
            "untrained_model_config": str(out_dir / "untrained_model_config.json"),
            "standard_model_config": str(out_dir / "standard_model_config.json"),
            "halt_aware_model_config": str(out_dir / "halt_aware_model_config.json"),
        },
    )


if __name__ == "__main__":
    main()
