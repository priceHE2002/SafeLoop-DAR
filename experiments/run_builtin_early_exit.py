from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
from safeloop.adapters.factory import build_adapter
from safeloop.evaluation.tasks.registry import get_task_evaluator
from safeloop.utils.config import load_experiment_config, load_json
from safeloop.utils.io import ensure_dir, write_json
from safeloop.workloads.datasets import load_benchmark_config


def _set_threshold_on_config(model: Any, threshold: float) -> None:
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "early_exit_threshold"):
        setattr(cfg, "early_exit_threshold", threshold)


def _generate_with_hf(adapter: Any, prompt: str, max_new_tokens: int) -> str:
    adapter.load()
    assert adapter._model is not None and adapter._tokenizer is not None
    inputs = adapter._tokenizer(prompt, return_tensors="pt")
    device = adapter._infer_input_device()
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}
    output_ids = adapter._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
    return adapter._tokenizer.decode(new_tokens, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a model's built-in early-exit threshold grid.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--output-dir", default="runs/builtin_early_exit")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    root = Path(cfg["_project_root"])
    model_cfg = load_json(root / cfg["model_config"])
    samples = load_benchmark_config(root / cfg["benchmark_config"])
    adapter = build_adapter(model_cfg)
    thresholds = [float(item) for item in args.thresholds.split(",")]
    max_new_tokens = int(cfg.get("max_new_tokens", 64))
    results = []
    for threshold in thresholds:
        if not hasattr(adapter, "load"):
            results.append({"threshold": threshold, "error": "adapter does not expose HF generation"})
            continue
        adapter.load()
        _set_threshold_on_config(adapter._model, threshold)
        for sample in samples:
            output = _generate_with_hf(adapter, sample.prompt, max_new_tokens)
            evaluator = get_task_evaluator(sample.task, sample.stage)
            evaluation = evaluator.evaluate(
                request_id=sample.request_id,
                task=sample.task,
                stage=sample.stage,
                output=output,
                expected=sample.expected,
                depth=-1,
                metadata=sample.metadata or {},
            )
            results.append(evaluation.to_dict() | {"threshold": threshold})

    out_dir = ensure_dir(root / args.output_dir)
    write_json(
        out_dir / "builtin_early_exit.json",
        {
            "model_config": model_cfg,
            "thresholds": thresholds,
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
