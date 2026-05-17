from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
from safeloop.adapters.factory import build_adapter
from safeloop.evaluation.risk_compute import frontier_point, summarize_selected_rows
from safeloop.evaluation.splits import split_by_request_three_way
from safeloop.evaluation.task_degradation import annotate_task_degradation
from safeloop.features.builder import FeatureBuilder
from safeloop.halting.policies import GroupCalibratedPolicy
from safeloop.risk.predictor import OnlineLogisticRiskPredictor
from safeloop.risk.ucb_calibration import UCBGroupCalibrator
from safeloop.tracing.collector import TraceCollector
from safeloop.utils.config import load_experiment_config, load_json
from safeloop.utils.io import ensure_dir, write_json, write_traces
from safeloop.workloads.datasets import load_benchmark_config


FEATURE_SETS = {
    "confidence": {"entropy", "top1_prob", "top1_top2_margin", "logit_delta"},
    "halt_signals_lite": {"hidden_delta", "residual_novelty", "relative_depth"},
    "safeloop_dar_full": set(),
}


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _mask_rows(rows, feature_set: str):
    allowed = FEATURE_SETS[feature_set]
    if feature_set == "safeloop_dar_full":
        return [replace(row, features=dict(row.features)) for row in rows]
    return [replace(row, features={key: value for key, value in row.features.items() if key in allowed}) for row in rows]


def _fixed_depth_point(rows, depth: int) -> dict[str, Any]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.request_id, row.position)].append(row)
    selected = []
    for token_rows in grouped.values():
        token_rows = sorted(token_rows, key=lambda row: row.depth)
        choice = token_rows[-1]
        for row in token_rows:
            if row.depth >= depth:
                choice = row
                break
        selected.append(choice)
    return summarize_selected_rows(selected, total_tokens=len(grouped))


def _collect_and_evaluate(
    name: str,
    model_cfg: dict,
    samples,
    output_dir: Path,
    max_new_tokens: int,
):
    traces = TraceCollector(build_adapter(model_cfg)).collect(
        samples=samples,
        mode="teacher_forced",
        max_new_tokens=max_new_tokens,
    )
    evaluated, task_summary = annotate_task_degradation(traces)
    model_dir = ensure_dir(output_dir / name)
    write_traces(model_dir / "teacher_forced.jsonl", traces)
    write_traces(model_dir / "teacher_forced.evaluated.jsonl", evaluated)
    write_json(model_dir / "task_metrics.json", task_summary)
    return evaluated


def _build_frontier(
    traces,
    risk_targets: list[float],
    delta: float,
    split_salt: str,
) -> dict[str, Any]:
    rows = FeatureBuilder().build_many(traces, label_type="task_degradation")
    train_rows, calibration_rows, test_rows, split_ids = split_by_request_three_way(
        rows,
        train_fraction=0.5,
        calibration_fraction=0.25,
        salt=split_salt,
    )
    max_depth = max((row.depth for row in rows), default=1)
    points = []
    for depth in range(1, max_depth + 1):
        point = _fixed_depth_point(test_rows, depth)
        point["method"] = f"fixed_depth_{depth}"
        points.append(point)

    for target in risk_targets:
        for method in FEATURE_SETS:
            train = _mask_rows(train_rows, method)
            calibration = _mask_rows(calibration_rows, method)
            test = _mask_rows(test_rows, method)
            predictor = OnlineLogisticRiskPredictor().fit(train)
            scores = predictor.predict_rows(calibration)
            calibrator = UCBGroupCalibrator(target_risk=target, delta=delta).fit(calibration, scores)
            policy = GroupCalibratedPolicy(predictor=predictor, calibrator=calibrator)
            point = frontier_point(test, policy)
            point["method"] = method
            point["target_risk"] = target
            point["calibration_method"] = "ucb"
            point["thresholds"] = calibrator.thresholds
            points.append(point)

    return {
        "label_type": "task_degradation",
        "calibration_method": "ucb",
        "delta": delta,
        "splits": split_ids.to_dict(),
        "points": points,
    }


def _best_depth_at_risk(frontier: dict[str, Any], target: float) -> float | None:
    candidates = [
        point
        for point in frontier.get("points", [])
        if not str(point.get("method", "")).startswith("oracle") and float(point.get("risk", 1.0)) <= target
    ]
    if not candidates:
        return None
    return min(float(point.get("avg_depth", 0.0)) for point in candidates)


def _compare_frontiers(
    baseline: dict[str, Any],
    trained: dict[str, Any],
    risk_targets: list[float],
) -> dict[str, Any]:
    comparisons = []
    improved = 0
    comparable = 0
    for target in risk_targets:
        base_depth = _best_depth_at_risk(baseline, target)
        trained_depth = _best_depth_at_risk(trained, target)
        comparison = {
            "risk_target": target,
            "baseline_best_avg_depth": base_depth,
            "trained_best_avg_depth": trained_depth,
            "avg_depth_delta": None,
            "improved": False,
        }
        if base_depth is not None and trained_depth is not None:
            comparable += 1
            comparison["avg_depth_delta"] = trained_depth - base_depth
            comparison["improved"] = trained_depth < base_depth
            improved += int(comparison["improved"])
        comparisons.append(comparison)
    baseline_fixed = {
        str(point.get("method")): point
        for point in baseline.get("points", [])
        if str(point.get("method", "")).startswith("fixed_depth_")
    }
    trained_fixed = {
        str(point.get("method")): point
        for point in trained.get("points", [])
        if str(point.get("method", "")).startswith("fixed_depth_")
    }
    fixed_depth_comparisons = []
    fixed_non_worse = 0
    fixed_comparable = 0
    for method, base_point in sorted(baseline_fixed.items()):
        trained_point = trained_fixed.get(method)
        if trained_point is None:
            continue
        fixed_comparable += 1
        risk_delta = float(trained_point.get("risk", 0.0)) - float(base_point.get("risk", 0.0))
        fixed_non_worse += int(risk_delta <= 1e-12)
        fixed_depth_comparisons.append(
            {
                "method": method,
                "baseline_risk": base_point.get("risk"),
                "trained_risk": trained_point.get("risk"),
                "risk_delta": risk_delta,
                "baseline_risk_by_group": base_point.get("risk_by_group", {}),
                "trained_risk_by_group": trained_point.get("risk_by_group", {}),
            }
        )
    return {
        "comparisons": comparisons,
        "comparable_targets": comparable,
        "improved_targets": improved,
        "fixed_depth_comparisons": fixed_depth_comparisons,
        "fixed_depth_non_worse": fixed_non_worse,
        "fixed_depth_comparable": fixed_comparable,
        "claim_pass": (
            comparable > 0
            and improved == comparable
            and fixed_comparable > 0
            and fixed_non_worse == fixed_comparable
        ),
        "claim": (
            "Halt-aware continued pretraining improves the controlled "
            "risk-compute frontier if trained_best_avg_depth is lower at the "
            "same target risk and fixed-depth risk is not worse."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs halt-aware pretraining frontiers.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    root = Path(cfg["_project_root"])
    out_dir = ensure_dir(_resolve(root, cfg.get("output_dir", "runs/pretraining_frontier_comparison")))
    benchmark = load_benchmark_config(_resolve(root, cfg["benchmark_config"]))
    baseline_cfg = load_json(_resolve(root, cfg["baseline_model_config"]))
    standard_cfg = (
        load_json(_resolve(root, cfg["standard_model_config"]))
        if cfg.get("standard_model_config")
        else None
    )
    trained_cfg = load_json(_resolve(root, cfg["trained_model_config"]))
    max_new_tokens = int(cfg.get("max_new_tokens", 8))
    risk_targets = [float(item) for item in cfg.get("risk_targets", [0.01, 0.02, 0.05, 0.1])]
    delta = float(cfg.get("delta", 0.05))
    split_salt = str(cfg.get("split_salt", "halt-aware-pt"))

    baseline_traces = _collect_and_evaluate(
        "baseline",
        baseline_cfg,
        benchmark,
        out_dir,
        max_new_tokens=max_new_tokens,
    )
    trained_traces = _collect_and_evaluate(
        "halt_aware",
        trained_cfg,
        benchmark,
        out_dir,
        max_new_tokens=max_new_tokens,
    )
    baseline_frontier = _build_frontier(baseline_traces, risk_targets, delta, split_salt)
    trained_frontier = _build_frontier(trained_traces, risk_targets, delta, split_salt)
    write_json(out_dir / "baseline_frontier.json", baseline_frontier)
    write_json(out_dir / "halt_aware_frontier.json", trained_frontier)
    standard_frontier = None
    standard_comparison = None
    if standard_cfg:
        standard_traces = _collect_and_evaluate(
            "standard",
            standard_cfg,
            benchmark,
            out_dir,
            max_new_tokens=max_new_tokens,
        )
        standard_frontier = _build_frontier(standard_traces, risk_targets, delta, split_salt)
        write_json(out_dir / "standard_frontier.json", standard_frontier)
        standard_comparison = _compare_frontiers(standard_frontier, trained_frontier, risk_targets)
    comparison = _compare_frontiers(baseline_frontier, trained_frontier, risk_targets)
    write_json(
        out_dir / "comparison.json",
        {
            "benchmark_samples": len(benchmark),
            "risk_targets": risk_targets,
            "baseline_model": baseline_cfg.get("model_name"),
            "standard_model": standard_cfg.get("model_name") if standard_cfg else None,
            "trained_model": trained_cfg.get("model_name"),
            "baseline_vs_halt_aware": comparison,
            "standard_vs_halt_aware": standard_comparison,
            "comparison": comparison,
        },
    )


if __name__ == "__main__":
    main()
