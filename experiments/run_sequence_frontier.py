from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
from safeloop.evaluation.sequence_risk import (
    aggregate_request_depth_rows,
    sequence_frontier_point,
    summarize_request_rows,
)
from safeloop.evaluation.splits import split_by_request_three_way
from safeloop.features.builder import FeatureBuilder
from safeloop.features.sets import mask_feature_rows, parse_feature_sets
from safeloop.halting.policies import GroupCalibratedPolicy
from safeloop.risk.calibration import GroupCalibrator
from safeloop.risk.predictor import OnlineLogisticRiskPredictor
from safeloop.risk.ucb_calibration import UCBGroupCalibrator
from safeloop.utils.config import load_experiment_config
from safeloop.utils.io import ensure_dir, read_traces, write_json


def _build_calibrator(method: str, target: float, delta: float):
    if method == "empirical":
        return GroupCalibrator(target_risk=target)
    if method == "ucb":
        return UCBGroupCalibrator(target_risk=target, delta=delta)
    raise ValueError(f"Unsupported calibration method: {method}")


def _fixed_depth_point(rows, depth: int) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.request_id].append(row)
    selected = []
    for request_rows in grouped.values():
        request_rows = sorted(request_rows, key=lambda row: row.depth)
        choice = request_rows[-1]
        for row in request_rows:
            if row.depth >= depth:
                choice = row
                break
        selected.append(choice)
    return summarize_request_rows(selected, total_requests=len(grouped))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build request-level risk-compute frontier.")
    parser.add_argument("--config")
    parser.add_argument("--trace")
    parser.add_argument("--output-dir")
    parser.add_argument("--targets")
    parser.add_argument("--label-type")
    parser.add_argument("--calibration-method")
    parser.add_argument("--delta", type=float)
    parser.add_argument("--feature-sets")
    parser.add_argument("--group-mode", choices=["all", "task", "stage"])
    parser.add_argument("--split-salt", default="")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config) if args.config else {}
    root = Path(cfg.get("_project_root", "."))
    trace = args.trace or cfg.get("trace_path")
    if not trace:
        raise SystemExit("--trace or config.trace_path is required")
    trace_path = Path(trace) if Path(trace).is_absolute() else root / trace
    output_dir = args.output_dir or cfg.get("output_dir", "runs/sequence_frontier")
    output_dir = str(Path(output_dir) if Path(output_dir).is_absolute() else root / output_dir)
    label_type = args.label_type or cfg.get("label_type", "task_degradation")
    calibration_method = args.calibration_method or cfg.get("calibration_method", "ucb")
    delta = args.delta if args.delta is not None else float(cfg.get("delta", 0.05))
    group_mode = args.group_mode or str(cfg.get("group_mode", "all"))
    feature_sets = parse_feature_sets(args.feature_sets or cfg.get("feature_sets", "hidden_lite,dynamics_only"))
    target_value = args.targets or cfg.get("risk_targets", "0.01,0.02,0.05,0.1")
    targets = [float(item) for item in target_value] if isinstance(target_value, list) else [
        float(item) for item in str(target_value).split(",")
    ]

    token_rows = FeatureBuilder().build_many(read_traces(trace_path), label_type=label_type)
    points = []
    split_payload = {}
    for feature_set in feature_sets:
        masked = mask_feature_rows(token_rows, feature_set)
        sequence_rows = aggregate_request_depth_rows(masked, group_mode=group_mode)
        train_rows, calibration_rows, test_rows, split_ids = split_by_request_three_way(
            sequence_rows,
            train_fraction=float(cfg.get("train_fraction", 0.5)),
            calibration_fraction=float(cfg.get("calibration_fraction", 0.25)),
            salt=args.split_salt or str(cfg.get("split_salt", "")),
        )
        split_payload[feature_set] = split_ids.to_dict()
        max_depth = max((row.depth for row in sequence_rows), default=1)
        if feature_set == feature_sets[0]:
            for depth in range(1, max_depth + 1):
                point = _fixed_depth_point(test_rows, depth)
                point["method"] = f"fixed_depth_{depth}"
                point["feature_set"] = feature_set
                points.append(point)
        for target in targets:
            predictor = OnlineLogisticRiskPredictor().fit(train_rows)
            scores = predictor.predict_rows(calibration_rows)
            calibrator = _build_calibrator(calibration_method, target, delta).fit(calibration_rows, scores)
            policy = GroupCalibratedPolicy(predictor=predictor, calibrator=calibrator)
            point = sequence_frontier_point(test_rows, policy)
            point["method"] = "sequence_controller"
            point["feature_set"] = feature_set
            point["target_risk"] = target
            point["calibration_method"] = calibration_method
            point["thresholds"] = calibrator.thresholds
            points.append(point)

    out_dir = ensure_dir(output_dir)
    write_json(
        out_dir / "sequence_frontier.json",
        {
            "unit": "request",
            "label_type": label_type,
            "calibration_method": calibration_method,
            "delta": delta,
            "group_mode": group_mode,
            "feature_sets": feature_sets,
            "splits": split_payload,
            "points": points,
        },
    )


if __name__ == "__main__":
    main()
