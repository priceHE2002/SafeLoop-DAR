from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401
from safeloop.evaluation.risk_compute import frontier_point, summarize_selected_rows
from safeloop.evaluation.splits import split_by_request
from safeloop.features.builder import FeatureBuilder
from safeloop.halting.policies import GroupCalibratedPolicy
from safeloop.risk.calibration import GroupCalibrator
from safeloop.risk.predictor import OnlineLogisticRiskPredictor
from safeloop.utils.config import load_experiment_config
from safeloop.utils.io import ensure_dir, read_traces, write_json


FEATURE_SETS = {
    "entropy_only": {"entropy"},
    "margin_only": {"top1_top2_margin"},
    "confidence": {"entropy", "top1_prob", "top1_top2_margin", "logit_delta"},
    "dar_no_token_stage": {
        "entropy",
        "top1_prob",
        "top1_top2_margin",
        "logit_delta",
        "hidden_delta",
        "depth_attention_stability",
        "depth_attention_shift",
        "residual_novelty",
    },
    "safeloop_dar": set(),
}


def split_rows(rows):
    return split_by_request(rows, train_fraction=0.67)


def mask_rows(rows, feature_set: str):
    allowed = FEATURE_SETS[feature_set]
    if feature_set == "safeloop_dar":
        return [replace(row, features=dict(row.features)) for row in rows]
    return [replace(row, features={k: v for k, v in row.features.items() if k in allowed}) for row in rows]


def fixed_depth_point(rows, depth: int) -> dict:
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


def oracle_point(rows) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.request_id, row.position)].append(row)
    selected = []
    for token_rows in grouped.values():
        token_rows = sorted(token_rows, key=lambda row: row.depth)
        choice = token_rows[-1]
        for row in token_rows:
            if row.label == 0:
                choice = row
                break
        selected.append(choice)
    return summarize_selected_rows(selected, total_tokens=len(grouped))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build risk-compute frontier.")
    parser.add_argument("--config")
    parser.add_argument("--trace")
    parser.add_argument("--output-dir")
    parser.add_argument("--targets")
    parser.add_argument("--label-type")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config) if args.config else {}
    root = Path(cfg.get("_project_root", "."))
    trace = args.trace or cfg.get("trace_path")
    if not trace:
        raise SystemExit("--trace or config.trace_path is required")
    output_dir = args.output_dir or cfg.get("output_dir", "runs/frontier")
    trace = str(Path(trace) if Path(trace).is_absolute() else root / trace)
    output_dir = str(Path(output_dir) if Path(output_dir).is_absolute() else root / output_dir)
    target_value = args.targets or cfg.get("risk_targets", "0.005,0.01,0.02,0.05,0.1")
    label_type = args.label_type or cfg.get("label_type", "teacher_consistency")

    traces = read_traces(trace)
    rows = FeatureBuilder().build_many(traces, label_type=label_type)
    cal_rows, test_rows = split_rows(rows)
    if isinstance(target_value, list):
        targets = [float(item) for item in target_value]
    else:
        targets = [float(item) for item in str(target_value).split(",")]
    points = []
    max_depth = max((row.depth for row in rows), default=1)
    for depth in range(1, max_depth + 1):
        point = fixed_depth_point(test_rows, depth)
        point["method"] = f"fixed_depth_{depth}"
        points.append(point)
    oracle = oracle_point(test_rows)
    oracle["method"] = "oracle_halting"
    points.append(oracle)

    for target in targets:
        for method in FEATURE_SETS:
            method_cal_rows = mask_rows(cal_rows, method)
            method_test_rows = mask_rows(test_rows, method)
            predictor = OnlineLogisticRiskPredictor().fit(method_cal_rows)
            cal_scores = predictor.predict_rows(method_cal_rows)
            calibrator = GroupCalibrator(target_risk=target).fit(method_cal_rows, cal_scores)
            policy = GroupCalibratedPolicy(predictor=predictor, calibrator=calibrator)
            point = frontier_point(method_test_rows, policy)
            point["method"] = method
            point["target_risk"] = target
            point["thresholds"] = calibrator.thresholds
            points.append(point)

    out_dir = ensure_dir(output_dir)
    write_json(out_dir / "frontier.json", {"points": points})


if __name__ == "__main__":
    main()
