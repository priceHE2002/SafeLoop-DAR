from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from safeloop.evaluation.bootstrap import bootstrap_mean_ci
from safeloop.evaluation.risk_compute import first_exit_by_token
from safeloop.evaluation.splits import split_by_request_three_way
from safeloop.features.builder import FeatureBuilder
from safeloop.features.sets import mask_feature_rows
from safeloop.halting.policies import GroupCalibratedPolicy
from safeloop.risk.calibration import GroupCalibrator
from safeloop.risk.predictor import OnlineLogisticRiskPredictor
from safeloop.risk.ucb_calibration import UCBGroupCalibrator
from safeloop.utils.config import load_experiment_config
from safeloop.utils.io import ensure_dir, read_traces, write_json


def _calibrator(method: str, target: float, delta: float):
    if method == "empirical":
        return GroupCalibrator(target_risk=target)
    if method == "ucb":
        return UCBGroupCalibrator(target_risk=target, delta=delta)
    raise ValueError(f"Unsupported calibration method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap CIs for one SafeLoop frontier point.")
    parser.add_argument("--config")
    parser.add_argument("--trace")
    parser.add_argument("--output-dir")
    parser.add_argument("--feature-set", default="hidden_lite")
    parser.add_argument("--target-risk", type=float, default=0.02)
    parser.add_argument("--label-type", default="task_degradation")
    parser.add_argument("--calibration-method", default="ucb")
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split-salt", default="")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config) if args.config else {}
    root = Path(cfg.get("_project_root", "."))
    trace = args.trace or cfg.get("trace_path")
    if not trace:
        raise SystemExit("--trace or config.trace_path is required")
    trace_path = Path(trace) if Path(trace).is_absolute() else root / trace
    output_dir = args.output_dir or cfg.get("output_dir", "runs/bootstrap_frontier")
    output_dir = str(Path(output_dir) if Path(output_dir).is_absolute() else root / output_dir)
    feature_set = str(cfg.get("feature_set", args.feature_set))
    target_risk = float(cfg.get("target_risk", args.target_risk))
    repeats = int(cfg.get("repeats", args.repeats))

    rows = FeatureBuilder().build_many(read_traces(trace_path), label_type=args.label_type)
    rows = mask_feature_rows(rows, feature_set)
    train_rows, calibration_rows, test_rows, split_ids = split_by_request_three_way(
        rows,
        train_fraction=float(cfg.get("train_fraction", 0.5)),
        calibration_fraction=float(cfg.get("calibration_fraction", 0.25)),
        salt=args.split_salt or str(cfg.get("split_salt", "")),
    )
    predictor = OnlineLogisticRiskPredictor().fit(train_rows)
    calibrator = _calibrator(args.calibration_method, target_risk, args.delta).fit(
        calibration_rows,
        predictor.predict_rows(calibration_rows),
    )
    selected = list(
        first_exit_by_token(test_rows, GroupCalibratedPolicy(predictor=predictor, calibrator=calibrator)).values()
    )
    out_dir = ensure_dir(output_dir)
    write_json(
        out_dir / "bootstrap_frontier.json",
        {
            "feature_set": feature_set,
            "target_risk": target_risk,
            "label_type": args.label_type,
            "calibration_method": args.calibration_method,
            "splits": split_ids.to_dict(),
            "risk_ci": bootstrap_mean_ci([float(row.label) for row in selected], repeats, seed=args.seed),
            "avg_depth_ci": bootstrap_mean_ci([float(row.depth) for row in selected], repeats, seed=args.seed),
            "tokens": len(selected),
        },
    )


if __name__ == "__main__":
    main()
