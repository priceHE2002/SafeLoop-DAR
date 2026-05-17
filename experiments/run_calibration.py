from __future__ import annotations

import argparse
from collections import defaultdict

import _bootstrap  # noqa: F401
from safeloop.features.builder import FeatureBuilder
from safeloop.evaluation.splits import split_by_request_three_way
from safeloop.risk.calibration import GroupCalibrator
from safeloop.risk.predictor import OnlineLogisticRiskPredictor
from safeloop.risk.ucb_calibration import UCBGroupCalibrator
from safeloop.utils.io import ensure_dir, read_traces, write_json


def build_calibrator(method: str, target: float, delta: float):
    if method == "empirical":
        return GroupCalibrator(target_risk=target)
    if method == "ucb":
        return UCBGroupCalibrator(target_risk=target, delta=delta)
    raise ValueError(f"Unsupported calibration method: {method}")


def empirical_risk(rows, scores, calibrator):
    accepted = [row.label for row, score in zip(rows, scores) if calibrator.should_exit(row.group, score)]
    by_group: dict[str, list[tuple[int, bool]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        by_group[row.group].append((row.label, calibrator.should_exit(row.group, score)))
    return {
        "accepted": len(accepted),
        "coverage": len(accepted) / max(len(rows), 1),
        "risk": sum(accepted) / max(len(accepted), 1),
        "by_group": {
            group: {
                "tokens": float(len(values)),
                "accepted": float(sum(1 for _, is_accepted in values if is_accepted)),
                "coverage": sum(1 for _, is_accepted in values if is_accepted) / max(len(values), 1),
                "risk": (
                    sum(label for label, is_accepted in values if is_accepted)
                    / max(sum(1 for _, is_accepted in values if is_accepted), 1)
                ),
            }
            for group, values in sorted(by_group.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate in-domain risk calibration.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output-dir", default="runs/calibration")
    parser.add_argument("--targets", default="0.005,0.01,0.02,0.05,0.1")
    parser.add_argument("--label-type", default="task_degradation")
    parser.add_argument("--calibration-method", default="ucb")
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--split-salt", default="")
    args = parser.parse_args()

    rows = FeatureBuilder().build_many(read_traces(args.trace), label_type=args.label_type)
    train_rows, calibration_rows, test_rows, split_ids = split_by_request_three_way(
        rows,
        train_fraction=0.5,
        calibration_fraction=0.25,
        salt=args.split_salt,
    )
    predictor = OnlineLogisticRiskPredictor().fit(train_rows)
    train_scores = predictor.predict_rows(train_rows)
    calibration_scores = predictor.predict_rows(calibration_rows)
    test_scores = predictor.predict_rows(test_rows)
    results = []
    for target in [float(item) for item in args.targets.split(",")]:
        calibrator = build_calibrator(args.calibration_method, target, args.delta).fit(
            calibration_rows,
            calibration_scores,
        )
        results.append(
            {
                "target_risk": target,
                "train": empirical_risk(train_rows, train_scores, calibrator),
                "calibration": empirical_risk(calibration_rows, calibration_scores, calibrator),
                "test": empirical_risk(test_rows, test_scores, calibrator),
                "calibrator": calibrator.to_dict(),
            }
        )
    out_dir = ensure_dir(args.output_dir)
    write_json(
        out_dir / "calibration.json",
        {
            "label_type": args.label_type,
            "calibration_method": args.calibration_method,
            "delta": args.delta,
            "splits": split_ids.to_dict(),
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
