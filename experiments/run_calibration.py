from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from safeloop.features.builder import FeatureBuilder
from safeloop.evaluation.splits import split_by_request
from safeloop.risk.calibration import GroupCalibrator
from safeloop.risk.predictor import OnlineLogisticRiskPredictor
from safeloop.utils.io import ensure_dir, read_traces, write_json


def split_rows(rows):
    return split_by_request(rows, train_fraction=0.5)


def empirical_risk(rows, scores, calibrator):
    accepted = [
        row.label
        for row, score in zip(rows, scores)
        if calibrator.should_exit(row.group, score)
    ]
    return {
        "accepted": len(accepted),
        "coverage": len(accepted) / max(len(rows), 1),
        "risk": sum(accepted) / max(len(accepted), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate in-domain risk calibration.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output-dir", default="runs/calibration")
    parser.add_argument("--targets", default="0.005,0.01,0.02,0.05,0.1")
    parser.add_argument("--label-type", default="teacher_consistency")
    args = parser.parse_args()

    rows = FeatureBuilder().build_many(read_traces(args.trace), label_type=args.label_type)
    cal_rows, test_rows = split_rows(rows)
    predictor = OnlineLogisticRiskPredictor().fit(cal_rows)
    cal_scores = predictor.predict_rows(cal_rows)
    test_scores = predictor.predict_rows(test_rows)
    results = []
    for target in [float(item) for item in args.targets.split(",")]:
        calibrator = GroupCalibrator(target_risk=target).fit(cal_rows, cal_scores)
        results.append(
            {
                "target_risk": target,
                "calibration": empirical_risk(cal_rows, cal_scores, calibrator),
                "test": empirical_risk(test_rows, test_scores, calibrator),
                "thresholds": calibrator.thresholds,
            }
        )
    out_dir = ensure_dir(args.output_dir)
    write_json(out_dir / "calibration.json", {"results": results})


if __name__ == "__main__":
    main()
