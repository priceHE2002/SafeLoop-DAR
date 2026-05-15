from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from safeloop.evaluation.report import prediction_report
from safeloop.features.builder import FeatureBuilder
from safeloop.risk.predictor import OnlineLogisticRiskPredictor
from safeloop.utils.io import ensure_dir, read_traces, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on one trace file and evaluate on another.")
    parser.add_argument("--train-trace", required=True)
    parser.add_argument("--test-trace", required=True)
    parser.add_argument("--output-dir", default="runs/transfer")
    parser.add_argument("--label-type", default="teacher_consistency")
    args = parser.parse_args()

    builder = FeatureBuilder()
    train_rows = builder.build_many(read_traces(args.train_trace), label_type=args.label_type)
    test_rows = builder.build_many(read_traces(args.test_trace), label_type=args.label_type)
    predictor = OnlineLogisticRiskPredictor().fit(train_rows)
    out_dir = ensure_dir(args.output_dir)
    write_json(
        out_dir / "transfer.json",
        {
            "train": prediction_report(train_rows, predictor.predict_rows(train_rows)),
            "test": prediction_report(test_rows, predictor.predict_rows(test_rows)),
        },
    )


if __name__ == "__main__":
    main()
