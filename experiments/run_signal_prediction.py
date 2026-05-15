from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from safeloop.evaluation.report import prediction_report
from safeloop.features.builder import FeatureBuilder
from safeloop.risk.predictor import OnlineLogisticRiskPredictor
from safeloop.utils.io import ensure_dir, read_traces, write_json, write_jsonl


def _bucket(row) -> int:
    return sum(ord(ch) for ch in row.request_id) % 10


def split_rows(rows, train_ratio: float = 0.7):
    train = []
    test = []
    cutoff = int(train_ratio * 10)
    for row in rows:
        (train if _bucket(row) < cutoff else test).append(row)
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate safe-exit risk prediction.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output-dir", default="runs/signal_prediction")
    parser.add_argument("--label-type", default="teacher_consistency")
    args = parser.parse_args()

    traces = read_traces(args.trace)
    rows = FeatureBuilder().build_many(traces, label_type=args.label_type)
    train_rows, test_rows = split_rows(rows)
    predictor = OnlineLogisticRiskPredictor().fit(train_rows)
    train_scores = predictor.predict_rows(train_rows)
    test_scores = predictor.predict_rows(test_rows)

    out_dir = ensure_dir(args.output_dir)
    write_json(out_dir / "predictor.json", predictor.to_dict())
    write_json(
        out_dir / "metrics.json",
        {
            "train": prediction_report(train_rows, train_scores),
            "test": prediction_report(test_rows, test_scores),
        },
    )
    write_jsonl(
        Path(out_dir) / "feature_rows.jsonl",
        (row.to_dict() | {"score": score} for row, score in zip(rows, predictor.predict_rows(rows))),
    )


if __name__ == "__main__":
    main()
