from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from safeloop.evaluation.report import prediction_report
from safeloop.evaluation.splits import split_by_request
from safeloop.features.builder import FeatureBuilder
from safeloop.risk.predictor import OnlineLogisticRiskPredictor
from safeloop.utils.config import load_experiment_config
from safeloop.utils.io import ensure_dir, read_traces, write_json, write_jsonl


def split_rows(rows, train_ratio: float = 0.7):
    return split_by_request(rows, train_fraction=train_ratio)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate safe-exit risk prediction.")
    parser.add_argument("--config")
    parser.add_argument("--trace")
    parser.add_argument("--output-dir")
    parser.add_argument("--label-type")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config) if args.config else {}
    root = Path(cfg.get("_project_root", "."))
    trace = args.trace or cfg.get("trace_path")
    if not trace:
        raise SystemExit("--trace or config.trace_path is required")
    output_dir = args.output_dir or cfg.get("output_dir", "runs/signal_prediction")
    trace = str(Path(trace) if Path(trace).is_absolute() else root / trace)
    output_dir = str(Path(output_dir) if Path(output_dir).is_absolute() else root / output_dir)
    label_type = args.label_type or cfg.get("label_type", "teacher_consistency")

    traces = read_traces(trace)
    rows = FeatureBuilder().build_many(traces, label_type=label_type)
    train_rows, test_rows = split_rows(rows)
    predictor = OnlineLogisticRiskPredictor().fit(train_rows)
    train_scores = predictor.predict_rows(train_rows)
    test_scores = predictor.predict_rows(test_rows)

    out_dir = ensure_dir(output_dir)
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
