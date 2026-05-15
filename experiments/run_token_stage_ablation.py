from __future__ import annotations

import argparse
from dataclasses import replace

import _bootstrap  # noqa: F401
from safeloop.evaluation.report import prediction_report
from safeloop.features.builder import FeatureBuilder
from safeloop.risk.predictor import OnlineLogisticRiskPredictor
from safeloop.utils.io import ensure_dir, read_traces, write_json


FEATURE_SETS = {
    "confidence": {"entropy", "top1_prob", "top1_top2_margin", "logit_delta"},
    "confidence_hidden": {"entropy", "top1_prob", "top1_top2_margin", "logit_delta", "hidden_delta"},
    "dar": {
        "entropy",
        "top1_prob",
        "top1_top2_margin",
        "logit_delta",
        "hidden_delta",
        "depth_attention_stability",
        "depth_attention_shift",
        "residual_novelty",
    },
    "dar_token_stage": set(),
}


def mask_rows(rows, feature_set: str):
    allowed = FEATURE_SETS[feature_set]
    masked = []
    for row in rows:
        if feature_set != "dar_token_stage":
            clone = replace(row, features={k: v for k, v in row.features.items() if k in allowed})
        else:
            clone = replace(row, features=dict(row.features))
        masked.append(clone)
    return masked


def _bucket(row) -> int:
    return sum(ord(ch) for ch in row.request_id) % 4


def split_rows(rows):
    train = []
    test = []
    for row in rows:
        (test if _bucket(row) == 0 else train).append(row)
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature ablation for token/stage and DAR signals.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output-dir", default="runs/token_stage_ablation")
    parser.add_argument("--label-type", default="teacher_consistency")
    args = parser.parse_args()

    base_rows = FeatureBuilder().build_many(read_traces(args.trace), label_type=args.label_type)
    results = {}
    for name in FEATURE_SETS:
        rows = mask_rows(base_rows, name)
        train, test = split_rows(rows)
        predictor = OnlineLogisticRiskPredictor().fit(train)
        results[name] = {
            "train": prediction_report(train, predictor.predict_rows(train)),
            "test": prediction_report(test, predictor.predict_rows(test)),
        }
    out_dir = ensure_dir(args.output_dir)
    write_json(out_dir / "ablation.json", results)


if __name__ == "__main__":
    main()
