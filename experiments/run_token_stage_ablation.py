from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace

import _bootstrap  # noqa: F401
from safeloop.evaluation.report import prediction_report
from safeloop.evaluation.splits import split_by_request_three_way
from safeloop.features.builder import FeatureBuilder
from safeloop.risk.predictor import OnlineLogisticRiskPredictor
from safeloop.utils.io import ensure_dir, read_traces, write_json


FEATURE_SETS = {
    "confidence": {"entropy", "top1_prob", "top1_top2_margin", "logit_delta"},
    "confidence_hidden": {"entropy", "top1_prob", "top1_top2_margin", "logit_delta", "hidden_delta"},
    "implicit_no_group": {
        "entropy",
        "top1_prob",
        "top1_top2_margin",
        "logit_delta",
        "hidden_delta",
        "depth_attention_stability",
        "depth_attention_shift",
        "residual_novelty",
    },
    "explicit_group_only": {
        "relative_depth",
        "is_high_risk_token",
        "is_tool_stage",
        "is_code_stage",
        "is_final_stage",
        "group_low_risk_text",
        "group_tool_json",
        "group_code",
        "group_math_final",
        "group_entity",
    },
    "random_group": set(),
    "hybrid_full": set(),
}


def mask_rows(rows, feature_set: str):
    allowed = FEATURE_SETS[feature_set]
    masked = []
    for row in rows:
        if feature_set == "hybrid_full":
            clone = replace(row, features=dict(row.features))
        elif feature_set == "random_group":
            features = {
                key: value
                for key, value in row.features.items()
                if not key.startswith("group_") and key not in {"is_high_risk_token", "is_tool_stage", "is_code_stage", "is_final_stage"}
            }
            bucket = int(hashlib.sha256(f"{row.request_id}:{row.position}".encode("utf-8")).hexdigest()[:2], 16) % 5
            features[f"random_group_{bucket}"] = 1.0
            clone = replace(row, features=features)
        else:
            clone = replace(row, features={k: v for k, v in row.features.items() if k in allowed})
        masked.append(clone)
    return masked


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature ablation for token/stage and depth-dynamics signals.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output-dir", default="runs/token_stage_ablation")
    parser.add_argument("--label-type", default="task_degradation")
    parser.add_argument("--split-salt", default="")
    args = parser.parse_args()

    base_rows = FeatureBuilder().build_many(read_traces(args.trace), label_type=args.label_type)
    results = {}
    for name in FEATURE_SETS:
        rows = mask_rows(base_rows, name)
        train, calibration, test, split_ids = split_by_request_three_way(rows, salt=args.split_salt)
        predictor = OnlineLogisticRiskPredictor().fit(train)
        results[name] = {
            "train": prediction_report(train, predictor.predict_rows(train)),
            "calibration": prediction_report(calibration, predictor.predict_rows(calibration)),
            "test": prediction_report(test, predictor.predict_rows(test)),
        }
    out_dir = ensure_dir(args.output_dir)
    write_json(out_dir / "ablation.json", {"splits": split_ids.to_dict(), "results": results})


if __name__ == "__main__":
    main()
