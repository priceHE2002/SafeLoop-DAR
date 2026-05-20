from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
from safeloop.evaluation.scaling import fit_exponential_decay, pearson
from safeloop.features.builder import FeatureBuilder
from safeloop.utils.config import load_experiment_config
from safeloop.utils.io import ensure_dir, read_traces, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit depth-dynamics scaling curves.")
    parser.add_argument("--config")
    parser.add_argument("--trace")
    parser.add_argument("--output-dir")
    parser.add_argument("--label-type", default="task_degradation")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config) if args.config else {}
    root = Path(cfg.get("_project_root", "."))
    trace = args.trace or cfg.get("trace_path")
    if not trace:
        raise SystemExit("--trace or config.trace_path is required")
    trace_path = Path(trace) if Path(trace).is_absolute() else root / trace
    output_dir = args.output_dir or cfg.get("output_dir", "runs/depth_scaling")
    output_dir = str(Path(output_dir) if Path(output_dir).is_absolute() else root / output_dir)

    rows = FeatureBuilder().build_many(read_traces(trace_path), label_type=args.label_type)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.request_id, row.position)].append(row)

    fits = []
    floors = []
    labels = []
    for (request_id, position), token_rows in sorted(grouped.items()):
        token_rows = sorted(token_rows, key=lambda row: row.depth)
        points = [(float(row.depth), float(row.features.get("residual_novelty", 0.0))) for row in token_rows]
        fit = fit_exponential_decay(points)
        first_safe = next((row.depth for row in token_rows if row.label == 0), token_rows[-1].depth)
        final_label = token_rows[0].label
        fits.append(
            {
                "request_id": request_id,
                "position": position,
                "first_safe_depth": first_safe,
                "early_depth_label": final_label,
                **fit,
            }
        )
        floors.append(float(fit["c"]))
        labels.append(float(first_safe))

    out_dir = ensure_dir(output_dir)
    write_json(
        out_dir / "depth_scaling.json",
        {
            "label_type": args.label_type,
            "fit_count": len(fits),
            "floor_vs_first_safe_depth_pearson": pearson(floors, labels),
            "fits": fits,
            "model": "residual_novelty(depth) = a * exp(-b * depth) + c",
        },
    )


if __name__ == "__main__":
    main()
