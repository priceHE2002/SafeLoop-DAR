from __future__ import annotations

import argparse
from collections import defaultdict

import _bootstrap  # noqa: F401
from safeloop.features.builder import FeatureBuilder
from safeloop.utils.io import ensure_dir, read_traces, write_json


def fixed_depth_risk(rows, depth: int) -> dict[str, float]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.request_id, row.position)].append(row)
    chosen = []
    for token_rows in grouped.values():
        token_rows = sorted(token_rows, key=lambda row: row.depth)
        selected = token_rows[-1]
        for row in token_rows:
            if row.depth >= depth:
                selected = row
                break
        chosen.append(selected)
    return {
        "depth": float(depth),
        "tokens": float(len(chosen)),
        "risk": sum(row.label for row in chosen) / max(len(chosen), 1),
        "avg_depth": sum(row.depth for row in chosen) / max(len(chosen), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed budget scaling.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output-dir", default="runs/budget_scaling")
    parser.add_argument("--label-type", default="teacher_consistency")
    args = parser.parse_args()

    rows = FeatureBuilder().build_many(read_traces(args.trace), label_type=args.label_type)
    max_depth = max((row.depth for row in rows), default=1)
    points = [fixed_depth_risk(rows, depth) for depth in range(1, max_depth + 1)]
    out_dir = ensure_dir(args.output_dir)
    write_json(out_dir / "budget_scaling.json", {"points": points})


if __name__ == "__main__":
    main()
