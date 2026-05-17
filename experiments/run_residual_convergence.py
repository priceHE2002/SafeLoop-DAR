from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from safeloop.features.builder import FeatureBuilder
from safeloop.utils.io import ensure_dir, read_traces, write_json


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs) ** 0.5
    den_y = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / max(den_x * den_y, 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure novelty vs next-depth gain.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output-dir", default="runs/residual_convergence")
    parser.add_argument("--label-type", default="task_degradation")
    args = parser.parse_args()

    traces = read_traces(args.trace)
    builder = FeatureBuilder()
    novelty = []
    next_gain = []
    for trace in traces:
        rows = builder.build_rows(trace, label_type=args.label_type)
        by_depth = {row.depth: row for row in rows}
        for depth, row in by_depth.items():
            nxt = by_depth.get(depth + 1)
            if nxt is None:
                continue
            novelty.append(row.features.get("residual_novelty", 0.0))
            next_gain.append(float(row.label - nxt.label))
    report = {
        "pairs": len(novelty),
        "pearson_novelty_next_gain": pearson(novelty, next_gain),
        "avg_novelty": sum(novelty) / max(len(novelty), 1),
        "avg_next_gain": sum(next_gain) / max(len(next_gain), 1),
    }
    out_dir = ensure_dir(args.output_dir)
    write_json(out_dir / "residual_convergence.json", report)


if __name__ == "__main__":
    main()
