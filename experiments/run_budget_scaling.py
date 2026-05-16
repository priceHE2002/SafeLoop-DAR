from __future__ import annotations

import argparse
from collections import defaultdict

import _bootstrap  # noqa: F401
from safeloop.features.builder import FeatureBuilder
from safeloop.features.token_stage import HIGH_RISK_TYPES
from safeloop.types import FeatureRow
from safeloop.utils.io import ensure_dir, read_traces, write_json


def _group_token_rows(rows: list[FeatureRow]) -> dict[tuple[str, int], list[FeatureRow]]:
    grouped: dict[tuple[str, int], list[FeatureRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.request_id, row.position)].append(row)
    return grouped


def _choose_at_depth(rows: list[FeatureRow], depth: int) -> list[FeatureRow]:
    chosen = []
    grouped = _group_token_rows(rows)
    for token_rows in grouped.values():
        token_rows = sorted(token_rows, key=lambda row: row.depth)
        selected = token_rows[-1]
        for row in token_rows:
            if row.depth >= depth:
                selected = row
                break
        chosen.append(selected)
    return chosen


def _risk_by(rows: list[FeatureRow], key_name: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[FeatureRow]] = defaultdict(list)
    for row in rows:
        grouped[str(getattr(row, key_name))].append(row)
    return {
        key: {
            "tokens": float(len(group_rows)),
            "risk": sum(row.label for row in group_rows) / max(len(group_rows), 1),
            "avg_depth": sum(row.depth for row in group_rows) / max(len(group_rows), 1),
        }
        for key, group_rows in sorted(grouped.items())
    }


def fixed_depth_risk(rows: list[FeatureRow], depth: int) -> dict:
    chosen = _choose_at_depth(rows, depth)
    return {
        "depth": float(depth),
        "tokens": float(len(chosen)),
        "risk": sum(row.label for row in chosen) / max(len(chosen), 1),
        "avg_depth": sum(row.depth for row in chosen) / max(len(chosen), 1),
        "risk_by_token_type": _risk_by(chosen, "token_type"),
        "risk_by_task": _risk_by(chosen, "task"),
        "risk_by_group": _risk_by(chosen, "group"),
    }


def _summarize_records(records: list[dict], key_name: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record[key_name])].append(record)
    out = {}
    for key, values in sorted(grouped.items()):
        depths = [float(item["required_depth"]) for item in values]
        out[key] = {
            "tokens": float(len(values)),
            "avg_required_depth": sum(depths) / max(len(depths), 1),
            "min_required_depth": min(depths) if depths else 0.0,
            "max_required_depth": max(depths) if depths else 0.0,
        }
    return out


def required_depth_summary(rows: list[FeatureRow]) -> dict:
    records = []
    for token_rows in _group_token_rows(rows).values():
        token_rows = sorted(token_rows, key=lambda row: row.depth)
        first_safe = token_rows[-1]
        for row in token_rows:
            if row.label == 0:
                first_safe = row
                break
        first = token_rows[0]
        records.append(
            {
                "request_id": first.request_id,
                "position": first.position,
                "task": first.task,
                "token_type": first.token_type,
                "group": first.group,
                "is_high_risk": first.token_type in HIGH_RISK_TYPES,
                "required_depth": first_safe.depth,
                "full_depth": token_rows[-1].depth,
            }
        )

    depths = [float(item["required_depth"]) for item in records]
    high_risk = [item for item in records if item["is_high_risk"]]
    high_depths = [float(item["required_depth"]) for item in high_risk]
    return {
        "tokens": float(len(records)),
        "overall_avg_required_depth": sum(depths) / max(len(depths), 1),
        "overall_max_required_depth": max(depths) if depths else 0.0,
        "high_risk_tokens": float(len(high_risk)),
        "high_risk_avg_required_depth": sum(high_depths) / max(len(high_depths), 1),
        "by_token_type": _summarize_records(records, "token_type"),
        "by_task": _summarize_records(records, "task"),
        "by_group": _summarize_records(records, "group"),
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
    write_json(
        out_dir / "budget_scaling.json",
        {
            "points": points,
            "required_depth": required_depth_summary(rows),
        },
    )


if __name__ == "__main__":
    main()
