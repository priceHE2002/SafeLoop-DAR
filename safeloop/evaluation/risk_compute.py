from __future__ import annotations

from collections import defaultdict
from typing import Any

from safeloop.halting.policies import GroupCalibratedPolicy
from safeloop.types import FeatureRow


def first_exit_by_token(rows: list[FeatureRow], policy: GroupCalibratedPolicy) -> dict[tuple[str, int], FeatureRow]:
    grouped: dict[tuple[str, int], list[FeatureRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.request_id, row.position)].append(row)
    chosen: dict[tuple[str, int], FeatureRow] = {}
    for key, token_rows in grouped.items():
        token_rows = sorted(token_rows, key=lambda row: row.depth)
        selected = token_rows[-1]
        for row in token_rows:
            if policy.decide(row).should_exit:
                selected = row
                break
        chosen[key] = selected
    return chosen


def summarize_selected_rows(selected: list[FeatureRow], total_tokens: int | None = None) -> dict[str, Any]:
    if not selected:
        return {
            "avg_depth": 0.0,
            "risk": 0.0,
            "coverage": 0.0,
            "tokens": 0.0,
            "risk_by_group": {},
            "avg_depth_by_group": {},
        }

    by_group: dict[str, list[FeatureRow]] = defaultdict(list)
    for row in selected:
        by_group[row.group].append(row)

    denominator = total_tokens if total_tokens is not None else len(selected)
    return {
        "avg_depth": sum(row.depth for row in selected) / len(selected),
        "risk": sum(row.label for row in selected) / len(selected),
        "coverage": len(selected) / max(denominator, 1),
        "tokens": float(len(selected)),
        "risk_by_group": {
            group: sum(row.label for row in group_rows) / max(len(group_rows), 1)
            for group, group_rows in sorted(by_group.items())
        },
        "avg_depth_by_group": {
            group: sum(row.depth for row in group_rows) / max(len(group_rows), 1)
            for group, group_rows in sorted(by_group.items())
        },
    }


def frontier_point(rows: list[FeatureRow], policy: GroupCalibratedPolicy) -> dict[str, Any]:
    chosen = first_exit_by_token(rows, policy)
    return summarize_selected_rows(list(chosen.values()), total_tokens=len(chosen))
