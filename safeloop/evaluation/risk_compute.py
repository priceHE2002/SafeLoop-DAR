from __future__ import annotations

from collections import defaultdict

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


def frontier_point(rows: list[FeatureRow], policy: GroupCalibratedPolicy) -> dict[str, float]:
    chosen = first_exit_by_token(rows, policy)
    if not chosen:
        return {"avg_depth": 0.0, "risk": 0.0, "coverage": 0.0}
    selected = list(chosen.values())
    return {
        "avg_depth": sum(row.depth for row in selected) / len(selected),
        "risk": sum(row.label for row in selected) / len(selected),
        "coverage": 1.0,
        "tokens": float(len(selected)),
    }

