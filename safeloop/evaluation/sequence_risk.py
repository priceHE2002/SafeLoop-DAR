from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from safeloop.halting.policies import GroupCalibratedPolicy
from safeloop.types import FeatureRow


def aggregate_request_depth_rows(
    rows: Iterable[FeatureRow],
    group_mode: str = "all",
) -> list[FeatureRow]:
    """Aggregate token-depth rows into request-depth rows.

    The request-level label is 1 if any token at that request/depth degrades.
    This is stricter than token-level control and better matches code/tool/RAG
    request-level failure.
    """

    grouped: dict[tuple[str, int], list[FeatureRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.request_id, row.depth)].append(row)

    sequence_rows: list[FeatureRow] = []
    for (request_id, depth), token_rows in sorted(grouped.items()):
        first = token_rows[0]
        keys = sorted({key for row in token_rows for key in row.features})
        features: dict[str, float] = {
            "request_token_count": float(len(token_rows)),
            "relative_depth": first.features.get("relative_depth", 0.0),
        }
        for key in keys:
            values = [row.features.get(key, 0.0) for row in token_rows]
            features[f"mean_{key}"] = sum(values) / max(len(values), 1)
            features[f"max_{key}"] = max(values) if values else 0.0
            features[f"min_{key}"] = min(values) if values else 0.0
        if group_mode == "task":
            group = f"task:{first.task}"
        elif group_mode == "stage":
            group = f"stage:{first.stage}"
        else:
            group = "request"
        sequence_rows.append(
            FeatureRow(
                request_id=request_id,
                position=-1,
                depth=depth,
                task=first.task,
                stage=first.stage,
                token_type="request",
                features=features,
                label=max(row.label for row in token_rows),
                group=group,
            )
        )
    return sequence_rows


def first_exit_by_request(rows: list[FeatureRow], policy: GroupCalibratedPolicy) -> dict[str, FeatureRow]:
    grouped: dict[str, list[FeatureRow]] = defaultdict(list)
    for row in rows:
        grouped[row.request_id].append(row)
    chosen: dict[str, FeatureRow] = {}
    for request_id, request_rows in grouped.items():
        request_rows = sorted(request_rows, key=lambda row: row.depth)
        selected = request_rows[-1]
        for row in request_rows:
            if policy.decide(row).should_exit:
                selected = row
                break
        chosen[request_id] = selected
    return chosen


def summarize_request_rows(selected: list[FeatureRow], total_requests: int | None = None) -> dict[str, Any]:
    if not selected:
        return {
            "avg_depth": 0.0,
            "request_risk": 0.0,
            "coverage": 0.0,
            "requests": 0.0,
            "risk_by_group": {},
            "avg_depth_by_group": {},
        }
    by_group: dict[str, list[FeatureRow]] = defaultdict(list)
    for row in selected:
        by_group[row.group].append(row)
    denominator = total_requests if total_requests is not None else len(selected)
    return {
        "avg_depth": sum(row.depth for row in selected) / len(selected),
        "request_risk": sum(row.label for row in selected) / len(selected),
        "risk": sum(row.label for row in selected) / len(selected),
        "coverage": len(selected) / max(denominator, 1),
        "requests": float(len(selected)),
        "risk_by_group": {
            group: sum(row.label for row in group_rows) / max(len(group_rows), 1)
            for group, group_rows in sorted(by_group.items())
        },
        "avg_depth_by_group": {
            group: sum(row.depth for row in group_rows) / max(len(group_rows), 1)
            for group, group_rows in sorted(by_group.items())
        },
    }


def sequence_frontier_point(rows: list[FeatureRow], policy: GroupCalibratedPolicy) -> dict[str, Any]:
    chosen = first_exit_by_request(rows, policy)
    return summarize_request_rows(list(chosen.values()), total_requests=len(chosen))
