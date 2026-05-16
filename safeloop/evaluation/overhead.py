from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class OverheadProfile:
    """A serving-path cost model for the halting controller."""

    name: str
    brake_ms_per_depth: float = 0.0
    scheduler_ms_per_token: float = 0.0
    cpu_sync_ms_per_depth: float = 0.0
    lm_head_ms_per_depth: float = 0.0
    fixed_ms_per_token: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OverheadProfile":
        return cls(
            name=str(data["name"]),
            brake_ms_per_depth=float(data.get("brake_ms_per_depth", 0.0)),
            scheduler_ms_per_token=float(data.get("scheduler_ms_per_token", 0.0)),
            cpu_sync_ms_per_depth=float(data.get("cpu_sync_ms_per_depth", 0.0)),
            lm_head_ms_per_depth=float(data.get("lm_head_ms_per_depth", 0.0)),
            fixed_ms_per_token=float(data.get("fixed_ms_per_token", 0.0)),
        )

    def overhead_ms(self, avg_depth: float) -> float:
        per_depth = self.brake_ms_per_depth + self.cpu_sync_ms_per_depth + self.lm_head_ms_per_depth
        return self.fixed_ms_per_token + self.scheduler_ms_per_token + avg_depth * per_depth

    def to_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "brake_ms_per_depth": self.brake_ms_per_depth,
            "scheduler_ms_per_token": self.scheduler_ms_per_token,
            "cpu_sync_ms_per_depth": self.cpu_sync_ms_per_depth,
            "lm_head_ms_per_depth": self.lm_head_ms_per_depth,
            "fixed_ms_per_token": self.fixed_ms_per_token,
        }


def infer_full_depth(points: list[dict[str, Any]]) -> int:
    depths = []
    for point in points:
        method = str(point.get("method", ""))
        if method.startswith("fixed_depth_"):
            try:
                depths.append(int(method.rsplit("_", 1)[-1]))
            except ValueError:
                continue
    if depths:
        return max(depths)
    avg_depths = [float(point.get("avg_depth", 0.0)) for point in points]
    return max(1, int(round(max(avg_depths or [1.0]))))


def estimate_overhead_point(
    point: dict[str, Any],
    profile: OverheadProfile,
    loop_step_ms: float,
    full_depth: int,
) -> dict[str, Any]:
    avg_depth = float(point.get("avg_depth", 0.0))
    baseline_ms = full_depth * loop_step_ms
    model_compute_ms = avg_depth * loop_step_ms
    overhead_ms = profile.overhead_ms(avg_depth)
    total_ms = model_compute_ms + overhead_ms
    saved_loop_steps = max(full_depth - avg_depth, 0.0)
    gross_saved_ms = saved_loop_steps * loop_step_ms
    net_saved_ms = baseline_ms - total_ms
    speedup = baseline_ms / total_ms if total_ms > 0 else 0.0
    overhead_loop_equiv = overhead_ms / loop_step_ms if loop_step_ms > 0 else 0.0
    min_saved_depth = overhead_loop_equiv
    return {
        "method": point.get("method", "unknown"),
        "target_risk": point.get("target_risk"),
        "risk": point.get("risk", 0.0),
        "tokens": point.get("tokens", 0.0),
        "profile": profile.name,
        "full_depth": full_depth,
        "avg_depth": avg_depth,
        "saved_loop_steps": saved_loop_steps,
        "loop_step_ms": loop_step_ms,
        "baseline_full_depth_ms_per_token": baseline_ms,
        "model_compute_ms_per_token": model_compute_ms,
        "brake_overhead_ms_per_token": overhead_ms,
        "total_ms_per_token": total_ms,
        "gross_saved_ms_per_token": gross_saved_ms,
        "net_saved_ms_per_token": net_saved_ms,
        "effective_speedup": speedup,
        "overhead_loop_step_equivalent": overhead_loop_equiv,
        "min_saved_depth_to_break_even": min_saved_depth,
        "is_break_even": net_saved_ms > 0.0,
        "overhead_exceeds_one_loop_step": overhead_loop_equiv > 1.0,
        "overhead_share": overhead_ms / total_ms if total_ms > 0 else 0.0,
    }


def build_overhead_report(
    frontier: dict[str, Any],
    profiles: list[OverheadProfile],
    loop_step_ms: float,
    full_depth: int | None = None,
) -> dict[str, Any]:
    points = list(frontier.get("points", []))
    resolved_full_depth = full_depth or infer_full_depth(points)
    estimates = [
        estimate_overhead_point(point, profile, loop_step_ms, resolved_full_depth)
        for profile in profiles
        for point in points
    ]
    return {
        "assumptions": {
            "full_depth": resolved_full_depth,
            "loop_step_ms": loop_step_ms,
            "profiles": [profile.to_dict() for profile in profiles],
        },
        "points": estimates,
        "summary": summarize_estimates(estimates),
    }


def summarize_estimates(estimates: list[dict[str, Any]]) -> dict[str, Any]:
    by_profile: dict[str, list[dict[str, Any]]] = {}
    for estimate in estimates:
        by_profile.setdefault(str(estimate["profile"]), []).append(estimate)
    return {
        profile: {
            "points": float(len(values)),
            "break_even_rate": sum(1 for item in values if item["is_break_even"]) / max(len(values), 1),
            "avg_effective_speedup": sum(float(item["effective_speedup"]) for item in values)
            / max(len(values), 1),
            "avg_overhead_loop_step_equivalent": sum(
                float(item["overhead_loop_step_equivalent"]) for item in values
            )
            / max(len(values), 1),
            "worst_overhead_loop_step_equivalent": max(
                (float(item["overhead_loop_step_equivalent"]) for item in values),
                default=0.0,
            ),
        }
        for profile, values in sorted(by_profile.items())
    }
