from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from safeloop.evaluation.overhead import OverheadProfile, build_overhead_report
from safeloop.utils.config import load_experiment_config, load_json
from safeloop.utils.io import ensure_dir, write_json


DEFAULT_PROFILES = [
    OverheadProfile(
        name="hidden_only_fast_path",
        brake_ms_per_depth=0.01,
        scheduler_ms_per_token=0.005,
    ),
    OverheadProfile(
        name="hybrid_exit_only_lm_head",
        brake_ms_per_depth=0.01,
        scheduler_ms_per_token=0.01,
        fixed_ms_per_token=0.05,
    ),
    OverheadProfile(
        name="logits_every_depth_cpu_sync",
        brake_ms_per_depth=0.01,
        scheduler_ms_per_token=0.02,
        lm_head_ms_per_depth=0.2,
        cpu_sync_ms_per_depth=0.05,
    ),
]


def _resolve(root: Path, path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else root / raw


def _profiles_from_config(cfg: dict) -> list[OverheadProfile]:
    raw_profiles = cfg.get("overhead_profiles")
    if not raw_profiles:
        return DEFAULT_PROFILES
    return [OverheadProfile.from_dict(item) for item in raw_profiles]


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate overhead-aware risk-compute frontier.")
    parser.add_argument("--config")
    parser.add_argument("--frontier")
    parser.add_argument("--output-dir")
    parser.add_argument("--loop-step-ms", type=float)
    parser.add_argument("--profile")
    parser.add_argument("--full-depth", type=int)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config) if args.config else {}
    root = Path(cfg.get("_project_root", "."))
    frontier_path = args.frontier or cfg.get("frontier_path")
    if not frontier_path:
        raise SystemExit("--frontier or config.frontier_path is required")
    output_dir = args.output_dir or cfg.get("output_dir", "runs/overhead_frontier")
    profile_path = args.profile or cfg.get("profile_path")
    profile = load_json(_resolve(root, profile_path)) if profile_path else {}
    loop_step_ms = args.loop_step_ms or float(profile.get("loop_step_ms", cfg.get("loop_step_ms", 1.0)))
    full_depth = args.full_depth if args.full_depth is not None else cfg.get("full_depth")
    full_depth = int(full_depth) if full_depth is not None else None

    frontier = load_json(_resolve(root, frontier_path))
    report = build_overhead_report(
        frontier=frontier,
        profiles=_profiles_from_config(cfg),
        loop_step_ms=loop_step_ms,
        full_depth=full_depth,
    )
    if profile:
        report["measured_profile"] = profile
    out_dir = ensure_dir(_resolve(root, output_dir))
    write_json(out_dir / "overhead_frontier.json", report)


if __name__ == "__main__":
    main()
