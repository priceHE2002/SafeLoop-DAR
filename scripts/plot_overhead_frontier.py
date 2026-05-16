from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a compact overhead-aware frontier table.")
    parser.add_argument("--overhead-frontier", required=True)
    parser.add_argument("--profile", default="")
    args = parser.parse_args()

    data = json.loads(Path(args.overhead_frontier).read_text(encoding="utf-8"))
    points = data.get("points", [])
    if args.profile:
        points = [point for point in points if point.get("profile") == args.profile]

    for point in points:
        print(
            f"{point.get('profile','unknown'):>28} "
            f"{point.get('method','unknown'):>20} "
            f"target={point.get('target_risk','-')} "
            f"avg_depth={point.get('avg_depth',0):.3f} "
            f"risk={point.get('risk',0):.4f} "
            f"overhead={point.get('overhead_loop_step_equivalent',0):.3f}x_loop "
            f"speedup={point.get('effective_speedup',0):.3f} "
            f"break_even={point.get('is_break_even', False)}"
        )


if __name__ == "__main__":
    main()
