from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a compact frontier table.")
    parser.add_argument("--frontier", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.frontier).read_text(encoding="utf-8"))
    for point in data.get("points", []):
        print(
            f"{point.get('method','unknown'):>20} "
            f"target={point.get('target_risk','-')} "
            f"avg_depth={point.get('avg_depth',0):.3f} "
            f"risk={point.get('risk',0):.4f}"
        )


if __name__ == "__main__":
    main()

