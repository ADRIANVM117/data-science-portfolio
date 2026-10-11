"""Print a no-network capture plan from the public catalog."""
from __future__ import annotations

import argparse
from pathlib import Path

from scenario_portfolio_lab.capture import capture_plan
from scenario_portfolio_lab.catalog import read_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description="No-network capture plan")
    parser.add_argument("--catalog", type=Path, default=Path("config/universe.csv"))
    args = parser.parse_args()
    plan = capture_plan(read_catalog(args.catalog))
    print(f"VERIFIED_USD_INSTRUMENTS={len(plan)}")
    for row in plan: print(f"{row['asset_class']},{row['provider_symbol_candidate']}")


if __name__ == "__main__": main()
