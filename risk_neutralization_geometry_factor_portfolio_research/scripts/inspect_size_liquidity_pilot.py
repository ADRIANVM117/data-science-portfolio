"""Read-only schema/history inspection for the size-liquidity raw pilot.

It deliberately does not calculate a liquidity score, select a window, or
infer that a ticker-only OHLCV response identifies a historical security.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "size_liquidity_pilot"


def main() -> None:
    manifest = json.loads((RAW_ROOT / "manifest.json").read_text(encoding="utf-8"))
    summary: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    observations: list[dict[str, object]] = []
    for entry in manifest:
        if entry["request_status"] != "HTTP_200":
            continue
        payload = json.loads((PROJECT_ROOT / str(entry["local_raw_file"])).read_text(encoding="utf-8"))
        series = payload.get("Time Series (Daily)", {})
        prior = [(date, row) for date, row in series.items() if date < entry["listing_status_date"]]
        usable = 0
        zero_volume = 0
        invalid = 0
        for _, row in prior:
            try:
                close = float(row["4. close"])
                volume = float(row["6. volume"])
            except (KeyError, TypeError, ValueError):
                invalid += 1
                continue
            if close > 0 and volume >= 0:
                usable += 1
            else:
                invalid += 1
            zero_volume += volume == 0
        bucket = summary[str(entry["listing_status_date"])]
        bucket["raw_payloads"] += 1
        bucket["has_prior_raw_close_volume"] += usable > 0
        observations.append({
            "date": entry["listing_status_date"], "exchange": entry["exchange"], "symbol": entry["symbol"],
            "series_rows": len(series), "prior_rows": len(prior), "usable_prior_rows": usable,
            "zero_volume_prior_rows": zero_volume, "invalid_prior_rows": invalid,
            "first_provider_date": min(series) if series else None,
            "last_provider_date": max(series) if series else None,
        })
    print(json.dumps({"by_date": summary, "observations": observations}, indent=2, default=dict))


if __name__ == "__main__":
    main()
