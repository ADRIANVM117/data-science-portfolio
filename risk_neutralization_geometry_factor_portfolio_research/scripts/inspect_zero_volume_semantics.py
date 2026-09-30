"""Read-only raw-evidence scan for Alpha Vantage positive-close/zero-volume rows."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOTS = (
    PROJECT_ROOT / "data" / "raw" / "stage_0",
    PROJECT_ROOT / "data" / "raw" / "size_liquidity_pilot",
    PROJECT_ROOT / "data" / "raw" / "contemporary_measurement_pilot",
    PROJECT_ROOT / "data" / "raw" / "contemporary_liquidity_stress",
)


def main() -> None:
    seen_hashes: set[str] = set()
    series: list[dict[str, object]] = []
    for root in RAW_ROOTS:
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            continue
        for entry in json.loads(manifest_path.read_text(encoding="utf-8")):
            if entry.get("endpoint") != "TIME_SERIES_DAILY_ADJUSTED" or entry.get("request_status") != "HTTP_200":
                continue
            digest = entry.get("content_sha256")
            if not digest or digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            payload = json.loads((PROJECT_ROOT / entry["local_raw_file"]).read_text(encoding="utf-8"))
            daily = payload.get("Time Series (Daily)", {})
            zeros = [
                {"date": date, "open": row.get("1. open"), "high": row.get("2. high"), "low": row.get("3. low"), "close": row.get("4. close"), "volume": row.get("6. volume")}
                for date, row in daily.items()
                if float(row.get("4. close", "nan")) > 0 and float(row.get("6. volume", "nan")) == 0
            ]
            if zeros:
                series.append({"symbol": entry.get("symbol"), "count": len(zeros), "first": min(zeros, key=lambda x: x["date"]), "last": max(zeros, key=lambda x: x["date"])})
    print(json.dumps({"series_with_positive_close_zero_volume": len(series), "series": series}, indent=2))


if __name__ == "__main__":
    main()
