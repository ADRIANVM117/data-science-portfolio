"""Read-only PIT prerequisite inspection for the contemporary measurement pilot."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pit_value import market_cap_as_of  # noqa: E402


RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "contemporary_measurement_pilot"
DECISION = pd.Timestamp("2026-09-25 13:30:00+00:00")


def main() -> None:
    manifest = json.loads((RAW_ROOT / "manifest.json").read_text(encoding="utf-8"))
    latest: dict[tuple[str, str], dict[str, object]] = {}
    for entry in manifest:
        if entry["request_status"] == "HTTP_200":
            latest[(str(entry["symbol"]), str(entry["endpoint"]))] = entry
    observations: list[dict[str, object]] = []
    for symbol in sorted({key[0] for key in latest}):
        submissions = json.loads((PROJECT_ROOT / str(latest[(symbol, "submissions")]["local_raw_file"])).read_text())
        recent = submissions["filings"]["recent"]
        acceptance = dict(zip(recent["accessionNumber"], recent["acceptanceDateTime"]))
        facts = json.loads((PROJECT_ROOT / str(latest[(symbol, "companyfacts")]["local_raw_file"])).read_text())
        share_rows = []
        for rows in facts.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding", {}).get("units", {}).values():
            for row in rows:
                if row.get("accn") in acceptance:
                    share_rows.append({"issuer": symbol, "reported_shares": row.get("val"), "accn": row["accn"], "shares_observation_date": row.get("end"), "acceptanceDateTime": acceptance[row["accn"]]})
        shares = pd.DataFrame(share_rows, columns=["issuer", "reported_shares", "accn", "shares_observation_date", "acceptanceDateTime"])
        prices_payload = json.loads((PROJECT_ROOT / str(latest[(symbol, "TIME_SERIES_DAILY_ADJUSTED")]["local_raw_file"])).read_text())
        prices = pd.DataFrame([
            {"price_date": date, "close": row.get("4. close"), "volume": row.get("6. volume"), "split_coefficient": row.get("8. split coefficient")}
            for date, row in prices_payload.get("Time Series (Daily)", {}).items()
        ])
        market_cap = market_cap_as_of(prices, shares, symbol, DECISION)
        prior = prices.loc[pd.to_datetime(prices["price_date"]) < DECISION.tz_localize(None).normalize()]
        usable = (pd.to_numeric(prior["close"], errors="coerce").gt(0) & pd.to_numeric(prior["volume"], errors="coerce").ge(0))
        observations.append({
            "symbol": symbol, "market_cap_status": market_cap.status, "market_cap_reason": market_cap.reason,
            "share_rows_linked_to_acceptance": len(share_rows), "prior_raw_close_volume_rows": int(usable.sum()),
            "zero_volume_prior_rows": int(pd.to_numeric(prior["volume"], errors="coerce").eq(0).sum()),
        })
    print(json.dumps({"decision_time": str(DECISION), "market_cap_status_counts": Counter(x["market_cap_status"] for x in observations), "observations": observations}, indent=2, default=int))


if __name__ == "__main__":
    main()
