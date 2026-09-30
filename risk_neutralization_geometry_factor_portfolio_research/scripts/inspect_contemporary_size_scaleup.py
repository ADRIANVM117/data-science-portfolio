"""Read-only coverage and cross-sectional audit for frozen PIT Market Cap."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "contemporary_size_scaleup"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pit_value import market_cap_as_of  # noqa: E402


DECISION = pd.Timestamp("2026-09-25 13:30:00+00:00")
REFERENCE_LEVELS = (100_000_000, 1_000_000_000, 10_000_000_000)
STATUS_MAP = {
    "VALID": "SIZE_VALID",
    "MISSING_SHARES": "SIZE_UNKNOWN_MISSING_SHARES",
    "SPLIT_BASIS_MISMATCH": "SIZE_UNKNOWN_SPLIT_BASIS",
    "MISSING_PRICE": "SIZE_UNKNOWN_MISSING_PRICE",
    "AMBIGUOUS_SHARES": "SIZE_UNKNOWN_OTHER",
}


def load_json(relative_path: str) -> dict[str, object]:
    return json.loads((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))


def verified_payload(entry: dict[str, object]) -> bool:
    path = PROJECT_ROOT / str(entry["local_raw_file"])
    return path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == entry["content_sha256"]


def shares_from_payloads(symbol: str, facts: dict[str, object], submissions: dict[str, object]) -> pd.DataFrame:
    recent = submissions.get("filings", {}).get("recent", {})
    acceptance = dict(zip(recent.get("accessionNumber", []), recent.get("acceptanceDateTime", [])))
    rows: list[dict[str, object]] = []
    units = facts.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding", {}).get("units", {})
    for unit_rows in units.values():
        for row in unit_rows:
            if row.get("accn") in acceptance:
                rows.append({
                    "issuer": symbol, "reported_shares": row.get("val"), "accn": row["accn"],
                    "shares_observation_date": row.get("end"), "acceptanceDateTime": acceptance[row["accn"]],
                })
    return pd.DataFrame(rows, columns=["issuer", "reported_shares", "accn", "shares_observation_date", "acceptanceDateTime"])


def prices_from_payload(payload: dict[str, object]) -> pd.DataFrame:
    rows = [
        {"price_date": date, "close": values.get("4. close"), "split_coefficient": values.get("8. split coefficient")}
        for date, values in payload.get("Time Series (Daily)", {}).items()
    ]
    return pd.DataFrame(rows, columns=["price_date", "close", "split_coefficient"])


def audit() -> dict[str, object]:
    selection = json.loads((RAW_ROOT / "selection.json").read_text(encoding="utf-8"))
    manifest = json.loads((RAW_ROOT / "manifest.json").read_text(encoding="utf-8"))
    entries = {(str(x["symbol"]), str(x["endpoint"])): x for x in manifest}
    records: list[dict[str, object]] = []
    hash_failures = [str(x["local_raw_file"]) for x in manifest if x["local_raw_file"] and not verified_payload(x)]
    for selected in selection["selections"]:
        symbol = str(selected["symbol"])
        needed = [entries.get((symbol, endpoint)) for endpoint in ("submissions", "companyfacts", "TIME_SERIES_DAILY_ADJUSTED")]
        missing_endpoint = next((endpoint for endpoint, entry in zip(("submissions", "companyfacts", "TIME_SERIES_DAILY_ADJUSTED"), needed) if entry is None or entry["request_status"] != "HTTP_200"), None)
        base = {"symbol": symbol, "exchange": selected["exchange"], "ipo_cohort": selected["ipo_cohort"], "cik": selected["cik"]}
        if missing_endpoint:
            records.append({**base, "status": "SIZE_UNKNOWN_OTHER", "reason": f"raw acquisition unavailable for {missing_endpoint}", "market_cap": None})
            continue
        try:
            submissions, facts, prices = (load_json(str(entry["local_raw_file"])) for entry in needed)
            registered_tickers = submissions.get("tickers", [])
            if len(registered_tickers) != 1:
                records.append({
                    **base, "status": "SIZE_UNKNOWN_IDENTITY_OR_MAPPING",
                    "reason": "SEC registrant lists multiple tickers; no evidence links issuer-level DEI shares to the selected security",
                    "market_cap": None, "registrant_ticker_count": len(registered_tickers),
                })
                continue
            result = market_cap_as_of(prices_from_payload(prices), shares_from_payloads(symbol, facts, submissions), symbol, DECISION)
            records.append({
                **base, "status": STATUS_MAP.get(result.status, "SIZE_UNKNOWN_OTHER"), "reason": result.reason,
                "market_cap": result.market_cap, "raw_price_date": str(result.raw_price_date) if result.raw_price_date is not None else None,
                "raw_close": result.raw_close, "reported_shares": result.shares_value,
                "shares_accession": result.shares_source_accession, "shares_reference_date": str(result.shares_reference_date) if result.shares_reference_date is not None else None,
                "shares_pit_eligibility": str(result.shares_pit_eligibility) if result.shares_pit_eligibility is not None else None,
                "split_event_date": str(result.split_event_date) if result.split_event_date is not None else None,
                "split_coefficient": result.split_coefficient,
            })
        except Exception as error:  # Audit parsing failures remain unknown, never repaired.
            records.append({**base, "status": "SIZE_UNKNOWN_OTHER", "reason": f"parse/measurement exception: {type(error).__name__}", "market_cap": None})
    frame = pd.DataFrame(records)
    valid = frame.loc[frame["status"].eq("SIZE_VALID") & frame["market_cap"].notna()].copy()
    quantiles = valid["market_cap"].quantile([0, .01, .05, .10, .25, .50, .75, .90, .95, .99, 1.0]).to_dict()
    return {
        "decision_time": str(DECISION), "population_identity_resolved": selection["population_identity_resolved"],
        "n_candidates_audited": len(frame), "raw_payload_hash_failures": hash_failures,
        "status_counts": dict(Counter(frame["status"])),
        "status_by_exchange": frame.groupby(["exchange", "status"], observed=True).size().rename("count").reset_index().to_dict("records"),
        "status_by_ipo_cohort": frame.groupby(["ipo_cohort", "status"], observed=True).size().rename("count").reset_index().to_dict("records"),
        "market_cap_distribution": {str(key): value for key, value in quantiles.items()},
        "reference_level_counts": {str(level): int(valid["market_cap"].lt(level).sum()) for level in REFERENCE_LEVELS},
        "lower_extremes": valid.nsmallest(5, "market_cap").to_dict("records"),
        "upper_extremes": valid.nlargest(5, "market_cap").to_dict("records"),
        "records": records,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, default=str))
