"""Preserve a predeclared current-exchange stress sample for liquidity audit.

The complete current BATS and NYSE ARCA resolved strata are selected before
OHLCV is read. The script acquires only raw daily responses; it does not create
a universe, score liquidity, or choose a threshold.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = PROJECT_ROOT / "data" / "raw" / "contemporary_universe_audit"
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "contemporary_liquidity_stress"
EXCHANGES = {"BATS", "NYSE ARCA"}
EXCLUSIONS = (r"\bwarrants?\b", r"\bunits\b|\bunit\s+(?:\d|\()", r"\bcontingent value rights?\b|\btradeable rights?\b", r"\bpreferred\b", r"\bdepositary\b", r"\badr\b|\bads\b|american depositary", r"\betf\b|\betn\b", r"closed[ -]end fund")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def selected_symbols() -> list[dict[str, str]]:
    manifest = json.loads((BASELINE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    listing = next(x for x in reversed(manifest) if x["endpoint"] == "LISTING_STATUS" and x["request_status"] == "HTTP_200")
    sec = next(x for x in reversed(manifest) if x["endpoint"] == "sec_company_tickers_exchange" and x["request_status"] == "HTTP_200")
    with (PROJECT_ROOT / str(listing["local_raw_file"])).open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    sec_payload = json.loads((PROJECT_ROOT / str(sec["local_raw_file"])).read_text())
    sec_rows = [dict(zip(sec_payload["fields"], row)) for row in sec_payload["data"]]
    sec_counts: dict[str, int] = {}
    for row in sec_rows:
        sec_counts[str(row["ticker"])] = sec_counts.get(str(row["ticker"]), 0) + 1
    candidate_counts: dict[str, int] = {}
    for row in rows:
        if row["assetType"] == "Stock" and not any(re.search(p, row["name"], re.I) for p in EXCLUSIONS):
            candidate_counts[row["symbol"]] = candidate_counts.get(row["symbol"], 0) + 1
    return sorted([
        {"symbol": row["symbol"], "exchange": row["exchange"], "selection_rule": "all current uniquely-SEC-mapped candidates in complete BATS and NYSE ARCA strata"}
        for row in rows
        if row["exchange"] in EXCHANGES and row["assetType"] == "Stock"
        and not any(re.search(p, row["name"], re.I) for p in EXCLUSIONS)
        and candidate_counts[row["symbol"]] == 1 and sec_counts.get(row["symbol"]) == 1
    ], key=lambda x: (x["exchange"], x["symbol"]))


def persist(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as target:
        target.write(payload)
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        print("Liquidity stress acquisition not run: ALPHAVANTAGE_API_KEY is not configured.")
        return 2
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    entries = []
    for item in selected_symbols():
        retrieved_at = utc_now(); payload = None; content_type = None
        url = "https://www.alphavantage.co/query?" + urlencode({"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": item["symbol"], "outputsize": "full", "apikey": api_key})
        try:
            with urlopen(Request(url, headers={"User-Agent": "liquidity-representation-audit/1.0"}), timeout=30) as response:
                payload = response.read(); status = f"HTTP_{response.status}"; content_type = response.headers.get_content_type()
        except HTTPError as error:
            payload = error.read(); status = f"HTTP_ERROR_{error.code}"; content_type = error.headers.get_content_type() if error.headers else None
        except URLError as error:
            status = f"NETWORK_ERROR_{error.reason}"
        entry = {**item, "endpoint": "TIME_SERIES_DAILY_ADJUSTED", "retrieved_at_utc": retrieved_at, "request_status": status, "content_type": content_type, "local_raw_file": None, "content_sha256": None, "byte_size": None}
        if payload is not None:
            run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            path = RAW_ROOT / f"{item['exchange']}__{item['symbol']}__{run}.json"
            entry["content_sha256"] = persist(path, payload); entry["local_raw_file"] = path.relative_to(PROJECT_ROOT).as_posix(); entry["byte_size"] = len(payload)
        entries.append(entry); print(f"{item['exchange']} {item['symbol']}: {status}"); time.sleep(0.25)
    manifest_path = RAW_ROOT / "manifest.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    manifest_path.write_text(json.dumps([*existing, *entries], indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
