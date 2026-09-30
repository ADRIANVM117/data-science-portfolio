"""Preserve a stratified current-identity pilot for size/liquidity feasibility.

This does not construct a universe or a liquidity metric.  It acquires only
the three raw responses required to test frozen Market Cap prerequisites and
daily raw close/volume availability for preselected current identities.
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
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "contemporary_measurement_pilot"
T0_DATE = "2026-09-25"
PER_EXCHANGE = 3
EXCLUSIONS = (
    r"\bwarrants?\b", r"\bunits\b|\bunit\s+(?:\d|\()", r"\bcontingent value rights?\b|\btradeable rights?\b",
    r"\bpreferred\b", r"\bdepositary\b", r"\badr\b|\bads\b|american depositary", r"\betf\b|\betn\b", r"closed[ -]end fund",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def row_digest(row: dict[str, str]) -> str:
    return hashlib.sha256("\x1f".join(row[key] for key in ("symbol", "name", "exchange", "assetType", "ipoDate", "delistingDate", "status")).encode()).hexdigest()


def resolved_pilot() -> list[dict[str, str]]:
    manifest = json.loads((BASELINE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    listing = next(item for item in reversed(manifest) if item["endpoint"] == "LISTING_STATUS" and item["request_status"] == "HTTP_200")
    sec = next(item for item in reversed(manifest) if item["endpoint"] == "sec_company_tickers_exchange" and item["request_status"] == "HTTP_200")
    with (PROJECT_ROOT / str(listing["local_raw_file"])).open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    sec_payload = json.loads((PROJECT_ROOT / str(sec["local_raw_file"])).read_text(encoding="utf-8"))
    sec_rows = [dict(zip(sec_payload["fields"], row)) for row in sec_payload["data"]]
    sec_by_ticker: dict[str, list[dict[str, object]]] = {}
    for row in sec_rows:
        sec_by_ticker.setdefault(str(row["ticker"]), []).append(row)
    candidate_count: dict[str, int] = {}
    for row in rows:
        if row["assetType"] == "Stock" and not any(re.search(pattern, row["name"], re.I) for pattern in EXCLUSIONS):
            candidate_count[row["symbol"]] = candidate_count.get(row["symbol"], 0) + 1
    resolved = [
        row for row in rows
        if row["assetType"] == "Stock" and not any(re.search(pattern, row["name"], re.I) for pattern in EXCLUSIONS)
        and candidate_count[row["symbol"]] == 1 and len(sec_by_ticker.get(row["symbol"], [])) == 1
    ]
    result: list[dict[str, str]] = []
    for exchange in sorted({row["exchange"] for row in resolved}):
        for row in sorted((item for item in resolved if item["exchange"] == exchange), key=row_digest)[:PER_EXCHANGE]:
            selected = row.copy()
            selected["cik"] = str(sec_by_ticker[row["symbol"]][0]["cik"]).zfill(10)
            selected["selection_digest"] = row_digest(row)
            selected["selection_rule"] = f"lowest {PER_EXCHANGE} canonical-row SHA-256 digests per AV exchange among current uniquely SEC-mapped candidates"
            result.append(selected)
    return result


def persist(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as target:
        target.write(payload)
    return hashlib.sha256(payload).hexdigest()


def fetch(url: str, headers: dict[str, str], endpoint: str, selection: dict[str, str], suffix: str) -> dict[str, object]:
    retrieved_at = utc_now()
    payload: bytes | None = None
    content_type: str | None = None
    try:
        with urlopen(Request(url, headers=headers), timeout=30) as response:
            payload = response.read(); status = f"HTTP_{response.status}"; content_type = response.headers.get_content_type()
    except HTTPError as error:
        payload = error.read(); status = f"HTTP_ERROR_{error.code}"; content_type = error.headers.get_content_type() if error.headers else None
    except URLError as error:
        status = f"NETWORK_ERROR_{error.reason}"
    entry: dict[str, object] = {
        "symbol": selection["symbol"], "exchange": selection["exchange"], "cik": selection["cik"], "endpoint": endpoint,
        "selection_digest": selection["selection_digest"], "selection_rule": selection["selection_rule"], "retrieved_at_utc": retrieved_at,
        "request_status": status, "content_type": content_type, "local_raw_file": None, "content_sha256": None, "byte_size": None,
    }
    if payload is not None:
        run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = RAW_ROOT / selection["symbol"] / f"{endpoint}__{suffix}__{run}.json"
        entry["content_sha256"] = persist(path, payload)
        entry["local_raw_file"] = path.relative_to(PROJECT_ROOT).as_posix(); entry["byte_size"] = len(payload)
    return entry


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    sec_user_agent = os.environ.get("SEC_EDGAR_USER_AGENT", "").strip()
    if not api_key or not sec_user_agent:
        print("Measurement pilot not run: required local credentials are not configured.")
        return 2
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for selection in resolved_pilot():
        cik = selection["cik"]
        endpoints = (
            ("submissions", f"https://data.sec.gov/submissions/CIK{cik}.json", {"User-Agent": sec_user_agent}, cik),
            ("companyfacts", f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", {"User-Agent": sec_user_agent}, cik),
            ("TIME_SERIES_DAILY_ADJUSTED", "https://www.alphavantage.co/query?" + urlencode({"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": selection["symbol"], "outputsize": "full", "apikey": api_key}), {"User-Agent": "contemporary-universe-audit/1.0"}, T0_DATE),
        )
        for endpoint, url, headers, suffix in endpoints:
            entry = fetch(url, headers, endpoint, selection, suffix)
            entries.append(entry)
            print(f"{selection['exchange']} {selection['symbol']} {endpoint}: {entry['request_status']}")
            time.sleep(0.25)
    manifest_path = RAW_ROOT / "manifest.json"
    prior = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    manifest_path.write_text(json.dumps([*prior, *entries], indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
