"""Acquire a small, deterministic raw OHLCV pilot for the gate-feasibility audit.

This is not universe construction.  It selects one surviving provider-labelled
Stock row per observed Alpha Vantage exchange and audit date, then preserves
only the provider's unmodified TIME_SERIES_DAILY_ADJUSTED response.  The
selection is deterministic before any OHLCV response is inspected.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LISTING_ROOT = PROJECT_ROOT / "data" / "raw" / "listing_status_audit"
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "size_liquidity_pilot"
API_URL = "https://www.alphavantage.co/query"
DATES = ("2010-01-04", "2014-07-10", "2020-08-31", "2024-01-02")
EXCLUSIONS = {
    "warrant": r"\bwarrants?\b",
    "unit": r"\bunits\b|\bunit\s+(?:\d|\()",
    "right": r"\bcontingent value rights?\b|\btradeable rights?\b",
    "preferred": r"\bpreferred\b",
    "depositary": r"\bdepositary\b",
    "adr_ads": r"\badr\b|\bads\b|american depositary",
    "etf_etn": r"\betf\b|\betn\b",
    "closed_end_fund": r"closed[ -]end fund",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def raw_digest(row: dict[str, str]) -> str:
    fields = ("symbol", "name", "exchange", "assetType", "ipoDate", "delistingDate", "status")
    return hashlib.sha256("\x1f".join(row[field] for field in fields).encode("utf-8")).hexdigest()


def is_directly_excluded(row: dict[str, str]) -> bool:
    return any(re.search(pattern, row["name"], flags=re.IGNORECASE) for pattern in EXCLUSIONS.values())


def select_pilot() -> list[dict[str, str]]:
    manifest = json.loads((LISTING_ROOT / "manifest.json").read_text(encoding="utf-8"))
    selections: list[dict[str, str]] = []
    for date in DATES:
        entry = next(
            item for item in manifest
            if item["request_status"] == "HTTP_200"
            and item["parameters"] == {"date": date, "state": "active"}
        )
        with (PROJECT_ROOT / entry["local_raw_file"]).open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        candidates = [row for row in rows if row["assetType"] == "Stock" and not is_directly_excluded(row)]
        for exchange in sorted({row["exchange"] for row in candidates}):
            pool = [row for row in candidates if row["exchange"] == exchange]
            chosen = min(pool, key=raw_digest).copy()
            chosen["listing_status_date"] = date
            chosen["selection_rule"] = "minimum SHA-256 canonical-row digest within date/exchange survivor stratum"
            chosen["selection_digest"] = raw_digest(chosen)
            selections.append(chosen)
    return selections


def persist_raw(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as target:
        target.write(payload)
    return hashlib.sha256(payload).hexdigest()


def acquire(selection: dict[str, str], api_key: str, run_id: str) -> dict[str, object]:
    retrieved_at = utc_now()
    query = {"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": selection["symbol"], "outputsize": "full", "apikey": api_key}
    request = Request(f"{API_URL}?{urlencode(query)}", headers={"User-Agent": "size-liquidity-gate-feasibility-audit/1.0"})
    payload: bytes | None = None
    content_type: str | None = None
    try:
        with urlopen(request, timeout=30) as response:
            payload = response.read()
            status = f"HTTP_{response.status}"
            content_type = response.headers.get_content_type()
    except HTTPError as error:
        payload = error.read()
        status = f"HTTP_ERROR_{error.code}"
        content_type = error.headers.get_content_type() if error.headers else None
    except URLError as error:
        status = f"NETWORK_ERROR_{error.reason}"
    entry: dict[str, object] = {
        "listing_status_date": selection["listing_status_date"],
        "symbol": selection["symbol"], "name": selection["name"], "exchange": selection["exchange"],
        "endpoint": "TIME_SERIES_DAILY_ADJUSTED", "parameters": {"outputsize": "full"},
        "selection_rule": selection["selection_rule"], "selection_digest": selection["selection_digest"],
        "retrieved_at_utc": retrieved_at, "request_status": status, "content_type": content_type,
        "local_raw_file": None, "content_sha256": None, "byte_size": None,
    }
    if payload is not None:
        path = RAW_ROOT / f"{selection['listing_status_date']}__{selection['exchange']}__{selection['symbol']}__{run_id}.json"
        entry["content_sha256"] = persist_raw(path, payload)
        entry["local_raw_file"] = path.relative_to(PROJECT_ROOT).as_posix()
        entry["byte_size"] = len(payload)
    return entry


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        print("Pilot acquisition not run: ALPHAVANTAGE_API_KEY is not configured.")
        return 2
    selections = select_pilot()
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for selection in selections:
        entry = acquire(selection, api_key, run_id)
        manifest.append(entry)
        print(f"{entry['listing_status_date']} {entry['exchange']} {entry['symbol']}: {entry['request_status']}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {len(selections)} raw pilot requests in {manifest_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
