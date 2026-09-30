"""Preserve raw Alpha Vantage LISTING_STATUS evidence for a feasibility audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "listing_status_audit"
DATES = ("2010-01-04", "2014-07-10", "2020-08-31", "2024-01-02")
STATES = ("active", "delisted")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def acquire_one(date: str, state: str, api_key: str, run_id: str) -> dict[str, Any]:
    retrieved_at_utc = utc_now()
    parameters = {"date": date, "state": state}
    query = urllib.parse.urlencode({"function": "LISTING_STATUS", **parameters, "apikey": api_key})
    request = urllib.request.Request(
        f"https://www.alphavantage.co/query?{query}",
        headers={"User-Agent": "listing-status-feasibility-audit/1.0"},
    )
    payload: bytes | None = None
    content_type: str | None = None
    request_status: str
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
            content_type = response.headers.get_content_type()
            request_status = f"HTTP_{response.status}"
    except urllib.error.HTTPError as error:
        payload = error.read()
        content_type = error.headers.get_content_type() if error.headers else None
        request_status = f"HTTP_{error.code}"
    except urllib.error.URLError as error:
        request_status = f"NETWORK_ERROR_{error.reason}"

    entry: dict[str, Any] = {
        "endpoint": "LISTING_STATUS",
        "parameters": parameters,
        "retrieved_at_utc": retrieved_at_utc,
        "request_status": request_status,
        "content_type": content_type,
        "local_raw_file": None,
        "content_sha256": None,
        "byte_size": None,
    }
    if payload is not None:
        RAW_ROOT.mkdir(parents=True, exist_ok=True)
        suffix = ".csv" if content_type in {"text/csv", "text/plain"} else ".bin"
        filename = f"LISTING_STATUS__{date}__{state}__{run_id}{suffix}"
        path = RAW_ROOT / filename
        with path.open("xb") as raw_file:
            raw_file.write(payload)
        entry.update(
            local_raw_file=str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            content_sha256=hashlib.sha256(payload).hexdigest(),
            byte_size=len(payload),
        )
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description="Acquire immutable LISTING_STATUS audit responses.")
    parser.add_argument("--api-key-env", default="ALPHAVANTAGE_API_KEY")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"Listing-status audit not run: environment variable {args.api_key_env} is not configured.")
        return 2

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW_ROOT / "manifest.json"
    manifest = load_manifest(manifest_path)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for date in DATES:
        for state in STATES:
            entry = acquire_one(date, state, api_key, run_id)
            manifest.append(entry)
            print(f"LISTING_STATUS date={date} state={state}: {entry['request_status']}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {len(DATES) * len(STATES)} requests in {manifest_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
