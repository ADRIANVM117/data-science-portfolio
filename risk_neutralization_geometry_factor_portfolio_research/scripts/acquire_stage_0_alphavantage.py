"""Stage 0 raw-only Alpha Vantage acquisition for the frozen audit sample.

This script deliberately stores provider bytes without parsing or transforming
them.  It does not construct research characteristics or portfolio objects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv


SYMBOLS = ("AAPL", "JPM", "XOM", "UNH", "WMT")
ENDPOINTS: tuple[tuple[str, dict[str, str]], ...] = (
    ("TIME_SERIES_DAILY_ADJUSTED", {"outputsize": "full"}),
    ("OVERVIEW", {}),
    ("INCOME_STATEMENT", {}),
    ("BALANCE_SHEET", {}),
    ("CASH_FLOW", {}),
    ("EARNINGS", {}),
    ("SHARES_OUTSTANDING", {}),
)
API_URL = "https://www.alphavantage.co/query"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "stage_0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def persist_raw(raw_path: Path, payload: bytes) -> str:
    """Store response bytes once. Exclusive creation prevents overwrites."""
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("xb") as raw_file:
        raw_file.write(payload)
    return hashlib.sha256(payload).hexdigest()


def acquire_one(
    symbol: str,
    endpoint: str,
    parameters: dict[str, str],
    api_key: str,
    run_id: str,
) -> dict[str, Any]:
    """Request one endpoint and return a credential-free manifest entry."""
    retrieved_at_utc = utc_now()
    query = {"function": endpoint, "symbol": symbol, **parameters, "apikey": api_key}
    request = Request(
        f"{API_URL}?{urlencode(query)}",
        headers={"User-Agent": "stage-0-data-audit/1.0"},
    )

    status: str
    content_type: str | None = None
    payload: bytes | None = None
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

    entry: dict[str, Any] = {
        "symbol": symbol,
        "endpoint": endpoint,
        "parameters": parameters,
        "retrieved_at_utc": retrieved_at_utc,
        "request_status": status,
        "content_type": content_type,
        "local_raw_file": None,
        "content_sha256": None,
        "byte_size": None,
    }
    if payload is not None:
        raw_path = RAW_ROOT / symbol / f"{endpoint}__{run_id}.json"
        entry["content_sha256"] = persist_raw(raw_path, payload)
        entry["local_raw_file"] = raw_path.relative_to(PROJECT_ROOT).as_posix()
        entry["byte_size"] = len(payload)
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description="Acquire immutable Stage 0 Alpha Vantage audit responses.")
    parser.add_argument("--api-key-env", default="ALPHAVANTAGE_API_KEY")
    args = parser.parse_args()

    # Local-only secret loading. Existing process variables retain precedence.
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"Stage 0 acquisition not run: environment variable {args.api_key_env} is not configured.")
        return 2

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW_ROOT / "manifest.json"
    manifest = load_manifest(manifest_path)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    for symbol in SYMBOLS:
        for endpoint, parameters in ENDPOINTS:
            entry = acquire_one(symbol, endpoint, parameters, api_key, run_id)
            manifest.append(entry)
            print(f"{symbol} {endpoint}: {entry['request_status']}")

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {len(SYMBOLS) * len(ENDPOINTS)} requests in {manifest_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
