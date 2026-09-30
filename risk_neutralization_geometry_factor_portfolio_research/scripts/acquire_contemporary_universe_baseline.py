"""Acquire immutable source snapshots for the contemporary-universe audit only."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "contemporary_universe_audit"
T0_DATE = "2026-09-25"  # XNYS session; decision is its regular-session open.


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def persist(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as target:
        target.write(payload)
    return hashlib.sha256(payload).hexdigest()


def fetch(url: str, headers: dict[str, str], endpoint: str, parameters: dict[str, str], suffix: str) -> dict[str, object]:
    retrieved_at = utc_now()
    payload: bytes | None = None
    content_type: str | None = None
    try:
        with urlopen(Request(url, headers=headers), timeout=30) as response:
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
        "endpoint": endpoint, "parameters": parameters, "retrieved_at_utc": retrieved_at,
        "request_status": status, "content_type": content_type, "local_raw_file": None,
        "content_sha256": None, "byte_size": None,
    }
    if payload is not None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = RAW_ROOT / f"{endpoint}__{suffix}__{run_id}.bin"
        entry["content_sha256"] = persist(path, payload)
        entry["local_raw_file"] = path.relative_to(PROJECT_ROOT).as_posix()
        entry["byte_size"] = len(payload)
    return entry


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    sec_user_agent = os.environ.get("SEC_EDGAR_USER_AGENT", "").strip()
    if not api_key:
        print("Contemporary audit not run: ALPHAVANTAGE_API_KEY is not configured.")
        return 2
    if not sec_user_agent:
        print("Contemporary audit not run: SEC_EDGAR_USER_AGENT is not configured.")
        return 2
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    av_url = "https://www.alphavantage.co/query?" + urlencode({
        "function": "LISTING_STATUS", "date": T0_DATE, "state": "active", "apikey": api_key,
    })
    entries = [fetch(av_url, {"User-Agent": "contemporary-universe-audit/1.0"}, "LISTING_STATUS", {"date": T0_DATE, "state": "active"}, T0_DATE)]
    time.sleep(0.25)
    entries.append(fetch(
        "https://www.sec.gov/files/company_tickers_exchange.json",
        {"User-Agent": sec_user_agent}, "sec_company_tickers_exchange", {}, "current",
    ))
    manifest_path = RAW_ROOT / "manifest.json"
    prior = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    manifest_path.write_text(json.dumps([*prior, *entries], indent=2) + "\n", encoding="utf-8")
    for entry in entries:
        print(f"{entry['endpoint']}: {entry['request_status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
