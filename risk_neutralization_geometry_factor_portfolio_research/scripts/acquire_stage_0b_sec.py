"""Stage 0B raw-only SEC EDGAR acquisition for five frozen issuers."""

from __future__ import annotations

import hashlib
import json
import os
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


SYMBOLS = ("AAPL", "JPM", "XOM", "UNH", "WMT")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "stage_0b_sec"
TICKER_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
PAUSE_SECONDS = 0.25  # Four requests per second: conservative below SEC's rate limit.


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_once(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        file.write(payload)
    return hashlib.sha256(payload).hexdigest()


def request_raw(
    url: str, endpoint: str, issuer: str, suffix: str, user_agent: str, run_id: str
) -> tuple[dict[str, Any], bytes | None]:
    """Fetch one raw response and return a manifest entry without User-Agent text."""
    retrieved_at_utc = utc_now()
    payload: bytes | None = None
    content_type: str | None = None
    try:
        request = Request(url, headers={"User-Agent": user_agent})
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
        "issuer": issuer,
        "endpoint": endpoint,
        "retrieved_at_utc": retrieved_at_utc,
        "request_status": status,
        "content_type": content_type,
        "local_raw_file": None,
        "content_sha256": None,
        "byte_size": None,
    }
    if payload is not None:
        path = RAW_ROOT / issuer / f"{endpoint}__{suffix}__{run_id}"
        entry["content_sha256"] = write_once(path, payload)
        entry["local_raw_file"] = path.relative_to(PROJECT_ROOT).as_posix()
        entry["byte_size"] = len(payload)
    time.sleep(PAUSE_SECONDS)
    return entry, payload


def append_manifest(entries: list[dict[str, Any]]) -> None:
    manifest_path = RAW_ROOT / "manifest.json"
    prior = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    manifest_path.write_text(json.dumps([*prior, *entries], indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Acquire raw SEC evidence for Stage 0B.")
    parser.add_argument("--symbols", nargs="+", choices=SYMBOLS, default=SYMBOLS)
    parser.add_argument(
        "--cik-override", action="append", default=[], metavar="SYMBOL=CIK",
        help="Explicit, documented issuer identity override for a frozen audit symbol.",
    )
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    user_agent = os.environ.get("SEC_EDGAR_USER_AGENT", "").strip()
    if not user_agent:
        print("Stage 0B acquisition not run: SEC_EDGAR_USER_AGENT is not configured.")
        return 2

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    manifest_entries: list[dict[str, Any]] = []
    overrides: dict[str, str] = {}
    for item in args.cik_override:
        symbol, separator, cik = item.partition("=")
        if not separator or symbol not in SYMBOLS or not cik.isdigit():
            parser.error("--cik-override must have the form SYMBOL=NUMERIC_CIK")
        overrides[symbol] = cik.zfill(10)

    entry, ticker_payload = request_raw(TICKER_URL, "company_tickers_exchange", "SEC", ".json", user_agent, run_id)
    manifest_entries.append(entry)
    if ticker_payload is None:
        append_manifest(manifest_entries)
        print("Ticker-to-CIK source unavailable; no issuer acquisition attempted.")
        return 1

    tickers = json.loads(ticker_payload)
    rows = tickers["data"]
    header = tickers["fields"]
    ticker_index = {row[header.index("ticker")]: row for row in rows}

    for symbol in args.symbols:
        row = ticker_index.get(symbol)
        if row is None:
            manifest_entries.append({
                "issuer": symbol, "endpoint": "identity", "retrieved_at_utc": utc_now(),
                "request_status": "SYMBOL_NOT_IN_SEC_TICKER_SOURCE", "content_type": None,
                "local_raw_file": None, "content_sha256": None, "byte_size": None,
            })
            continue
        ticker_cik = str(row[header.index("cik")]).zfill(10)
        cik = overrides.get(symbol, ticker_cik)
        if symbol in overrides:
            manifest_entries.append({
                "issuer": symbol,
                "endpoint": "identity_resolution",
                "retrieved_at_utc": utc_now(),
                "request_status": "EXPLICIT_CIK_OVERRIDE",
                "content_type": None,
                "local_raw_file": None,
                "content_sha256": None,
                "byte_size": None,
                "ticker_source_cik": ticker_cik,
                "resolved_cik": cik,
            })
        submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        entry, payload = request_raw(submissions_url, "submissions", symbol, ".json", user_agent, run_id)
        manifest_entries.append(entry)
        if payload is None:
            continue
        submission = json.loads(payload)

        filing_inventories = [submission.get("filings", {}).get("recent", {})]
        for historical in submission.get("filings", {}).get("files", []):
            name = historical["name"]
            url = f"https://data.sec.gov/submissions/{name}"
            history_entry, history_payload = request_raw(
                url, "submissions_history", symbol, f"__{name}", user_agent, run_id
            )
            manifest_entries.append(history_entry)
            if history_payload is not None:
                filing_inventories.append(json.loads(history_payload))

        facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        facts_entry, _ = request_raw(facts_url, "companyfacts", symbol, ".json", user_agent, run_id)
        manifest_entries.append(facts_entry)

        candidates = []
        for inventory in filing_inventories:
            candidates.extend(
                (form, accession)
                for form, accession in zip(inventory.get("form", []), inventory.get("accessionNumber", []))
                if form in {"10-K", "10-Q", "10-K/A", "10-Q/A"}
            )
        # Preserve headers for the latest filing of each relevant form only.
        selected: dict[str, str] = {}
        for form, accession in candidates:
            selected.setdefault(form, accession)
        cik_archive = str(int(cik))
        for form, accession in selected.items():
            accession_path = accession.replace("-", "")
            url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik_archive}/"
                f"{accession_path}/{accession}.hdr.sgml"
            )
            header_entry, _ = request_raw(url, "filing_header", symbol, f"__{form}__{accession}.sgml", user_agent, run_id)
            manifest_entries.append(header_entry)

    append_manifest(manifest_entries)
    print(f"Recorded {len(manifest_entries)} Stage 0B requests in {RAW_ROOT / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
