"""Acquire a predeclared contemporary PIT Market Cap scale-up sample only."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = PROJECT_ROOT / "data" / "raw" / "contemporary_universe_audit"
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "contemporary_size_scaleup"
T0_DATE = "2026-09-25"
PER_LARGE_EXCHANGE_COHORT = 10
SMALL_EXCHANGE_CENSUS_MAX = 50
EXCLUSIONS = (
    r"\bwarrants?\b", r"\bunits\b|\bunit\s+(?:\d|\()", r"\bcontingent value rights?\b|\btradeable rights?\b",
    r"\bpreferred\b", r"\bdepositary\b", r"\badr\b|\bads\b|american depositary", r"\betf\b|\betn\b", r"closed[ -]end fund",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def digest(row: dict[str, str]) -> str:
    keys = ("symbol", "name", "exchange", "assetType", "ipoDate", "delistingDate", "status")
    return hashlib.sha256("\x1f".join(row[key] for key in keys).encode()).hexdigest()


def ipo_cohort(ipo_date: str) -> str:
    if not ipo_date:
        return "MISSING_IPO_DATE"
    year = int(ipo_date[:4])
    if year <= 2000:
        return "IPO_THROUGH_2000"
    if year <= 2010:
        return "IPO_2001_2010"
    if year <= 2020:
        return "IPO_2011_2020"
    return "IPO_2021_PLUS"


def selections() -> tuple[list[dict[str, str]], int]:
    manifest = json.loads((BASELINE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    listing = next(item for item in reversed(manifest) if item["endpoint"] == "LISTING_STATUS" and item["request_status"] == "HTTP_200")
    sec = next(item for item in reversed(manifest) if item["endpoint"] == "sec_company_tickers_exchange" and item["request_status"] == "HTTP_200")
    with (PROJECT_ROOT / str(listing["local_raw_file"])).open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    sec_payload = json.loads((PROJECT_ROOT / str(sec["local_raw_file"])).read_text(encoding="utf-8"))
    sec_by_ticker: dict[str, list[dict[str, object]]] = {}
    for values in sec_payload["data"]:
        row = dict(zip(sec_payload["fields"], values))
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
    by_exchange: dict[str, list[dict[str, str]]] = {}
    for row in resolved:
        by_exchange.setdefault(row["exchange"], []).append(row)
    chosen: list[dict[str, str]] = []
    for exchange, members in sorted(by_exchange.items()):
        if len(members) <= SMALL_EXCHANGE_CENSUS_MAX:
            selected = sorted(members, key=digest)
            rule = f"census of current identity-resolved candidates on {exchange} (exchange count <= {SMALL_EXCHANGE_CENSUS_MAX})"
            for row in selected:
                row = row.copy(); row["sample_rule"] = rule; row["ipo_cohort"] = ipo_cohort(row["ipoDate"]); chosen.append(row)
            continue
        cohorts: dict[str, list[dict[str, str]]] = {}
        for row in members:
            cohorts.setdefault(ipo_cohort(row["ipoDate"]), []).append(row)
        for cohort, cohort_members in sorted(cohorts.items()):
            rule = f"lowest {PER_LARGE_EXCHANGE_COHORT} canonical-row SHA-256 digests in {exchange}/{cohort} among current identity-resolved candidates"
            for row in sorted(cohort_members, key=digest)[:PER_LARGE_EXCHANGE_COHORT]:
                row = row.copy(); row["sample_rule"] = rule; row["ipo_cohort"] = cohort; chosen.append(row)
    for row in chosen:
        row["cik"] = str(sec_by_ticker[row["symbol"]][0]["cik"]).zfill(10)
        row["selection_digest"] = digest(row)
    return chosen, len(resolved)


def persist(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as target:
        target.write(payload)
    return hashlib.sha256(payload).hexdigest()


def fetch(url: str, headers: dict[str, str], endpoint: str, selection: dict[str, str], suffix: str) -> dict[str, object]:
    retrieved_at = utc_now(); payload: bytes | None = None; content_type: str | None = None
    try:
        with urlopen(Request(url, headers=headers), timeout=30) as response:
            payload = response.read(); status = f"HTTP_{response.status}"; content_type = response.headers.get_content_type()
    except HTTPError as error:
        payload = error.read(); status = f"HTTP_ERROR_{error.code}"; content_type = error.headers.get_content_type() if error.headers else None
    except URLError as error:
        status = f"NETWORK_ERROR_{error.reason}"
    entry: dict[str, object] = {
        "symbol": selection["symbol"], "exchange": selection["exchange"], "ipo_cohort": selection["ipo_cohort"], "cik": selection["cik"],
        "endpoint": endpoint, "selection_digest": selection["selection_digest"], "sample_rule": selection["sample_rule"],
        "retrieved_at_utc": retrieved_at, "request_status": status, "content_type": content_type,
        "local_raw_file": None, "content_sha256": None, "byte_size": None,
    }
    if payload is not None:
        run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = RAW_ROOT / selection["symbol"] / f"{endpoint}__{suffix}__{run}__{uuid.uuid4().hex[:12]}.json"
        entry["content_sha256"] = persist(path, payload)
        entry["local_raw_file"] = path.relative_to(PROJECT_ROOT).as_posix(); entry["byte_size"] = len(payload)
    return entry


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    sec_user_agent = os.environ.get("SEC_EDGAR_USER_AGENT", "").strip()
    if not api_key or not sec_user_agent:
        print("Size scale-up not run: required local credentials are not configured.")
        return 2
    selected, population = selections()
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    selection_path = RAW_ROOT / "selection.json"
    selection_payload = {"t0_date": T0_DATE, "population_identity_resolved": population, "n_selected": len(selected), "selections": selected}
    if selection_path.exists():
        if json.loads(selection_path.read_text(encoding="utf-8")) != selection_payload:
            raise RuntimeError("existing selection.json differs from the declared immutable scale-up selection")
    else:
        selection_path.write_text(json.dumps(selection_payload, indent=2) + "\n", encoding="utf-8")
    manifest_path = RAW_ROOT / "manifest.json"
    entries: list[dict[str, object]] = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    completed = {(str(item["symbol"]), str(item["endpoint"])) for item in entries}
    for index, selection in enumerate(selected, start=1):
        cik = selection["cik"]
        requests = (
            ("submissions", f"https://data.sec.gov/submissions/CIK{cik}.json", {"User-Agent": sec_user_agent}, cik, 0.20),
            ("companyfacts", f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", {"User-Agent": sec_user_agent}, cik, 0.20),
            ("TIME_SERIES_DAILY_ADJUSTED", "https://www.alphavantage.co/query?" + urlencode({"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": selection["symbol"], "outputsize": "full", "apikey": api_key}), {"User-Agent": "contemporary-size-scaleup/1.0"}, T0_DATE, 0.45),
        )
        for endpoint, url, headers, suffix, pause in requests:
            if (selection["symbol"], endpoint) not in completed:
                entries.append(fetch(url, headers, endpoint, selection, suffix))
                manifest_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
                time.sleep(pause)
        print(f"{index}/{len(selected)} {selection['exchange']} {selection['symbol']}: acquired")
    print(f"Completed {len(selected)} selections from population {population}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
