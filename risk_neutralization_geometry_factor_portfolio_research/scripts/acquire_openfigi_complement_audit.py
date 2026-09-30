"""Preserve raw, unauthenticated OpenFIGI mapping evidence for the AV audit."""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "openfigi_complement_audit"
API_URL = "https://api.openfigi.com/v3/mapping"
BATCHES = {
    "ticker_us": [
        {"idType": "TICKER", "idValue": symbol, "exchCode": "US"}
        for symbol in ("AAPL", "SPY", "ACHR-WS", "ACACU", "CNH")
    ],
    "ticker_us_include_unlisted": [
        {"idType": "TICKER", "idValue": symbol, "exchCode": "US", "includeUnlistedEquities": True}
        for symbol in ("ACB", "ACI", "CNH", "ACHR-WS", "ACACU")
    ],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def acquire_batch(label: str, jobs: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    retrieved_at_utc = utc_now()
    request_body = json.dumps(jobs, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=request_body,
        method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "openfigi-complement-feasibility-audit/1.0"},
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
        "endpoint": "OpenFIGI_v3_mapping",
        "audit_batch": label,
        "jobs": jobs,
        "retrieved_at_utc": retrieved_at_utc,
        "request_status": request_status,
        "content_type": content_type,
        "local_raw_file": None,
        "content_sha256": None,
        "byte_size": None,
    }
    if payload is not None:
        RAW_ROOT.mkdir(parents=True, exist_ok=True)
        path = RAW_ROOT / f"OpenFIGI_v3_mapping__{label}__{run_id}.json"
        with path.open("xb") as raw_file:
            raw_file.write(payload)
        entry.update(
            local_raw_file=str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            content_sha256=hashlib.sha256(payload).hexdigest(),
            byte_size=len(payload),
        )
    return entry


def main() -> int:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW_ROOT / "manifest.json"
    manifest = load_manifest(manifest_path)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for label, jobs in BATCHES.items():
        entry = acquire_batch(label, jobs, run_id)
        manifest.append(entry)
        print(f"OpenFIGI {label}: {entry['request_status']}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {len(BATCHES)} requests in {manifest_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
